from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from falcon.llm import Generation, LLM


@dataclass
class HFConfig:
    model_id: str
    device: str = "auto"  # auto|cpu|cuda
    dtype: str = "auto"   # auto|float16|bfloat16|float32
    temperature: float = 0.7
    max_new_tokens: int = 256
    do_sample: bool = True
    enable_scoring: bool = True


class HFTransformersAdapter(LLM):
    """HuggingFace local adapter with scoring support."""

    def __init__(self, cfg: HFConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        if cfg.dtype == "auto":
            torch_dtype = None
        elif cfg.dtype == "float16":
            torch_dtype = torch.float16
        elif cfg.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {cfg.dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def generate(self, prompt: str) -> Generation:
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model.generate(
            **enc,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            do_sample=self.cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out[0][enc["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return Generation(text=text, meta={"provider": "hf", "token_logprobs": None})

    @torch.inference_mode()
    def score_text(self, text: str) -> Optional[float]:
        if not self.cfg.enable_scoring:
            return None

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        logits = self.model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        mean_lp = token_lp.mean().item()

        # exp(mean logprob) is a geometric-mean probability-like quantity.
        # Do NOT clamp to 1.0; caller will normalize weights.
        val = float(torch.exp(torch.tensor(mean_lp)).item())
        return float(max(1e-12, val))

    @torch.inference_mode()
    def score_tokens(self, text: str) -> Optional[List[float]]:
        if not self.cfg.enable_scoring:
            return None

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        logits = self.model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)[0]
        return [float(x) for x in token_lp.detach().cpu().tolist()]
