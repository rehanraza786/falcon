from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from falcon.llm import Generation


@dataclass
class HFConfig:
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    temperature: float = 0.7
    max_new_tokens: int = 256
    do_sample: bool = True
    enable_scoring: bool = True


class HFTransformersAdapter:
    def __init__(self, cfg: HFConfig):
        self.cfg = cfg

        if cfg.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = cfg.device

        if cfg.dtype == "float16":
            torch_dtype = torch.float16
        elif cfg.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(cfg.model_id)

        # Detect whether model is encoder-decoder (e.g. T5/FLAN-T5)
        self.is_seq2seq = bool(getattr(model_config, "is_encoder_decoder", False))

        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch_dtype,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch_dtype,
            )

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> Generation:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.is_seq2seq:
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal models, generated output includes prompt tokens
            input_len = inputs["input_ids"].shape[1]
            gen_tokens = outputs[0][input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return Generation(text=text, meta={"model_id": self.cfg.model_id})

    def score_text(self, text: str) -> Optional[float]:
        if not self.cfg.enable_scoring:
            return None

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        try:
            with torch.no_grad():
                if self.is_seq2seq:
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                else:
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            return float(-loss.item()) if loss is not None else None
        except Exception:
            return None

    def score_tokens(self, text: str) -> Optional[List[float]]:
        # Optional token-level scoring not implemented here
        return None