#!/usr/bin/env bash
set -euo pipefail

MODES=("hard" "soft")
DATASETS=("truthfulqa" "strategyqa")
CONFIG_FILE="config.yaml"
OUTPUT_DIR="./outputs"

mkdir -p "$OUTPUT_DIR"

echo "------------------------------------------------"
echo "   FALCON: Starting Automated Ablation Study      "
echo "------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "[*] Running FALCON in $mode mode on $dataset..."

    OUT_FILE="$OUTPUT_DIR/${dataset}_${mode}_results.json"
    TMP_CFG="$OUTPUT_DIR/tmp_${dataset}_${mode}_config.yaml"

    python - <<'PY' "$CONFIG_FILE" "$TMP_CFG" "$dataset"
import sys, yaml
src, dst, ds = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
cfg.setdefault("eval", {})
cfg["eval"]["dataset"] = ds

# ALWAYS override split so base config can't break dataset loads
if ds == "strategyqa":
    cfg["eval"]["split"] = "test"
elif ds == "truthfulqa":
    cfg["eval"]["split"] = "validation"
else:
    cfg["eval"]["split"] = cfg["eval"].get("split", "validation")

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    python main.py \
      --mode eval \
      --config "$TMP_CFG" \
      --logic "$mode" \
      --out "$OUT_FILE"

    rm -f "$TMP_CFG"
    echo "[+] Saved results to $OUT_FILE"
    echo "------------------------------------------------"
  done
done

echo "Ablation Study Complete. All results in $OUTPUT_DIR"
