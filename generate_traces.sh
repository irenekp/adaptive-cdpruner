#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# --- Activate conda env 'cdpruner' in a non-interactive shell ---
if command -v conda >/dev/null 2>&1; then
    # Initialize conda for this *shell only* (no need to run `conda init`)
    eval "$(conda shell.bash hook)"
    conda activate cdpruner
else
    echo "Error: conda not found in PATH" >&2
    exit 1
fi


export DATA_ROOT="/storage/ice1/5/3/istephen3/playground/data"

Q="$DATA_ROOT/eval/pope/llava_pope_test.jsonl"
IMG="$DATA_ROOT/eval/pope/val2014/val2014"
ANN="$DATA_ROOT/eval/pope/answers"

ANS_DIR="$DATA_ROOT/eval/pope/answers"

OUT_DIR="traces/pope_superserve"
mkdir -p "${OUT_DIR}"


python make_traces.py \
  --pope_jsonl "${Q}" \
  --images_root "${IMG}" \
  --ann_dir "${ANN}" \
  --outdir "${OUT_DIR}" \
  --duration 300 \
  --seed 0 \
  --max_new_tokens 32 \
  --deadline_ms 300 \
  \
  --const_qps 3000 \
  \
  --bursty_base 1500 \
  --bursty_variants 2950 4900 5550 \
  --bursty_cv2 2 4 8 \
  \
  --tv_lambda1 2500 \
  --tv_lambda2_list 4800 6800 7800 \
  --tv_accel_list 250 500 5000 \
  --tv_cv2a 8

