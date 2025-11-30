#!/usr/bin/env bash
set -euo pipefail
for cv2 in 1.5 2 2.5; do
    for qps in 5 8 10; do
        /storage/ice1/5/3/istephen3/conda-envs/cdpruner/bin/python gen_gqa_traces.py \
            --mode gamma-bursty \
            --qps "$qps" \
            --duration 60.0 \
            --gqa_jsonl /storage/ice1/5/3/istephen3/playground/data/gqa/processed/gqa_balanced_val_rawpaths.jsonl \
            --seed 0 \
            --max_new_tokens 1 \
            --deadline_ms 1200 \
            --max-reqs 200 \
            --cv2 "$cv2" \
            --out "traces/fig9/gqa_gamma_bursty_qps_${qps}_${cv2}_200.jsonl"

        /storage/ice1/5/3/istephen3/conda-envs/cdpruner/bin/python run_trial_fig9.py \
            --trace-path "traces/fig9/gqa_gamma_bursty_qps_${qps}_${cv2}_200.jsonl" \
            --tag "a10080gb_gamma_${qps}_${cv2}_100_v2"

        cp "results/fig9/summary.json" "results/fig9/summary_${qps}_${cv2}.json"
    done
done