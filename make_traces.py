#!/usr/bin/env python3
"""
make_traces.py — batch generator for POPE traces (constant / bursty / time-varying)
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pope_jsonl", type=Path, required=True)
    ap.add_argument("--images_root", type=Path, required=True)
    ap.add_argument("--ann_dir", type=Path, required=False, default=None)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--deadline_ms", type=int, default=300)
    ap.add_argument("--pruning_ratio", type=float, default=None)

    # grid defaults
    ap.add_argument("--const_qps", type=float, nargs="*", default=[3000.0])
    ap.add_argument("--bursty_base", type=float, default=1500.0)
    ap.add_argument("--bursty_variants", type=float, nargs="*", default=[2950.0, 4900.0, 5550.0])
    ap.add_argument("--bursty_cv2", type=float, nargs="*", default=[2.0, 4.0, 8.0])
    ap.add_argument("--tv_lambda1", type=float, default=2500.0)
    ap.add_argument("--tv_lambda2_list", type=float, nargs="*", default=[4800.0, 6800.0, 7800.0])
    ap.add_argument("--tv_accel_list", type=float, nargs="*", default=[250.0, 500.0, 5000.0])

    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    gen = Path(__file__).parent / "gen_pope_traces.py"

    def common_args():
        arr = ["--pope_jsonl", args.pope_jsonl, "--images_root", args.images_root]
        if args.ann_dir is not None:
            arr += ["--ann_dir", args.ann_dir]
        arr += ["--seed", args.seed, "--max_new_tokens", args.max_new_tokens, "--deadline_ms", args.deadline_ms]
        if args.pruning_ratio is not None:
            arr += ["--pruning_ratio", args.pruning_ratio]
        return arr

    # Constant
    for qps in args.const_qps:
        out = outdir / f"pope_constant_qps{int(qps)}.jsonl"
        cmd = [sys.executable, gen, "--mode", "constant", "--qps", qps, "--duration", args.duration] + common_args() + ["--out", out]
        run(cmd)

    # Bursty
    for vqps in args.bursty_variants:
        for cv2 in args.bursty_cv2:
            out = outdir / f"pope_bursty_base{int(args.bursty_base)}_var{int(vqps)}_cv2{int(cv2)}.jsonl"
            cmd = [sys.executable, gen, "--mode", "bursty",
                   "--base_qps", args.bursty_base, "--variant_qps", vqps, "--cv2", cv2,
                   "--duration", args.duration] + common_args() + ["--out", out]
            run(cmd)

    # Time-varying
    for lam2 in args.tv_lambda2_list:
        for acc in args.tv_accel_list:
            out = outdir / f"pope_timevary_l1{int(args.tv_lambda1)}_l2{int(lam2)}_a{int(acc)}.jsonl"
            cmd = [sys.executable, gen, "--mode", "timevary",
                   "--lambda1", args.tv_lambda1, "--lambda2", lam2, "--accel", acc,
                   "--duration", args.duration] + common_args() + ["--out", out]
            run(cmd)

if __name__ == "__main__":
    main()
