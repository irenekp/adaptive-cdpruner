#!/usr/bin/env python3
"""
gen_pope_traces.py — CDPruner/POPE trace generator (bursty & time-varying)
-------------------------------------------------------------------------

Emits JSONL where each line matches your consumer.py expectations:

{
  "arrival_time": <float seconds since t0>,
  "image_path": "<path to image file>",
  "question": "<question text>",
  "max_new_tokens": <int>,
  "deadline_ms": <int>,
  "pruning_ratio": <float | null>,        # optional
  "pope_category": "<str | null>",        # optional
  "gt_answer": "<yes|no|unknown>"         # extra (kept for evaluation)
}

Arrival modes:
- constant: fixed QPS
- bursty:   base constant stream + gamma-bursty variant (control via CV^2)
- timevary: linear change in mean rate with acceleration a (q/s^2)
"""
from __future__ import annotations
import os
data_root = os.path.expandvars("$DATA_ROOT")
pope_jsonl = os.path.join(data_root, "eval", "pope", "llava_pope_test.jsonl")
img_path = os.path.join(data_root, "eval", "pope", "val2014", "val2014")
ann_path = os.path.join(data_root, "eval", "pope", "coco")
# -------------------------------------------------------------------------
import argparse, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------- POPE loading ----------------

@dataclass
class PopeItem:
    image_path: str
    question: str
    category: Optional[str]
    image_id: Optional[int]
    question_id: Optional[int]
    raw: dict

def _read_jsonl(p: Path) -> Iterator[dict]:
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_pope_items(pope_jsonl: Path, images_root: Path) -> List[PopeItem]:
    items: List[PopeItem] = []
    for rec in _read_jsonl(pope_jsonl):
        q = rec.get("text") or rec.get("question") or rec.get("query") or ""
        qid = rec.get("question_id", None)
        cat = rec.get("category")
        img = rec.get("image") or rec.get("file_name") or ""
        if isinstance(img, str) and img:
            img_name = Path(img).name
        else:
            # derive from image_id if present
            iid = rec.get("image_id")
            if isinstance(iid, (int, float)):
                img_name = f"COCO_val2014_{int(iid):012d}.jpg"
            else:
                continue
        image_id = rec.get("image_id") if isinstance(rec.get("image_id"), int) else None
        items.append(PopeItem(
            image_path=(images_root / img_name).as_posix(),
            question=str(q),
            category=str(cat) if cat is not None else None,
            image_id=image_id,
            question_id=qid,
            raw=rec
        ))
    if not items:
        raise RuntimeError(f"No items found in {pope_jsonl}")
    return items

def _load_ann_files(ann_dir: Optional[Path]) -> List[dict]:
    out: List[dict] = []
    if not ann_dir:
        return out
    for name in ("llava-v1.5-13b.jsonl", "dummy.jsonl"):
        p = ann_dir / name
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.append(json.loads(line))
            except Exception:
                raise RuntimeError(f"Failed to read annotation file: {p}")
    return out

def build_gt_lookup(items: List[PopeItem], ann_dir: Optional[Path]) -> Dict[Tuple[Optional[int], str], str]:
    def norm(q: str)->str:
        return " ".join(q.lower().strip().split())
    table: Dict[int, str] = {}
    # merge annotations
    for rec in _load_ann_files(ann_dir):
        # use question_id as the integer key, since there's no image_id
        iid = rec.get("question_id")

        # take the actual question text from 'prompt' (first line only)
        prompt = rec.get("prompt") or rec.get("text") or ""
        q = prompt.split("\n", 1)[0].strip()

        # derive a yes/no label from the full-sentence answer
        raw_ans = (rec.get("label") or
                rec.get("answer") or
                rec.get("text") or "").strip().lower()
        if "yes" in raw_ans.lower():
            a = "yes"
        elif "no" in raw_ans.lower():
            a = "no"
        else:
            continue

        if isinstance(iid, int):
            table[iid] = a
    return table


# ---------------- Arrival processes ----------------

def gen_constant(qps: float, duration: float) -> List[float]:
    n = max(0, int(qps*duration))
    if n==0: return []
    dt = 1.0/max(qps,1e-9)
    return [i*dt for i in range(n)]

def gen_gamma_bursty(mean_qps: float, cv2: float, duration: float, rng: random.Random) -> List[float]:
    if mean_qps<=0: return []
    mean = 1.0/mean_qps
    k = 1.0/max(cv2,1e-9)          # shape
    theta = mean/k                 # scale
    ts: List[float] = []
    t=0.0
    while t<duration:
        ia = rng.gammavariate(k, theta)
        t += ia
        if t<duration: ts.append(t)
    return ts

def _poisson_knuth(rng: random.Random, mean: float) -> int:
    if mean<=0: return 0
    if mean<20:
        L = math.exp(-mean); k=0; p=1.0
        while p>L:
            k+=1; p*=rng.random()
        return k-1
    # normal approx
    return max(0, int(rng.gauss(mean, math.sqrt(mean)) + 0.5))

def gen_timevary(
    l1: float,
    l2: float,
    accel: float,
    duration: float,
    rng: random.Random,
    cv2a: float,
) -> List[float]:
    """
    Time-varying Gamma arrival process with *fixed* CV^2 = cv2a.

    At time t, we define:
      lambda(t) = clamp(l1 + accel * t, [min(l1,l2), max(l1,l2)])
      mu(t)     = 1 / lambda(t)
      k         = 1 / cv2a
      theta(t)  = mu(t) / k

    Then we sample inter-arrival times:
      T ~ Gamma(k, theta(t))
    """
    low, high = (l1, l2) if l1 <= l2 else (l2, l1)
    t = 0.0
    ts: List[float] = []
    k = 1.0 / max(cv2a, 1e-9)

    while t < duration:
        lam = l1 + accel * t
        lam = min(max(lam, low), high)
        if lam <= 0.0:
            break

        mu = 1.0 / lam          # mean inter-arrival
        theta = mu / k          # scale

        ia = rng.gammavariate(k, theta)
        t += ia
        if t < duration:
            ts.append(t)

    return ts

def gen_bursty(
    base_qps: float,
    variant_qps: float,
    cv2: float,
    duration: float,
    rng: random.Random,
) -> List[float]:
    """Base+variant bursty process: deterministic base + Gamma variant."""
    base = gen_constant(base_qps, duration) if base_qps > 0 else []
    var  = gen_gamma_bursty(variant_qps, cv2, duration, rng) if variant_qps > 0 else []
    ts = base + var
    ts.sort()
    return ts


def write_trace(out_path: Path, arrivals: List[float], items: List[PopeItem], 
                max_new_tokens: int, deadline_ms: int, pruning_ratio: Optional[float],
                gt_lookup: Dict[Tuple[Optional[int], str], str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with out_path.open("w", encoding="utf-8") as f:
        qi=0
        for a in arrivals:
            it = items[qi % len(items)]
            qi += 1
            gt = gt_lookup.get(it.question_id)
            row = {
                "arrival_time": float(a),
                "image_path": it.image_path,
                "question": it.question,
                "max_new_tokens": int(max_new_tokens),
                "deadline_ms": int(deadline_ms),
                "pruning_ratio": float(pruning_ratio) if pruning_ratio is not None else None,
                "pope_category": it.category,
                "gt_answer": gt,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n

import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["constant", "gamma-bursty", "bursty", "timevary"],
        help="Arrival process type",
    )
    p.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Trace duration in seconds",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )

    # existing arrival-process args...
    p.add_argument("--qps", type=float, help="Mean QPS for constant/timevary")
    p.add_argument("--cv2", type=float, help="CV^2 for gamma-bursty/bursty")
    p.add_argument("--base_qps", type=float, help="Base QPS for bursty/timevary")
    p.add_argument("--variant_qps", type=float, help="Variant QPS for bursty")
    p.add_argument("--lambda1", type=float, help="Initial QPS for timevary")
    p.add_argument("--lambda2", type=float, help="Final QPS for timevary")
    p.add_argument("--accel", type=float, help="Arrival acceleration (qps^2)")
    p.add_argument("--cv2a", type=float, help="Arrival CV^2 for timevary")

    # 🔹 NEW: POPE / VLM bits so your current script invocation works
    p.add_argument(
        "--pope_jsonl",
        type=Path,
        required=True,
        help="Path to llava_pope_test.jsonl",
    )
    p.add_argument(
        "--images_root",
        type=Path,
        required=True,
        help="Root directory with COCO val2014 images",
    )
    p.add_argument(
        "--ann_dir",
        type=Path,
        required=True,
        help="COCO annotation directory (for POPE)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens per request (stored as metadata in trace)",
    )
    p.add_argument(
        "--deadline_ms",
        type=float,
        default=300.0,
        help="Per-request SLO in milliseconds (stored as metadata in trace)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Load POPE items + GT lookup once per run
    items = load_pope_items(args.pope_jsonl, args.images_root)
    gt_lookup = build_gt_lookup(items, args.ann_dir)

    # ---------------- Arrival generation ----------------
    if args.mode == "constant":
        if args.qps is None:
            raise SystemExit("--qps is required for mode=constant")
        arrivals = gen_constant(args.qps, args.duration)

    elif args.mode == "gamma-bursty":
        if args.qps is None:
            raise SystemExit("--qps is required for mode=gamma-bursty")
        if args.cv2 is None:
            raise SystemExit("--cv2 is required for mode=gamma-bursty")
        arrivals = gen_gamma_bursty(args.qps, args.cv2, args.duration, rng)

    elif args.mode == "bursty":
        if args.base_qps is None:
            raise SystemExit("--base_qps is required for mode=bursty")
        if args.variant_qps is None:
            raise SystemExit("--variant_qps is required for mode=bursty")
        if args.cv2 is None:
            raise SystemExit("--cv2 is required for mode=bursty")
        arrivals = gen_bursty(
            args.base_qps,
            args.variant_qps,
            args.cv2,
            args.duration,
            rng,
        )

    elif args.mode == "timevary":
        for nm in ("lambda1", "lambda2", "accel"):
            if getattr(args, nm) is None:
                raise SystemExit(f"--{nm} required for mode=timevary")
        arrivals = gen_timevary(
            args.lambda1,
            args.lambda2,
            args.accel,
            args.duration,
            rng,
            cv2a=args.cv2a,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    out_path = args.out if isinstance(args.out, Path) else Path(args.out)
    write_trace(
        out_path=out_path,
        arrivals=arrivals,
        items=items,
        max_new_tokens=args.max_new_tokens,
        deadline_ms=args.deadline_ms,
        pruning_ratio=None,          # or args.pruning_ratio if you add one later
        gt_lookup=gt_lookup,
    )


if __name__ == "__main__":
    main()
