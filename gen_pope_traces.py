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
                pass
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

def gen_bursty(base_qps: float, variant_qps: float, cv2: float, duration: float, rng: random.Random) -> List[float]:
    base = gen_constant(base_qps, duration)
    var  = gen_gamma_bursty(variant_qps, cv2, duration, rng)
    out = base + var
    out.sort()
    return out

def _poisson_knuth(rng: random.Random, mean: float) -> int:
    if mean<=0: return 0
    if mean<20:
        L = math.exp(-mean); k=0; p=1.0
        while p>L:
            k+=1; p*=rng.random()
        return k-1
    # normal approx
    return max(0, int(rng.gauss(mean, math.sqrt(mean)) + 0.5))

def gen_timevary(l1: float, l2: float, accel: float, duration: float, rng: random.Random, dt: float=0.01) -> List[float]:
    low, high = (l1,l2) if l1<=l2 else (l2,l1)
    t=0.0
    ts: List[float] = []
    while t<duration:
        lam = l1 + accel*t
        lam = min(max(lam, low), high)
        expected = max(lam,0.0)*dt
        k = _poisson_knuth(rng, expected)
        for _ in range(k):
            ts.append(t + rng.random()*dt)
        t += dt
    ts.sort()
    return ts

# ---------------- Assemble rows ----------------

def write_trace(out_path: Path, arrivals: List[float], items: List[PopeItem], 
                max_new_tokens: int, deadline_ms: int, pruning_ratio: Optional[float],
                gt_lookup: Dict[Tuple[Optional[int], str], str]) -> int:
    def norm(q: str)->str: return " ".join(q.lower().strip().split())
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["constant","bursty","timevary"], required=True)
    ap.add_argument("--duration", type=float, required=True, help="seconds")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pope_jsonl", type=Path, required=True)
    ap.add_argument("--images_root", type=Path, required=True)
    ap.add_argument("--ann_dir", type=Path, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--deadline_ms", type=int, default=300)
    ap.add_argument("--pruning_ratio", type=float, default=None)

    # constant
    ap.add_argument("--qps", type=float)

    # bursty
    ap.add_argument("--base_qps", type=float)
    ap.add_argument("--variant_qps", type=float)
    ap.add_argument("--cv2", type=float)

    # timevary
    ap.add_argument("--lambda1", type=float)
    ap.add_argument("--lambda2", type=float)
    ap.add_argument("--accel", type=float)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = load_pope_items(args.pope_jsonl, args.images_root)
    gt_lookup = build_gt_lookup(items, args.ann_dir)

    if args.mode=="constant":
        if args.qps is None: ap.error("--qps required for mode=constant")
        arrivals = gen_constant(args.qps, args.duration)
    elif args.mode=="bursty":
        for nm in ("base_qps","variant_qps","cv2"):
            if getattr(args, nm) is None: ap.error(f"--{nm} required for mode=bursty")
        arrivals = gen_bursty(args.base_qps, args.variant_qps, args.cv2, args.duration, rng)
    else:
        for nm in ("lambda1","lambda2","accel"):
            if getattr(args, nm) is None: ap.error(f"--{nm} required for mode=timevary")
        arrivals = gen_timevary(args.lambda1, args.lambda2, args.accel, args.duration, rng)

    n = write_trace(args.out, arrivals, items, args.max_new_tokens, args.deadline_ms, args.pruning_ratio, gt_lookup)
    print(f"Wrote {n} rows -> {args.out}")

if __name__ == "__main__":
    main()
