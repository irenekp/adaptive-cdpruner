#!/usr/bin/env python
import argparse
import json
import time
from typing import List, Dict

import requests


def load_trace(path: str) -> List[Dict]:
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    # Just in case, sort by arrival_time
    events.sort(key=lambda e: e["arrival_time"])
    return events


def replay_trace(trace_path: str, server_url: str, max_reqs: int | None = None):
    events = load_trace(trace_path)
    if max_reqs is not None:
        events = events[:max_reqs]

    base_url = server_url.rstrip("/")
    t0 = time.time()

    print(f"Replaying {len(events)} requests to {base_url}/generate")

    for idx, ev in enumerate(events):
        arrival = ev["arrival_time"]  # seconds from t0
        target_time = t0 + arrival
        now = time.time()
        sleep_for = target_time - now
        if sleep_for > 0:
            time.sleep(sleep_for)

        payload = {
            "prompt": ev["question"],
            "image_path": ev["image_path"],
            # "max_new_tokens": ev["max_new_tokens"],
            # "deadline_ms": ev["deadline_ms"],
        }

        send_time = time.time()
        try:
            resp = requests.post(f"{base_url}/generate", json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[{idx}] ERROR sending request: {e}")
            continue

        finish_time = time.time()
        latency_ms = (finish_time - send_time) * 1000.0
        deadline = ev.get("deadline_ms", None)

        print(
            f"[{idx}] Q: {ev['question']!r} "
            f"→ answer: {data.get('output', '')[:80]!r} "
            f"(latency={latency_ms:.1f} ms"
            + (f", deadline={deadline} ms" if deadline is not None else "")
            + ")"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-path",
        type=str,
        required=True,
        help="Path to JSONL trace file.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://0.0.0.0:8000",
        help="Base URL of FastAPI server (without /generate).",
    )
    parser.add_argument(
        "--max-reqs",
        type=int,
        default=None,
        help="Optional cap on number of requests to replay.",
    )
    args = parser.parse_args()

    replay_trace(args.trace_path, args.server_url, args.max_reqs)


if __name__ == "__main__":
    main()

'''
python send_requests_to_cdpruner_server.py \
  --trace-path /home/hice1/istephen3/CDPruner/traces/sample_trace.jsonl \
  --server-url http://0.0.0.0:8000
'''