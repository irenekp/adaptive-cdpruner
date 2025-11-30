#!/usr/bin/env python3
import argparse
import json
import time
import asyncio
from typing import List, Dict
import os

import aiohttp

REQUEST_SLO_MS = float(os.getenv("CDPRUNER_REQUEST_SLO_MS", "1100"))
slo_violated_count = 0 
slo_violated_count_lock = asyncio.Lock() 
METRICS: List[Dict] = []
metrics_lock = asyncio.Lock()
returned_vtns = []

def load_trace(path: str) -> List[Dict]:
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    events.sort(key=lambda e: e["arrival_time"])
    return events


async def send_request(session: aiohttp.ClientSession, base_url: str, idx: int, ev: Dict, t0: float):
    """Wait until arrival_time and then send the request asynchronously."""
    global slo_violated_count, slo_violated_count_lock, returned_vtns
    arrival = ev["arrival_time"]
    target_time = t0 + arrival
    now = time.time()
    sleep_for = target_time - now
    if sleep_for > 0:
        await asyncio.sleep(sleep_for)

    payload = {
        "prompt": ev["question"],
        "image_path": ev.get("image_path"),
    }

    send_time = time.time()
    try:
        async with session.post(f"{base_url}/generate", json=payload, timeout=300) as resp:
            data = await resp.json()
    except Exception as e:
        print(f"[{idx}] ERROR sending request: {e}")
        return

    finish_time = time.time()
    latency_ms = (finish_time - send_time) * 1000.0
    deadline = ev.get("deadline_ms")
    slo_miss = False

    vtn = data.get("vtn")
    returned_vtns.append(vtn)

    if latency_ms > REQUEST_SLO_MS: # TODO: use REQUEST_SLO_MS or deadline?
        async with slo_violated_count_lock:
            slo_violated_count += 1
        slo_miss = True
    # record metrics for offline analysis
    record = {
        "idx": idx,
        "arrival_time": ev.get("arrival_time"),
        "latency_ms": latency_ms,
        "deadline_ms": deadline,
        "slo_miss": slo_miss,
    }
    async with metrics_lock:
        METRICS.append(record)

    print(
        f"[{idx}] Q: {ev['question']!r} "
        f"→ answer: {data.get('output', '')[:80]!r} "
        f"(vtn={vtn}, latency={latency_ms:.1f} ms"
        + (f", deadline={deadline} ms" if deadline is not None else "")
        + ")"
    )


async def replay_trace(trace_path: str, server_url: str, max_reqs: int | None = None):
    """Replay trace asynchronously — requests fire based on arrival times."""
    events = load_trace(trace_path)
    if max_reqs is not None:
        events = events[:max_reqs]

    base_url = server_url.rstrip("/")
    print(f"Replaying {len(events)} requests to {base_url}/generate")

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(send_request(session, base_url, idx, ev, t0))
            for idx, ev in enumerate(events)
        ]
        await asyncio.gather(*tasks)

async def replay_trace_return_metrics(trace_path: str, server_url: str, max_reqs: int):
    """
    Identical to replay_trace, except it returns METRICS instead of writing 
    to a file or printing anything. METRICS must be cleared before each run.
    """
    global METRICS
    METRICS = []
    await replay_trace(trace_path, server_url, max_reqs)
    return METRICS

def run_client_internal(trace_path: str, server_url: str, max_reqs: int):
    """
    Synchronous wrapper. Returns list of metrics dicts.
    """
    import asyncio
    return asyncio.run(replay_trace_return_metrics(trace_path, server_url, max_reqs))


def main():
    global slo_violated_count
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", type=str, required=True, help="Path to JSONL trace file.")
    parser.add_argument("--server-url", type=str, default="http://0.0.0.0:8000",
                        help="Base URL of FastAPI server (without /generate).")
    parser.add_argument("--max-reqs", type=int, default=None,
                        help="Optional cap on number of requests to replay.")
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="Optional JSONL file to write per-request metrics.")

    args = parser.parse_args()

    asyncio.run(replay_trace(args.trace_path, args.server_url, args.max_reqs))
    if args.metrics_out is not None:
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            for rec in METRICS:
                f.write(json.dumps(rec) + "\n")

    print(f"\nTotal SLO Violations: {slo_violated_count}")

    if returned_vtns:
        avg_vtn = sum(returned_vtns) / len(returned_vtns)
        print(f"Average returned visual token num: {avg_vtn:.2f}")

if __name__ == "__main__":
    main()

'''
python send_requests_to_cdpruner_server_async.py \
  --trace-path /home/hice1/istephen3/CDPruner/traces/sample_trace.jsonl \
  --server-url http://0.0.0.0:8000
'''