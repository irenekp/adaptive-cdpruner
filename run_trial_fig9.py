#!/usr/bin/env python3
import argparse
import os
import pathlib
import subprocess
import time

import requests
import json
import os
import subprocess
from send_requests_to_cdpruner_server_async import run_client_internal

def warmup(trace_path: str, server_url: str) -> None:
    """Run a small warmup batch (10 requests) against the server."""
    print(f"[warmup] Running 10 warmup requests on {server_url} ...", flush=True)
    cmd = [
        "python",
        "send_requests_to_cdpruner_server_async.py",
        "--trace-path",
        trace_path,
        "--server-url",
        server_url,
        "--max-reqs",
        "10",
        "--metrics-out",
        os.devnull,
    ]
    subprocess.run(cmd, check=True)
    print("[warmup] Done.\n", flush=True)


def set_fixed_vtn(server_url: str, vtn: int) -> None:
    url = server_url.rstrip("/") + "/set_fixed_vtn"
    resp = requests.post(url, json={"vtn": vtn})
    resp.raise_for_status()

def set_fixed_batch(server_url: str, batch: int) -> None:
    url = server_url.rstrip("/") + "/set_fixed_batch"
    resp = requests.post(url, json={"batch": batch})
    resp.raise_for_status()


def run_client_once(
    trace_path: str,
    server_url: str,
    max_reqs: int,
    metrics_out: str,
) -> None:
    cmd = [
        "python",
        "send_requests_to_cdpruner_server_async.py",
        "--trace-path",
        trace_path,
        "--server-url",
        server_url,
        "--max-reqs",
        str(max_reqs),
        "--metrics-out",
        metrics_out,
    ]
    print(">> Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the same GQA trace against CDPruner server for multiple (vtn, batch) baselines."
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        required=True,
        help="Path to GQA trace JSONL file.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the FastAPI server.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/fig9",
        help="Directory to write metrics JSONL files.",
    )
    parser.add_argument(
        "--max-reqs",
        type=int,
        default=1000,
        help="Max number of requests to replay from the trace.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="gamma_q3_cv1",
        help="Suffix tag for metrics filenames (e.g. 'gamma_q3_cv1').",
    )
    parser.add_argument(
        "--vtns",
        type=int,
        nargs="+",
        default=[64, 128, 256, 384, 576],
        help="List of visual token counts (VTNs) to test.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of fixed batch sizes to test.",
    )
    args = parser.parse_args()

    trace_path = args.trace_path
    server_url = args.server_url.rstrip("/")
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    warmup(trace_path, server_url)

    print(f"Trace:        {trace_path}")
    print(f"Server URL:   {server_url}")
    print(f"Results dir:  {results_dir}")
    print(f"VTNs:         {args.vtns}")
    print(f"Batches:      {args.batches}")
    print(f"Max reqs:     {args.max_reqs}")
    print(f"Tag:          {args.tag}")
    print()
    all_runs = []
    for vtn in args.vtns:
        for batch in args.batches:
            print("=" * 80)
            print(f"Config: VTN={vtn}, BATCH={batch}")
            print("=" * 80)
            print(f"Setting fixed VTN={vtn}...")
            set_fixed_vtn(server_url, vtn)

            print(f"Setting fixed BATCH={batch}...")
            set_fixed_batch(server_url, batch)
            time.sleep(0.5)
            metrics_out = results_dir / f"vtn{vtn}_B{batch}_{args.tag}.jsonl"
            metrics = run_client_internal(
                trace_path=trace_path,
                server_url=server_url,
                max_reqs=args.max_reqs
            )
            metrics_out = results_dir / f"vtn{vtn}_B{batch}_{args.tag}.jsonl"
            with open(metrics_out, "w") as f:
                for rec in metrics:
                    f.write(json.dumps(rec) + "\n")
            slo_attainment = sum(1 for m in metrics if m["latency_ms"] <= m["deadline_ms"]) / len(metrics)
            print(f"SLO attainment = {slo_attainment:.3f}")
            all_runs.append({
                "vtn": vtn,
                "batch": batch,
                "metrics": metrics,
                "slo": slo_attainment,
            })

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print("\nAll (vtn, batch) runs completed.")


if __name__ == "__main__":
    main()
