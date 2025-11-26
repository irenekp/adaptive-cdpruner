#!/usr/bin/env python
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image

from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class TraceRecord:
    image_path: str
    question: str
    gt_answer: str
    max_new_tokens: int


def load_trace(trace_path: str, max_samples: Optional[int] = None) -> List[TraceRecord]:
    records: List[TraceRecord] = []
    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = TraceRecord(
                image_path=d["image_path"],
                question=d["question"],
                gt_answer=d.get("gt_answer", "").strip(),
                max_new_tokens=int(d.get("max_new_tokens", 32)),
            )
            records.append(rec)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def normalize_answer(s: str) -> str:
    return s.strip().lower()


def set_visual_tokens(model, visual_token_num: int):    # wrong???
    """
    Adjust the pruning behavior for CDPruner towers.
    """
    if hasattr(model, "visual_token_num"):
        model.visual_token_num = visual_token_num
    else:
        print("cannot set visual token number")



def build_prompts_and_questions_for_records(
    records: List[TraceRecord],
    model,
    tokenizer,
    conv_mode: str,
) -> Tuple[List[str], List[str]]:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    questions: List[str] = []
    prompts: List[str] = []

    for rec in records:
        raw_q = rec.question
        qs = raw_q

        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
            else:
                qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        questions.append(raw_q)
        prompts.append(prompt)

    return questions, prompts


def tokenize_prompts(
    prompts: List[str],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    id_lists = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        for p in prompts
    ]
    max_len = max(len(ids) for ids in id_lists)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids = torch.full(
        (len(prompts), max_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    for i, ids in enumerate(id_lists):
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, : len(ids)] = ids_tensor

    return input_ids


def load_and_process_images_for_records(
    records: List[TraceRecord],
    image_processor,
    model,
    device: torch.device,
):
    images: List[Image.Image] = []
    image_sizes: List[Tuple[int, int]] = []
    for rec in records:
        img = Image.open(rec.image_path).convert("RGB")
        images.append(img)
        image_sizes.append(img.size)

    image_tensors: List[torch.Tensor] = []
    for img in images:
        t = process_images([img], image_processor, model.config)
        t = t.to(device=device, dtype=torch.float16)
        image_tensors.append(t)

    return image_tensors, image_sizes


def run_accuracy_profile(
    model,
    tokenizer,
    image_processor,
    device: torch.device,
    conv_mode: str,
    records: List[TraceRecord],
    visual_token_nums: List[int],
) -> Dict[int, float]:
    """
    Returns: accuracy_per_vtn[visual_token_num] = accuracy float
    """
    accuracy: Dict[int, float] = {}
    model.eval()

    for vtn in visual_token_nums:
        set_visual_tokens(model, vtn)
        correct = 0
        total = 0

        for rec in records:
            batch_records = [rec]
            questions, prompts = build_prompts_and_questions_for_records(
                batch_records, model, tokenizer, conv_mode
            )
            input_ids = tokenize_prompts(prompts, tokenizer, device)
            image_tensors, image_sizes = load_and_process_images_for_records(
                batch_records, image_processor, model, device
            )

            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    texts=questions,
                    max_new_tokens=rec.max_new_tokens,
                    do_sample=False,
                )

            if isinstance(out, tuple):
                output_ids = out[0]
            else:
                output_ids = out

            gen_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred = normalize_answer(gen_text)
            gold = normalize_answer(rec.gt_answer)

            if gold and gold in pred:
                correct += 1
            total += 1

        accuracy[vtn] = correct / total if total > 0 else 0.0

    return accuracy


def run_latency_profile(
    model,
    tokenizer,
    image_processor,
    device: torch.device,
    conv_mode: str,
    records: List[TraceRecord],
    visual_token_nums: List[int],
    batch_sizes: List[int],
    batches_per_setting: int = 3,
    warmup_iterations: int = 3,
) -> List[Dict[str, Any]]:
    """
    Returns: profile_rows: list of dicts with (visual_token_num, batch_size, latency_ms, latency_stddev_ms, throughput_qps)
    """
    profile_rows: List[Dict[str, Any]] = []
    model.eval()

    # simple cycling over records to form batches
    N = len(records)

    for vtn in visual_token_nums:
        set_visual_tokens(model, vtn)
        for B in batch_sizes:
            latencies_ms: List[float] = []
            if B > N:
                # if batch size > dataset, just skip or clip
                continue

            for b in range(warmup_iterations + batches_per_setting):
                start_idx = (b * B)
                if start_idx > N - B:
                    print("Not enough records to form a full batch, skipping...")
                    continue
                batch_records = records[start_idx : start_idx + B]
                
                torch.cuda.synchronize()
                t0 = time.time()
                questions, prompts = build_prompts_and_questions_for_records(
                    batch_records, model, tokenizer, conv_mode
                )
                input_ids = tokenize_prompts(prompts, tokenizer, device)
                image_tensors, image_sizes = load_and_process_images_for_records(
                    batch_records, image_processor, model, device
                )

                max_new_tokens = max(rec.max_new_tokens for rec in batch_records)
                with torch.inference_mode():
                    _ = model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        texts=questions,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                torch.cuda.synchronize()
                t1 = time.time()
                if b >= warmup_iterations:
                    latencies_ms.append((t1 - t0) * 1000.0)

            if not latencies_ms:
                continue

            numpy_latencies_ms = np.array(latencies_ms)
            avg_latency_ms = np.mean(numpy_latencies_ms)
            std_dev_latency_ms = np.std(numpy_latencies_ms)
            throughput_qps = (1000.0 * B) / avg_latency_ms if avg_latency_ms > 0 else 0.0

            profile_rows.append(
                {
                    "visual_token_num": vtn,
                    "batch_size": B,
                    "latency_ms": avg_latency_ms,
                    "latency_stddev_ms": std_dev_latency_ms,
                    "throughput_qps": throughput_qps,
                }
            )

    return profile_rows

def build_latency_buckets(
    profile_rows: List[Dict[str, Any]],
    bucket_width_ms: float,
) -> Dict[int, Dict[str, Any]]: # review
    """
    Returns:
      buckets[bucket_id] = {
        "latency_min": ...,
        "latency_max": ...,
        "choices": [rows...],
        "max_accuracy_choice": row,
        "max_batch_choice": row,
      }
    """
    buckets: Dict[int, Dict[str, Any]] = {}

    if not profile_rows:
        return buckets

    min_latency = min(r["latency_ms"] for r in profile_rows)
    max_latency = max(r["latency_ms"] for r in profile_rows)
    num_buckets = int((max_latency - min_latency) // bucket_width_ms) + 1

    for i in range(num_buckets):
        bucket_latency = min_latency + (i + 1) * bucket_width_ms
        candidates = [r for r in profile_rows if r["latency_ms"] < bucket_latency]

        if not candidates:
            continue

        best_row = max(
            candidates,
            key=lambda r: (r["batch_size"], r["accuracy"])
        )

        buckets[i] = {
            "latency_ms": bucket_latency,
            "best_choice": best_row,
        }

    return buckets

def save_profiler_plots(profile: Dict[str, Any], out_dir: str = "profiler_plots") -> None:
    """
    Save:
      1) Latency vs batch size (per visual_token_num)
      2) Throughput vs batch size (per visual_token_num)
      3) Latency vs accuracy (Pareto frontier)
    to PNG files under `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = profile.get("profile_rows", [])
    buckets = profile.get("buckets", [])
    accuracy = profile.get("accuracy", [])

    if not rows:
        return

    # --- common helpers ---
    vtn_values = sorted({r["visual_token_num"] for r in rows})
    # Build mapping (vtn, B) -> row for quick lookup
    row_map = {(r["visual_token_num"], r["batch_size"]): r for r in rows}

    # ----------------------------------------------------
    # 1) Latency vs batch size (one curve per vtn)
    # ----------------------------------------------------
    fig, ax = plt.subplots()
    for vtn in vtn_values:
        bs = sorted({r["batch_size"] for r in rows if r["visual_token_num"] == vtn})
        lat = [row_map[(vtn, b)]["latency_ms"] for b in bs]
        err = [row_map[(vtn, b)]["latency_stddev_ms"] for b in bs]
        ax.plot(bs, lat, marker="o", label=f"vtn={vtn}")
        ax.errorbar(bs, lat, yerr=err, fmt="none", capsize=4, alpha=0.7)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Batch size (per visual_token_num)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_vs_batch.png"))
    plt.close(fig)

    # ----------------------------------------------------
    # 2) Throughput vs batch size (one curve per vtn)
    # ----------------------------------------------------
    fig, ax = plt.subplots()
    for vtn in vtn_values:
        bs = sorted({r["batch_size"] for r in rows if r["visual_token_num"] == vtn})
        thr = [row_map[(vtn, b)]["throughput_qps"] for b in bs]
        ax.plot(bs, thr, marker="o", label=f"vtn={vtn}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Throughput vs Batch size (per visual_token_num)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_batch.png"))
    plt.close(fig)

    # ----------------------------------------------------
    # 3) Latency vs Accuracy (Pareto frontier only)
    # ----------------------------------------------------

    if buckets:
        fig, ax = plt.subplots()
        bucket_keys = sorted(buckets.keys())
        lat = [buckets[key]["latency_ms"] for key in bucket_keys]
        acc = [buckets[key]["best_choice"]["accuracy"] for key in bucket_keys]
        ax.scatter(lat, acc, marker="o")
        for key in bucket_keys:
            label = f"vtn={buckets[key]['best_choice']['visual_token_num']},B={buckets[key]['best_choice']['batch_size']}"
            ax.annotate(
                label,
                (buckets[key]["latency_ms"], buckets[key]["best_choice"]["accuracy"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Latency (Pareto frontier)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "latency_vs_accuracy_pareto.png"))
        plt.close(fig)
    
    # ----------------------------------------------------
    # 4) Accuracy vs vtn
    # ----------------------------------------------------
    fig, ax = plt.subplots()        
    acc = [accuracy[vtn] for vtn in vtn_values]
    ax.plot(vtn_values, acc, marker="o")
    ax.set_xlabel("Visual Token Number")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Visual Token Number")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_vtn.png"))
    plt.close(fig)

    # ----------------------------------------------------
    # 5) Batch Size vs Accuracy heatmap (cells = Latency ms)
    # ----------------------------------------------------
    if rows and accuracy:
        bs_values = sorted({r["batch_size"] for r in rows})
        vtn_values = sorted({r["visual_token_num"] for r in rows})
        acc_values = [accuracy[vtn] for vtn in vtn_values]
        heatmap_data = np.zeros((len(vtn_values), len(bs_values)))

        for i, vtn in enumerate(vtn_values):
            for j, b in enumerate(bs_values):
                row = row_map.get((vtn, b))
                if row:
                    heatmap_data[i, j] = row["latency_ms"]
                else:
                    heatmap_data[i, j] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            xticklabels=bs_values,
            yticklabels=[f"{a:.3f}, {vtn}" for a, vtn in zip(acc_values, vtn_values)],
            cmap="viridis",
            cbar_kws={"label": "Latency (ms)"},
            ax=ax,
        )
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Accuracy, Visual Token Number")
        ax.set_title("Accuracy, VTN vs Batch Size")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "batchsize_vs_accuracy_heatmap.png"))
        plt.close(fig)


def run_profiler(
    model,
    tokenizer,
    image_processor,
    device: torch.device,
    conv_mode: str,
    trace_path: str,
    visual_token_nums: List[int],
    batch_sizes: List[int],
    max_accuracy_samples: int = 200,
    max_latency_batches: int = 100,
    latency_bucket_width_ms: float = 10.0,
    warmup_iterations: int = 3,
) -> Dict[str, Any]:
    """
    High-level entrypoint. Returns:
      {
        "profile_rows": [...],
        "buckets": {...},
        "visual_token_nums": [...],
        "batch_sizes": [...],
        "latency_bucket_width_ms": float,
      }
    """
    # Load samples (shared for accuracy + latency)
    acc_records = load_trace(trace_path, max_samples=max_accuracy_samples)
    lat_records = load_trace(trace_path, max_samples=max(batch_sizes)*(max_latency_batches+warmup_iterations))

    accuracy = run_accuracy_profile(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        conv_mode=conv_mode,
        records=acc_records,
        visual_token_nums=visual_token_nums,
    )

    profile_rows = run_latency_profile(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        conv_mode=conv_mode,
        records=lat_records,
        visual_token_nums=visual_token_nums,
        batch_sizes=batch_sizes,
        batches_per_setting=max_latency_batches,
        warmup_iterations=warmup_iterations,
    )

    for row in profile_rows:
        vtn = row["visual_token_num"]
        row_acc = accuracy.get(vtn, 0.0)
        row["accuracy"] = row_acc

    buckets = build_latency_buckets(profile_rows, bucket_width_ms=latency_bucket_width_ms)
    profile = {
        "profile_rows": profile_rows,
        "accuracy": accuracy,
        "buckets": buckets,
        "visual_token_nums": visual_token_nums,
        "batch_sizes": batch_sizes,
        "latency_bucket_width_ms": latency_bucket_width_ms,
    }

    # Save plots to disk (directory can be overridden via env var)
    plot_dir = os.getenv("CDPRUNER_PROFILE_PLOTS_DIR", "profiler_plots")
    try:
        save_profiler_plots(profile, out_dir=plot_dir)
    except Exception as e:
        print(f"[Profiler] Failed to save plots: {e}")

    return profile
