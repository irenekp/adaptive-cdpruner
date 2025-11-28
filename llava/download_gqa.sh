python - << 'PY'
import os, json
from datasets import load_dataset

DATA_ROOT = "/storage/ice1/5/3/istephen3/playground/data"

RAW_IMG_DIR = os.path.join(DATA_ROOT, "gqa", "raw", "images")
PROC_DIR    = os.path.join(DATA_ROOT, "gqa", "processed")
HF_CACHE    = os.path.join(DATA_ROOT, "hf-cache")

os.makedirs(RAW_IMG_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# This uses the lmms-lab GQA packaging, subset = val_balanced_instructions
ds = load_dataset(
    "lmms-lab/GQA",
    "val_balanced_instructions",
    split="val",            # that split name is how they expose it
    cache_dir=HF_CACHE,
)

out_path = os.path.join(PROC_DIR, "gqa_balanced_val_rawpaths.jsonl")
with open(out_path, "w") as f:
    for ex in ds:
        rec = {
            "image_id":     ex["imageId"],
            "question_id":  ex["id"],
            "question":     ex["question"],
            "answer":       ex["answer"],
            "full_answer":  ex["fullAnswer"],
            # convention from GQA docs: {imageId}.jpg
            "image_path":   os.path.join(RAW_IMG_DIR, f"{ex['imageId']}.jpg"),
        }
        f.write(json.dumps(rec) + "\n")

print("Wrote", out_path)
PY
