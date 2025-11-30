import json
import os
import zipfile

jsonl_path = "/home/hice1/istephen3/CDPruner/traces/fig9/gqa_gamma_bursty_qps5_cv24_100.jsonl"          # path to your jsonl file
zip_output = "gqa_gamma_bursty_qps_5_cv2_4_100.zip"           # output zip file name

image_paths = []

with open(jsonl_path, "r") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            img_path = entry.get("image_path")
            if img_path:
                image_paths.append(img_path)

with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for path in image_paths:
        if os.path.exists(path):
            # Add to zip with only the file name, not full path
            zipf.write(path, arcname=os.path.basename(path))
        else:
            print(f"Warning: Image not found: {path}")

print(f"Done! Zipped {len(image_paths)} images into {zip_output}")
