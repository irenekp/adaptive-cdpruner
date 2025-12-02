import os
from PIL import Image

IMG_DIR = "results/fig9"

# ---- Your 9 required images ----
target_files = [
    "summary_5_1.png",
    "summary_5_2.png",
    "summary_5_3.png",
    "summary_8_1.png",
    "summary_8_2.png",
    "summary_8_3.png",
    "summary_12_1.png",
    "summary_12_2.png",
    "summary_12_3.png",
]

# ---- Load all images ----
images = [Image.open(os.path.join(IMG_DIR, f)) for f in target_files]

# ---- Ensure equal size ----
w, h = images[0].size

# ---- Tune this value to change spacing ----
SPACING = -40      # negative = overlap slightly (less gap)
# try SPACING = -30 or -50 depending on taste

# ---- Compute final canvas size ----
canvas_w = w * 3 + SPACING * 2
canvas_h = h * 3 + SPACING * 2

combined = Image.new("RGB", (canvas_w, canvas_h), "white")

# ---- Paste with reduced spacing ----
idx = 0
for r in range(3):
    for c in range(3):
        x = c * w + c * SPACING
        y = r * h + r * SPACING
        combined.paste(images[idx], (x, y))
        idx += 1

# ---- Save ----
output_path = "combined_summary_grid_tight.png"
combined.save(output_path, quality=95, dpi=(300, 300))
print("Saved:", output_path)