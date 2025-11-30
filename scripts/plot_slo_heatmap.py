#!/usr/bin/env python3
"""Plot heatmap of SLO attainment from results/fig9/summary.json
VTN on Y axis, Batch on X axis, annotated SLO attainment values (0.0-1.0).
Saves to results/fig9/slo_attainment_heatmap.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

SUMM = Path(__file__).resolve().parents[1] / 'results' / 'fig9' / 'summary.json'
OUT_DIR = SUMM.parent
OUT_FILE = OUT_DIR / 'slo_attainment_heatmap_8.png'

with SUMM.open('r', encoding='utf-8') as f:
    data = json.load(f)

records = []
for rec in data:
    v = rec.get('vtn')
    b = rec.get('batch')
    slo = rec.get('slo')
    try:
        vnum = int(v)
        bnum = int(b)
    except Exception:
        # skip non-numeric rows (e.g., 'adaptive')
        continue
    records.append((vnum, bnum, float(slo)))

if not records:
    raise SystemExit('No numeric records found in summary.json')

vt_vals = sorted({r[0] for r in records})
batch_vals = sorted({r[1] for r in records})
mat = np.full((len(vt_vals), len(batch_vals)), np.nan, dtype=float)
for v, b, slo in records:
    i = vt_vals.index(v)
    j = batch_vals.index(b)
    mat[i, j] = slo

# force a yellow->purple palette using reversed 'plasma'
try:
    cmap = mpl.colormaps['plasma_r']
except Exception:
    try:
        cmap = mpl.colormaps.get_cmap('plasma_r')
    except Exception:
        # fallback for older matplotlib versions
        cmap = plt.cm.plasma_r

def text_color_for_value(v):
    if np.isnan(v):
        return 'gray'
    r, g, b, _ = cmap(v)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if lum < 0.5 else 'black'

fig, ax = plt.subplots(figsize=(1.2 * len(batch_vals) + 3, 1.2 * len(vt_vals) + 3))
im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(batch_vals)))
ax.set_yticks(range(len(vt_vals)))
ax.set_xticklabels(batch_vals)
ax.set_yticklabels(vt_vals)
ax.set_xlabel('Batch size')
ax.set_ylabel('VTN')
ax.set_title('SLO attainment')

# annotate with percentage format and automatic contrast
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        if np.isnan(val):
            txt = 'NA'
            color = 'gray'
        else:
            txt = f'{val*100:.0f}%'
            color = text_color_for_value(val)
        ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=11, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('SLO attainment')
plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, dpi=200)
print('Wrote', OUT_FILE)
