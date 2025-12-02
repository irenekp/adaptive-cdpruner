import json
import matplotlib.pyplot as plt
import os

# ---- Config: directory + parameter grids ----
BASE_DIR = "/home/hice1/nmeda6/adaptive-cdpruner/results/fig9"

qps_values = ["5", "8", "12"]        # middle value in filename
cv2_values = ["1", "2", "3"]     # last value in filename


def generate_plot(input_json_path, output_plot_path, qps, cv2):
    # ---- Load JSON file ----
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # ---- Track max SLO per VTN + label ----
    fixed_vtns = {}
    adaptive_point = None

    for entry in data:
        slo = entry["slo"]

        if entry["vtn"] == "adaptive":
            vtn_val = entry.get("avg_vtn")
            if vtn_val is None:
                continue
            if adaptive_point is None or slo > adaptive_point[1]:
                adaptive_point = (vtn_val, slo)
        else:
            vtn_val = entry["vtn"]
            if vtn_val not in fixed_vtns or slo > fixed_vtns[vtn_val]:
                fixed_vtns[vtn_val] = slo

    # ---- Prepare plotting data ----
    fixed_xs = sorted(fixed_vtns.keys())
    fixed_ys = [fixed_vtns[v] for v in fixed_xs]

    plt.figure()

    # ---- Plot fixed VTNs ----
    plt.plot(
        fixed_xs,
        fixed_ys,
        color='black',
        marker='s',
        markerfacecolor='red',
        linestyle='-',
        label="Fixed VTN"
    )

    # ---- Plot adaptive VTN ----
    if adaptive_point is not None:
        x_adapt, y_adapt = adaptive_point
        plt.plot(x_adapt, y_adapt, 'bo', label="Adaptive VTN")
        plt.annotate(
            f"SLO: {y_adapt}\nVTN: {x_adapt}",
            (x_adapt, y_adapt),
            textcoords="offset points",
            xytext=(5, -10),   # → shift right 5px, DOWN 25px
            ha='left',
            va='top'           # → anchor the text box from its top edge
        )

    plt.ylim(0.45, 1.05)
    plt.xlim(0, 650)
    plt.xlabel("Visual Token Number (VTN)")
    plt.ylabel("SLO Attainment")

    # ---- Add QPS + CV2 in the title ----
    plt.title(f"SLO vs VTN   |   QPS={qps}, CV$^2$={cv2}")

    plt.legend()

    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved annotated plot to: {output_plot_path}")


# ---- Loop over all QPS × CV2 combinations ----
for qps in qps_values:
    for cv2 in cv2_values:
        input_name = f"summary_{qps}_{cv2}.json"
        output_name = f"summary_{qps}_{cv2}.png"

        input_path = os.path.join(BASE_DIR, input_name)
        output_path = os.path.join(BASE_DIR, output_name)

        generate_plot(input_path, output_path, qps, cv2)
