#!/usr/bin/env python3
"""Plot the RTX 4090 H2D profiles used in the paper's traffic table."""
import numpy as np
from chart_style import RESULTS, WIDTH, BAR_COLORS, BAR_STYLE, HATCHES, plt, read, save

SOURCE = RESULTS / "section_6_3/h2d_traffic.csv"
SYSTEMS = ("PyTorch", "DALI", "CoorDL", "FFCV", "L3", "CPU-DCT", "FastLens")
LABELS = ("PyTorch", "DALI", "CoorDL", "FFCV", "L3", "DCT", "FastLens")


def main():
    traffic = {(r["phase"], r["model"], r["pipeline"]): float(r["h2d_bytes_per_image"]) / 1e6
               for r in read(SOURCE) if r["gpu"] == "4090" and r["mode"] == "profile"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.12), sharey=True)
    fig.subplots_adjust(left=.16, right=.99, bottom=.21, top=.56, wspace=.16)
    x = np.arange(2)
    for ax, model, title in zip(axes, ("vitti", "mobilenet24"), ("ViT-Ti", "MobileNetV2")):
        for i, (system, label) in enumerate(zip(SYSTEMS, LABELS)):
            values = [traffic[phase, model, system] for phase in ("training", "inference")]
            ax.bar(x + (i - 3) * .105, values, width=.10, label=label,
                   color=BAR_COLORS[system], hatch=HATCHES[system], **BAR_STYLE)
        ax.set_title(title, pad=2)
        ax.set_xticks(x, ("Training", "Inference"), fontsize=8)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, .6, 1.2], ["0", "0.6", "1.2"])
        ax.tick_params(axis="x", length=0)
        ax.margins(x=.06)
        ax.grid(axis="y", alpha=.15, linewidth=.5)
        ax.set_axisbelow(True)
    fig.supylabel("H2D (MB/image)", x=.01, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = (0, 4, 1, 5, 2, 6, 3)
    fig.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
               ncol=4, loc="upper center",
               frameon=False, handlelength=1.1, handletextpad=.4, columnspacing=.8)
    save(fig, "evaluation/h2d_traffic")


if __name__ == "__main__":
    main()
