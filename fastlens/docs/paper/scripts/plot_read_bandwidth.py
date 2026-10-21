#!/usr/bin/env python3
"""Plot the paper's bandwidth controls, retaining the H100 2000 MB/s recheck."""
import numpy as np
from chart_style import RESULTS, WIDTH, BAR_PALETTE, BAR_STYLE, plt, read, save

SOURCE = RESULTS / "section_6_5_2/bandwidth.csv"
BANDWIDTHS = (125, 500, 2000)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.20), sharey=True)
    fig.subplots_adjust(left=.16, right=.99, bottom=.34, top=.66, wspace=.20)
    for ax, gpu in zip(axes, ("4090", "H100")):
        rows = [r for r in read(SOURCE) if r["gpu"] == gpu]
        medians = {(int(r["crop"]), int(r["simulated_MB_s"]), r["arm"]): float(r["median_s"])
                   for r in rows}
        for i, (crop, color, hatch) in enumerate(((448, BAR_PALETTE[6], "//"),
                                                 (160, BAR_PALETTE[7], ""))):
            ratios = [medians[crop, mb, "full"] / medians[crop, mb, "compact"]
                      for mb in BANDWIDTHS]
            ax.bar(np.arange(3) + (i - .5) * .30, ratios, width=.28,
                   color=color, hatch=hatch, label=f"Crop {crop}", **BAR_STYLE)
        ax.axhline(1, color="#777777", ls="--", lw=.55)
        ax.set(xticks=range(3), xticklabels=["125", "500", "2,000"],
               ylim=(0, 2.1), yticks=[0, 1, 2])
        ax.set_title("RTX 4090" if gpu == "4090" else gpu, pad=2)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=.15, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Throughput ratio")
    fig.supxlabel("Bandwidth setting (MB/s)", y=.015, fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper center",
               frameon=False, handlelength=1.1, columnspacing=1, handletextpad=.4)
    save(fig, "evaluation/read_bandwidth")


if __name__ == "__main__":
    main()
