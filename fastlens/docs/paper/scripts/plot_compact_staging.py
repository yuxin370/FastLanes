#!/usr/bin/env python3
"""Plot H100 spatial selection and compact staging from measured byte counts."""
import numpy as np
from chart_style import RESULTS, WIDTH, BAR_PALETTE, BAR_STYLE, plt, read, save

SOURCE = RESULTS / "section_6_4/staging.csv"
FIGSIZE = (WIDTH, 1.15)


def main():
    rows = read(SOURCE)
    paths = (("full", "Full-Source\nRead", BAR_PALETTE[6], "\\\\"),
             ("source", "Projection\nPushdown Only", BAR_PALETTE[4], "//"),
             ("compact", "Projection Pushdown\n+ Compact Staging", BAR_PALETTE[7], ""))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=.17, right=.99, bottom=.21, top=.59, wspace=.20)
    for ax, crop in zip(axes, ("448", "160")):
        for i, (arm, label, color, hatch) in enumerate(paths):
            row = next(r for r in rows if r["crop"] == crop and r["arm"] == arm
                       and r["simulated_MB_s"] == "0")
            values = [float(row[key]) / 1e9 for key in ("issued_payload_bytes", "workset_h2d_bytes")]
            ax.bar(np.arange(2) + (i - 1) * .25, values, width=.23,
                   color=color, hatch=hatch, label=label, **BAR_STYLE)
        ax.set(xticks=range(2), xticklabels=["Payload", "H2D"],
               ylim=(0, 1.75), yticks=[0, .5, 1, 1.5])
        ax.set_title(f"Crop {crop}", pad=2)
        ax.set_xlim(-.5, 1.5)
        ax.grid(axis="y", alpha=.15, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Data (GB)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper center",
               frameon=False, handlelength=.9, columnspacing=.6, handletextpad=.3)
    save(fig, "evaluation/compact_staging_h100")


if __name__ == "__main__":
    main()
