#!/usr/bin/env python3
"""Redraw evaluation/locality.pdf; throughput is block-major / image-major."""
import numpy as np
from chart_style import RESULTS, WIDTH, BAR_PALETTE, BAR_STYLE, plt, read, save

SOURCE = RESULTS / "section_6_5_1/locality.csv"
FIGSIZE = (WIDTH, 1.65)


def main():
    rows = read(SOURCE)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=.37, right=.97, bottom=.28, top=.68, wspace=.30)
    for ax, gpu, metric in zip(axes, ("4090", "H100", "4090"),
                              ("throughput", "throughput", "bytes")):
        for i, batch in enumerate((64, 1024)):
            selected = [next(r for r in rows if r["experiment"] == "locality" and r["gpu"] == gpu
                             and r["configuration"] == f"{policy}_b{batch}")
                        for policy in ("shared", "correlated", "independent")]
            values = [float(r["throughput_ratio"]) if metric == "throughput" else
                      float(r["optimized_read_bytes"])/float(r["control_read_bytes"]) for r in selected]
            ax.barh(np.arange(3)+(i-.5)*.32, values, height=.29,
                   color=(BAR_PALETTE[6], BAR_PALETTE[7])[i], label=f"Batch {batch}",
                   hatch=("", "//")[i], **BAR_STYLE)
        ax.set_xscale("log")
        ax.axvline(1, color="#777777", ls="--", lw=.55)
        ax.set_xticks([.1, 1, 10], ["0.1", "1", "10"])
        ax.set_xlim(.15 if metric == "throughput" else .2, 12 if metric == "throughput" else 35)
        ax.set_title(("RTX 4090" if gpu == "4090" else gpu) + "\nThroughput"
                     if metric == "throughput" else "Both GPUs\nIssued bytes")
    axes[0].set_yticks(range(3), ["Partition-Shared\nCrop", "Correlated Crop", "Independent Crop"], fontsize=7.5)
    axes[0].set_ylim(2.6, -.6)
    fig.supxlabel("Cross-Image Block-Major / Image-Major Tiled", y=.015, fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper center",
               frameon=False, handlelength=1, columnspacing=1)
    save(fig, "evaluation/locality")


if __name__ == "__main__":
    main()
