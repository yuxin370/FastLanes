#!/usr/bin/env python3
"""Redraw evaluation/04_core_ablations.pdf from paired timing and separate H2D profiles."""
import numpy as np
from chart_style import RESULTS, WIDTH, BAR_PALETTE, BAR_STYLE, plt, read, save

TIMING = RESULTS / "section_6_4/core_timing.csv"
PROFILES = RESULTS / "section_6_4/core_h2d.csv"
FIGSIZE = (WIDTH, 1.55)


def main():
    timing = read(TIMING)
    h2d = {(r["gpu"], r["arm"]): int(r["total_h2d_bytes"]) for r in read(PROFILES)}
    groups = [("compressed_transfer", "Compressed\nH2D Transfer\n+ GPU-Native\nDecoding", "raw_int16", "compressed"),
              ("direct_construction", "Direct\nTensor\nConstruction", "grid", "direct")]
    positions = np.array([0., 1.6])
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=.17, right=.97, bottom=.45, top=.73, wspace=.30)
    for ax, gpu in zip(axes, ("4090", "H100")):
        values = [[], [], []]
        for configuration, _, control, optimized in groups:
            r = next(r for r in timing if r["gpu"] == gpu and r["experiment"] == "core"
                     and r["configuration"] == configuration)
            values[0].append(float(r["optimized_read_bytes"])/float(r["control_read_bytes"]))
            total = [h2d[gpu, arm] for arm in (control, optimized)]
            values[1].append(total[1]/total[0])
            values[2].append(float(r["optimized_median_s"])/float(r["control_median_s"]))
        for i, (metric, label, color) in enumerate(zip(
                values, ("Issued payload", "Total H2D", "Elapsed time"),
                (BAR_PALETTE[6], BAR_PALETTE[4], BAR_PALETTE[7]))):
            ax.bar(positions+(i-1)*.23, metric, width=.21, color=color, label=label,
                   hatch=("", "//", "\\\\")[i], **BAR_STYLE)
        ax.axhline(1, color="#777777", ls="--", lw=.55)
        ax.set(xticks=positions, xticklabels=[r[1] for r in groups],
               ylim=(0, 1.08), yticks=[0, .5, 1])
        ax.set_title("RTX 4090" if gpu == "4090" else gpu, pad=2)
        ax.set_xlim(-.45, 2.05)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Normalized cost")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper center",
               frameon=False, handlelength=1, columnspacing=.7, handletextpad=.4)
    save(fig, "evaluation/04_core_ablations")


if __name__ == "__main__":
    main()
