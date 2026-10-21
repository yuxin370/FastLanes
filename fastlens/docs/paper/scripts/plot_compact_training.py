#!/usr/bin/env python3
"""Plot compact training using the shared paper style and existing measurements.

The CSV was exported from ~/tmp/dct-compact-models-20261015/
first_round_summary.csv and the results in h100-first-pass.complete.json.
ViT-Ti/D3 IC and IC-batched on RTX 4090 use three fresh repeats from
~/tmp/dct-compact-vitti-ic-retest-20261018-131036/summary.json.
"""
import numpy as np
from chart_style import RESULTS, BAR_COLORS, BAR_PALETTE, BAR_STYLE, HATCHES, plt, read, save

MODELS = (("resnet50", "ResNet-50"), ("mobilenet_v2", "MobileNetV2"), ("vitti", "ViT-Ti"))
METHODS = (
    ("cpu_dct", "PyTorch DCT", BAR_COLORS["CPU-DCT"], HATCHES["CPU-DCT"]),
    ("ic_prepared_shared", "IC", BAR_PALETTE[2], "//"),
    ("ic_prepared_batched", "IC-batched", BAR_PALETTE[3], "xx"),
    ("fs_prefetched", "FS", BAR_PALETTE[4], "\\\\"),
    ("fastlens", "FastLens", BAR_COLORS["FastLens"], HATCHES["FastLens"]),
)


def main():
    rows = read(RESULTS / "section_6_2_2/compact_training.csv")
    rates = {(r["gpu"], r["model"], int(r["diagonals"]), r["method"]): float(r["images_s"])
             for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.13, top=.80, wspace=.13, hspace=.48)
    x = np.arange(len(MODELS))
    width = .15
    for row, gpu in enumerate(("4090", "H100")):
        for col, diagonals in enumerate((3, 15)):
            ax = axes[row, col]
            for i, (method, label, color, hatch) in enumerate(METHODS):
                values = [rates[gpu, model, diagonals, method] for model, _ in MODELS]
                ax.bar(x + (i - 2) * width, values, width, color=color, hatch=hatch,
                       label=label, **BAR_STYLE)
            ax.set_title(f"{'RTX 4090' if gpu == '4090' else gpu} · D{diagonals}", pad=3, fontsize=9.5)
            ax.set_xticks(x, [label for _, label in MODELS], fontsize=9)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", labelsize=9)
            ax.set_ylim(0, 2100)
            ax.set_yticks([0, 1000, 2000], ["0", "1k", "2k"])
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
    fig.supylabel("Images/s", x=.015, fontsize=9.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=5, loc="upper center",
               frameon=False, fontsize=9, handlelength=1.1, handletextpad=.4, columnspacing=1.0,
               borderaxespad=0)
    save(fig, "evaluation/compact_training")


if __name__ == "__main__":
    main()
