#!/usr/bin/env python3
"""Plot resident attainment: median and min--max of three per-round ratios.

Best RGB selects the fastest of six RGB paths within each round. FastLens DCT
uses its matched DCT resident reference; values above 100% are retained.
"""
import argparse
import numpy as np

from chart_style import RESULTS, WIDTH, BAR_PALETTE, BAR_COLORS, plt, read, save
from plot_deployment_training import MODELS


def resident_data():
    trials = read(RESULTS / "end_to_end/resident/trials.csv")
    trials += [r for r in read(RESULTS / "end_to_end/remaining/trials.csv")
               if r["phase"] == "training"]
    rates = {(r["gpu"], r["model"], r["path"], int(r["trial"])): float(r["images_per_s"])
             for r in trials}
    rgb_paths = ("pytorch", "dali", "coordl", "l3", "ffcv_raw", "ffcv_jpeg")
    data = {}
    for gpu in ("4090", "H100"):
        for model, _ in MODELS:
            rgb_model = "mobilenet24" if model == "mobilenet32" else model
            rgb = [100 * max(rates[gpu, rgb_model, path, trial] for path in rgb_paths)
                   / rates[gpu, rgb_model, "rgb_resident", trial] for trial in range(3)]
            dct = [100 * rates[gpu, model, "imagelanes_basic", trial]
                   / rates[gpu, model, "dct_resident", trial] for trial in range(3)]
            for domain, values in (("RGB", rgb), ("DCT", dct)):
                data[gpu, model, domain] = (np.median(values), min(values), max(values))
    return data


def main():
    data = resident_data()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 1.72), sharex=True, sharey=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.17, top=.78, hspace=.70)
    for ax, gpu in zip(axes, ("4090", "H100")):
        for domain, label, offset, color, marker in (
            ("RGB", "Best RGB", -.105, BAR_PALETTE[6], "o"),
            ("DCT", "FastLens DCT", .105, BAR_COLORS["FastLens"], "D"),
        ):
            values = np.array([data[gpu, model, domain] for model, _ in MODELS])
            median, low, high = values.T
            ax.errorbar(np.arange(len(MODELS)) + offset, median,
                        yerr=[median - low, high - median], fmt=marker,
                        color=color, markersize=3.3, markeredgecolor="#555555",
                        markeredgewidth=.35, ecolor="#666666", elinewidth=.65,
                        capsize=1.3, capthick=.65, label=label, zorder=3)
        ax.axhline(100, color="#777777", linestyle="--", linewidth=.6)
        ax.set(xlim=(-.5, 4.5), ylim=(75, 115.5), yticks=[80, 90, 100, 110])
        ax.set_title("RTX 4090" if gpu == "4090" else gpu, fontsize=7.5, pad=2)
        ax.tick_params(axis="y", labelsize=6.5, pad=2, length=2)
        ax.tick_params(axis="x", labelsize=7, pad=2, length=2)
        ax.grid(axis="y", alpha=.15, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].tick_params(axis="x", bottom=False)
    axes[1].set_xticks(range(len(MODELS)), [label for _, label in MODELS])
    fig.supylabel("Resident attainment (%)", x=.02, fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper center",
               bbox_to_anchor=(.57, 1), frameon=False, fontsize=7,
               handlelength=1, handletextpad=.4, columnspacing=1.5, borderaxespad=0)
    save(fig, "evaluation/training_resident_attainment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
