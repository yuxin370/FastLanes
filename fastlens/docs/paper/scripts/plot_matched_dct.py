#!/usr/bin/env python3
"""Combine matched DCT training and inference throughput in one figure."""
import numpy as np
from chart_style import RESULTS, BAR_COLORS, BAR_STYLE, HATCHES, plt, read, save

MODELS = [('vitti', 'ViT-Ti'), ('swinv2', 'SwinV2\nT'), ('mobilenet24', 'MV2\nDCT24'),
          ('mobilenet32', 'MV2\nDCT32'), ('resnet24', 'RN50\nDCT24'), ('resnet64', 'RN50\nDCT64'),
          ('efun', 'eFUN')]


def matched_data():
    trials = [dict(r, phase='training') for r in read(RESULTS/'end_to_end/resident/trials.csv')]
    trials += read(RESULTS/'end_to_end/remaining/trials.csv')
    trials += [dict(r, phase='inference', model='swinv2')
               for r in read(RESULTS/'end_to_end/swin_inference/trials.csv')]
    trials += read(RESULTS/'end_to_end/resnet/trials.csv')
    rates = {(r['phase'], r['gpu'], r['model'], r['path'], int(r['trial'])): float(r['images_per_s'])
             for r in trials}
    assert len(rates) == len(trials)
    data, ratios = {}, {}
    for task, phase, path in [('Training', 'training', 'imagelanes_basic'),
                              ('Inference', 'inference', 'imagelanes')]:
        data[task], ratios[task] = {}, {}
        for gpu in ('4090', 'H100'):
            for model, _ in MODELS:
                cpu = np.array([rates[phase, gpu, model, 'cpu_dct', t] for t in range(3)])
                native = np.array([rates[phase, gpu, model, path, t] for t in range(3)])
                for system, values in [('CPU-DCT', cpu), ('FastLens', native)]:
                    data[task][gpu, model, system] = (np.median(values), min(values), max(values))
                paired = native / cpu
                ratios[task][gpu, model] = (np.median(paired), min(paired), max(paired))
    return data, ratios


def main():
    data, _ = matched_data()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.00), sharex=True, sharey="col")
    fig.subplots_adjust(left=.075, right=.995, bottom=.18, top=.80, wspace=.12, hspace=.52)
    x = np.arange(len(MODELS))
    width = .34
    for row, gpu in enumerate(("4090", "H100")):
        for col, task in enumerate(("Training", "Inference")):
            ax = axes[row, col]
            for i, (system, label) in enumerate((("CPU-DCT", "PyTorch DCT"), ("FastLens", "FastLens"))):
                summaries = np.array([data[task][gpu, model, system] for model, _ in MODELS])
                medians = summaries[:, 0]
                ax.bar(x + (i - .5) * width, medians, width, color=BAR_COLORS[system], label=label,
                       hatch=HATCHES[system], **BAR_STYLE,
                       yerr=[medians - summaries[:, 1], summaries[:, 2] - medians],
                       error_kw={"elinewidth": .5, "capsize": 1, "capthick": .5})
            ax.set_title(f"{task} · {'RTX 4090' if gpu == '4090' else gpu}", pad=3, fontsize=9.5)
            ax.set_xticks(x, [label for _, label in MODELS], fontsize=8)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", labelsize=9)
            ax.margins(x=.05)
            ax.set_ylim(0, 3500 if col == 0 else 7000)
            ax.set_yticks([0, 1500, 3000] if col == 0 else [0, 3000, 6000],
                          ["0", "1.5k", "3k"] if col == 0 else ["0", "3k", "6k"])
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
    fig.supylabel("Images/s", x=.01, fontsize=9.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=2, loc="upper center",
               frameon=False, fontsize=9, handlelength=1, handletextpad=.4, columnspacing=1.5, borderaxespad=0)
    save(fig, "evaluation/06_matched_dct")


if __name__ == "__main__":
    main()
