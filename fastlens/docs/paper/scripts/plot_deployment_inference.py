#!/usr/bin/env python3
"""Plot completed full-validation reruns with three-trial medians and ranges."""
import numpy as np
from chart_style import RESULTS, BAR_COLORS, BAR_STYLE, HATCHES, plt, read, save
from plot_deployment_training import MODELS, SYSTEMS

BASELINES = RESULTS / 'end_to_end/remaining/summary.csv'
SWIN = RESULTS / 'end_to_end/swin_inference/summary.csv'
FIGSIZE = (7.0, 1.65)


def inference_data():
    results = {(r['gpu'], r['model'], r['path']): r for r in read(BASELINES)
               if r['phase'] == 'inference'}
    results.update({(r['gpu'], 'swinv2', r['path']): r for r in read(SWIN)})
    data = {}
    paths = ('pytorch', 'dali', 'coordl', 'ffcv_raw', 'ffcv_jpeg', 'l3', 'cpu_dct', 'imagelanes')
    for gpu in ('4090', 'H100'):
        for model, _ in MODELS:
            for system, path in zip(SYSTEMS, paths):
                source_model = 'mobilenet24' if model == 'mobilenet32' and path not in ('cpu_dct', 'imagelanes') else model
                result = results[gpu, source_model, path]
                assert int(result['trials']) == 3
                data[gpu, model, system] = tuple(float(result[k + '_images_per_s']) for k in ('median', 'min', 'max'))
    return data


def plot(axes, systems=SYSTEMS):
    data = inference_data()
    x = np.arange(len(MODELS))
    width = .76 / len(systems)
    for ax, gpu in zip(axes, ('4090', 'H100')):
        for i, system in enumerate(systems):
            summaries = np.array([data[gpu, model, system] for model, _ in MODELS])
            medians = summaries[:, 0]
            errors = [medians - summaries[:, 1], summaries[:, 2] - medians]
            label = {'FastLens': 'FastLens DCT', 'CPU-DCT': 'PyTorch DCT',
                     'FFCV': 'FFCV raw RGB'}.get(system, system + ' RGB')
            ax.bar(x + (i - (len(systems) - 1) / 2) * width, medians, width=width * .9,
                   color=BAR_COLORS[system], yerr=errors,
                   error_kw={'elinewidth': .5, 'capsize': 1, 'capthick': .5, 'ecolor': '#333333'},
                   label=label, hatch=HATCHES[system], **BAR_STYLE)
        ax.set_xticks(x, [label for _, label in MODELS], fontsize=7.5)
        ax.tick_params(axis='x', length=0)
        ax.set_title('RTX 4090' if gpu == '4090' else gpu, pad=3)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 7200)
        ax.set_yticks([0, 2000, 4000, 6000], ['0', '2k', '4k', '6k'])


def main():
    figure,axes=plt.subplots(1,2,figsize=FIGSIZE,sharey=True)
    figure.subplots_adjust(left=.065,right=.99,bottom=.22,top=.66,wspace=.12)
    plot(axes)
    axes[0].set_ylabel('Images/s')
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.52,1),fontsize=7,
                  handlelength=1.2,handletextpad=.3,columnspacing=.7,borderaxespad=0,frameon=False)
    save(figure, 'evaluation/01_deployment_inference')


if __name__ == "__main__":
    main()
