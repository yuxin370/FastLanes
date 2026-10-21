#!/usr/bin/env python3
"""Plot only executed observations; absent cells remain absent."""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4] / 'results'
OUTPUT = ROOT / 'figures'
COLORS = {'imagelanes':'#009e73', 'dct_major_pushdown':'#009e73', 'cpu_dct':'#e69f00',
          'galp':'#009e73',
          'rgbnomore':'#e69f00', 'pytorch':'#0072b2', 'rgb_pytorch':'#0072b2',
          'dali':'#56b4e9', 'rgb_dali':'#56b4e9', 'coordl':'#cc79a7', 'ffcv':'#d55e00'}
LABELS = {'imagelanes':'FastLens','dct_major_pushdown':'FastLens','cpu_dct':'CPU-DCT',
          'galp':'FastLens',
          'rgbnomore':'CPU-DCT','pytorch':'PyTorch RGB','rgb_pytorch':'PyTorch RGB',
          'dali':'DALI RGB','rgb_dali':'DALI RGB','coordl':'CoorDL RGB','ffcv':'FFCV RGB'}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(figure, name):
    figure.tight_layout()
    for extension in ('pdf','png'):
        figure.savefig(OUTPUT / f'{name}.{extension}', dpi=180)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    systems=['PyTorch','DALI','CoorDL','FFCV','FastLens']
    aliases={'PyTorch':'pytorch','DALI':'dali','CoorDL':'coordl','FFCV':'ffcv','FastLens':'imagelanes','CPU-DCT':'cpu_dct'}
    training=read(ROOT/'p0/training_window_summary.csv')
    figure,axes=plt.subplots(2,2,figsize=(11,6),sharey='col')
    for row_index,gpu in enumerate(['RTX 4090','H100']):
        for col_index,model in enumerate(['SwinV2-T','MobileNetV2']):
            ax=axes[row_index,col_index]
            for i,system in enumerate(systems):
                record=next((r for r in training if (r['gpu'],r['model'],r['pipeline'])==(gpu,model,system)),None)
                if record is None:
                    ax.text(i,.04,'TBD',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
                    continue
                value=float(record['median_images_per_s'])
                ax.bar(i,value,color=COLORS[aliases[system]])
                ax.errorbar(i,value,yerr=[[value-float(record['min_images_per_s'])],[float(record['max_images_per_s'])-value]],fmt='none',color='black',capsize=3)
            ax.set_xticks(range(5),[s+'\n'+('DCT' if s=='FastLens' else 'RGB') for s in systems],fontsize=8)
            ax.set_title(f'{gpu} / {model}',loc='left')
            ax.set_ylabel('Training images/s')
            ax.set_xlim(-.6,4.6)
    for col_index,model in enumerate(['SwinV2-T','MobileNetV2']):
        maximum=max(float(r['max_images_per_s']) for r in training if r['model']==model)
        axes[0,col_index].set_ylim(0,maximum*1.12)
    figure.suptitle('ImageNet-512: 65,536-image windows, BF16, median and min–max\nRGB/DCT deployment comparison; missing measurements marked TBD')
    save(figure,'training_windows')
    # Only correctness-accepted observations from the revised plan enter figures.
    rows = read(ROOT / 'inference_table.csv')
    for row in rows:
        row['pipeline']=aliases[row['pipeline']]
    for gpu in ('4090','H100'):
        selected = [r for r in rows if gpu in r['gpu']]
        if not selected:
            continue
        models = sorted({(r['model'],r['domain']) for r in selected})
        figure, axes = plt.subplots(len(models), 1, figsize=(10,2.2*len(models)), squeeze=False)
        for ax, (model, domain) in zip(axes[:,0],models):
            group = [r for r in selected if (r['model'],r['domain']) == (model,domain)]
            values = [float(r['median_images_per_s']) for r in group]
            ax.bar(range(len(group)),values,color=[COLORS[r['pipeline']] for r in group])
            ax.errorbar(range(len(group)),values,
                        yerr=[[v-float(r['min_images_per_s']) for v,r in zip(values,group)],
                              [float(r['max_images_per_s'])-v for v,r in zip(values,group)]],
                        fmt='none',ecolor='black',capsize=3)
            ax.set_xticks(range(len(group)),[LABELS[r['pipeline']] for r in group])
            ax.set_ylabel('Images/s')
            ax.set_title(f'{model} / {domain.upper()} / batch {group[0]["batch_size"]} / FP32',loc='left')
        figure.suptitle(f'{gpu}: ImageNet-512, 50,000 images; median and min–max\nPartial evaluation; shared host; RGB/DCT are deployment comparisons',y=1.005)
        save(figure,f'inference_{gpu}')
    available = read(ROOT/'analysis/pipeline_efficiency.csv')
    for row in available:
        row['pipeline']=aliases[row['pipeline']]
    if available:
        labels = [f'{"H100" if "H100" in r["gpu"] else "4090"} / {r["model"]} / {LABELS[r["pipeline"]]}' for r in available]
        figure, ax = plt.subplots(figsize=(11,max(4,len(labels)*.27)))
        ax.barh(range(len(labels)),[float(r['pipeline_efficiency']) for r in available],color=[COLORS[r['pipeline']] for r in available])
        ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis()
        ax.set_xlabel('End-to-end throughput / matched model-only throughput')
        ax.set_title('ImageNet-512 training windows / BF16 — matched resident-tensor ceiling')
        save(figure,'pipeline_efficiency')
    for path in sorted((ROOT/'ablation').glob('crop_pushdown_*.csv')):
        crop = read(path)
        figure, axes = plt.subplots(1,3,figsize=(12,3.4))
        for ax, key, label in zip(axes,('issued_read_bytes','h2d_bytes','images_per_s'),('Issued bytes/image','H2D bytes/image','Images/s')):
            for arm in ('on','off'):
                fractions=sorted({float(r['retained_fraction']) for r in crop if r['pushdown']==arm})
                values=[statistics.median(float(r[key])/(float(r['samples']) if 'bytes' in key else 1) for r in crop if r['pushdown']==arm and float(r['retained_fraction'])==f) for f in fractions]
                ax.plot(fractions,values,'o-',label=f'Pushdown {arm}')
            ax.set_xlabel('Retained crop area fraction')
            ax.set_ylabel(label)
            ax.legend()
        figure.suptitle('eFUN-DCT matched crop control; subset; separate H2D diagnostic')
        save(figure,path.stem)
    scaling=read(ROOT/'analysis/gpu_scaling.csv')
    if scaling:
        figure,ax=plt.subplots(figsize=(10,4))
        for i,r in enumerate(scaling):
            ax.bar(i-.18,float(r['gpu_e2e_scaling']),.36,color='#009e73',label='End-to-end' if i==0 else None)
            if r['gpu_model_scaling']:
                ax.bar(i+.18,float(r['gpu_model_scaling']),.36,color='#0072b2',label='Model only' if i==0 else None)
        ax.set_xticks(range(len(scaling)),[f'{r["model"]}\n{LABELS[r["pipeline"]]}' for r in scaling],rotation=20,ha='right')
        ax.set_ylabel('H100 / RTX 4090 throughput')
        ax.set_title('Same host, common batch and precision; shared-host observations')
        ax.legend()
        save(figure,'gpu_scaling')


if __name__ == '__main__':
    main()
