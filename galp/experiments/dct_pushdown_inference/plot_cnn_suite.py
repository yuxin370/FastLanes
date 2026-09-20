"""Export paper-sized figures from the completed CNN CSV summaries."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_cnn_suite import CONFIGS, DATA, PATHS


def save(fig, root, name):
    fig.savefig(root/f'{name}.png', dpi=180)
    fig.savefig(root/f'{name}.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DATA/'runs/dctnet_mobilenet24/rtx4090_cnn_complete_20260914')
    args = p.parse_args()
    root = args.output_dir/'figures'
    root.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'cnn_inference.csv').open() as f:
        inference = list(csv.DictReader(f))
    with (args.output_dir/'cnn_training.csv').open() as f:
        training = list(csv.DictReader(f))
    with (args.output_dir/'cnn_training_memory_probes.csv').open() as f:
        memory = list(csv.DictReader(f))
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(12,8), layout='constrained')
    colors = ['#999999','#666666','#e5ae38','#d45b5b','#b5d2e5','#679cc0','#8dc9a7','#29956a']
    for ax, (model, _, _) in zip(axes.flat, CONFIGS):
        rows = {r['route']:r for r in inference if r['model'] == model}
        routes = [route for route, _ in PATHS if route in rows]
        rates = [float(rows[route]['images_per_second']) for route in routes]
        bars = ax.barh(routes, rates, color=colors)
        ax.bar_label(bars, labels=[f'{v:.0f}' for v in rates], padding=3, fontsize=9)
        ax.invert_yaxis()
        ax.set(title=model, xlabel='Images/s (50,000 validation images)', xlim=(0,5500))
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('CNN inference end-to-end throughput — RTX 4090, FP32, batch 64')
    save(fig,root,'inference_e2e')

    fig, ax = plt.subplots(figsize=(9,4.5), layout='constrained')
    x=np.arange(len(CONFIGS))
    for i, mode in enumerate(('off','on')):
        values = [float(next(r for r in inference if r['model']==model and r['route']==f'N projected {mode}')['requested_file_bytes'])/1e9
                  for model,_,_ in CONFIGS]
        bars=ax.bar(x+(i-.5)*.35,values,.35,label=mode,color=['#999999','#29956a'][i])
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set(xticks=x,xticklabels=[r[0] for r in CONFIGS],ylabel='Requested payload (GB)',
           title='Fixed-interface I/O pushdown: complete 50K validation')
    ax.legend(title='Pushdown');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,root,'inference_payload')

    fig, ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    x=np.arange(len(memory))
    for i,(key,scale,label,color) in enumerate([
        ('torch_peak_allocated_bytes',2**30,'Torch allocated','#679cc0'),
        ('torch_peak_reserved_bytes',2**30,'Torch reserved','#b5d2e5'),
        ('nvml_process_peak_mib',1024,'NVML process total','#29956a')]):
        values=[float(r[key])/scale for r in memory]
        bars=ax.bar(x+(i-1)*.25,values,.25,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=9)
    ax.set(xticks=x,xticklabels=[r['model'] for r in memory],ylabel='Peak GPU memory (GiB)',
           title='B6 memory probe: 8,192 training + 1,000 validation images\nSeparate measurements, not additive; not full-epoch peaks')
    ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,root,'training_memory_probe')

    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for ax,(model,_,_) in zip(axes.flat,CONFIGS):
        drawn=False
        for route,color in [('B6','#29956a'),('A0 JPEG','#e1812c')]:
            rows=sorted((r for r in training if r['model']==model and r['route']==route and r['full_epoch']=='True'),key=lambda r:int(r['epoch']))
            if rows:
                drawn=True
                ax.plot([int(r['epoch']) for r in rows],[float(r['top1']) for r in rows],marker='o',label=route,color=color)
        if drawn:
            ax.legend()
        else:
            ax.text(.5,.5,'Pending',transform=ax.transAxes,ha='center')
        ax.set(title=model,xlabel='Completed epoch',ylabel='50K validation Top-1 (%)',xticks=[1,2],xlim=(.8,2.2))
        ax.grid(alpha=.2)
    fig.suptitle('Early training from scratch: two epochs remain inside the 10K-update warmup\nOnly completed epochs shown; no final-convergence claim')
    save(fig,root,'early_validation')
    print(root)


if __name__=='__main__':
    main()
