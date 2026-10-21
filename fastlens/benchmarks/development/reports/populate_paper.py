"""Populate the manuscript only from the completed first-pass paper result set."""
import csv
import json
from pathlib import Path
import re
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[4]
R=ROOT/'results'
PAPER=ROOT/'fastlens/docs/paper'


def rows(name):
    with (R/name).open() as f: return list(csv.DictReader(f))


def fmt(value,digits=1):
    return '--' if value is None or value=='' else f'{float(value):,.{digits}f}'


def tex(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(caption,label,headers,data,wide=True,textcols=1,group_first=False):
    kind='table*' if wide else 'table'
    spec='@{}'+'l'*textcols+'r'*(len(headers)-textcols)+'@{}'
    lines=[]
    for i,row in enumerate(data):
        if group_first and i and row[0]!=data[i-1][0]: lines.append(r'\midrule')
        lines.append(' & '.join(map(str,row))+r' \\')
    return '\n'.join([r'\begin{'+kind+'}[!t]',r'\centering',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\fontsize{8}{9}\selectfont\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{'+spec+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',
        *lines,r'\bottomrule',r'\end{tabular}',r'\end{'+kind+'}'])


def figure(file,caption,label):
    kind='figure*' if file=='01_deployment_inference' else 'figure'
    descriptions={
        '01_deployment_inference':'Vertical grouped bars compare four RGB input systems, PyTorch/CPU-DCT, and FastLens DCT on RTX 4090 and H100 for seven DCT workloads. CPU-DCT and FastLens use the same GPU model.',
        '02_pipeline_efficiency':'Grouped bars show inference throughput relative to the corresponding resident-input model throughput for SwinV2 and MobileNetV2 on both GPUs.',
        '03_crop_pushdown':'Three bar charts compare issued storage bytes, host-to-device bytes, and elapsed time with spatial pushdown off and on; both paths execute the same crop.',
        '04_core_ablations':'Grouped bars show optimized-to-control ratios for storage reads, host-to-device traffic, and elapsed time for spatial pushdown, direct construction, and compressed transfer.'}
    return '\n'.join([r'\begin{'+kind+'}[t]',r'\centering',
        r'\includegraphics[width=\linewidth]{figures/milestone_a/'+file+'.pdf}',
        r'\Description{'+descriptions[file]+'}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{'+kind+'}'])


def deployment_inference_table(deployment, matched):
    model_only = {(r['gpu'], r['dct_model']): r for r in rows('paper/deployment_model_only.csv')}
    data = []
    for r in deployment:
        reference = model_only[r['gpu'], r['dct_variant']]
        cpu = next(m['cpu_dct_images_per_s'] for m in matched
                   if (m['gpu'], m['dct_model']) == (r['gpu'], r['dct_variant']))
        data.append([tex(r['gpu']), tex(r['dct_variant']), fmt(reference['rgb_images_per_s'], 0),
                     *[fmt(r[s], 0) for s in ('PyTorch', 'DALI', 'CoorDL', 'FFCV')],
                     fmt(reference['dct_images_per_s'], 0), fmt(cpu, 0),
                     fmt(r['FastLens'], 0), fmt(r['deployment_ratio'], 3)])
    return table(
        r'Inference throughput (images/s). Model-only columns are independently timed medians with GPU-resident inputs, using the corresponding checkpoint, batch size, and precision. Input-system columns cover all 50,000 validation images. Deployment ratios compare FastLens with the fastest RGB input system, excluding model-only references. A dash denotes an unavailable measurement.',
        'tab:deployment-inference', ['GPU', 'DCT model', r'\shortstack{Model-only\\RGB}',
        r'\shortstack{PyTorch\\RGB}', r'\shortstack{DALI\\RGB}', r'\shortstack{CoorDL\\RGB}',
        r'\shortstack{FFCV\\RGB}', r'\shortstack{Model-only\\DCT}', r'\shortstack{PyTorch\\CPU-DCT}',
        r'\shortstack{\sys{}\\DCT}', 'Deploy. ratio'], data, textcols=2, group_first=True
    )



def training_window_section(training):
    """Deployment windows with aligned RGB crop decisions on both GPUs."""
    variants=[('ViT-Ti','ViT-Ti DCT','ViT-Ti DCT'),
              ('SwinV2-T','SwinV2-T','SwinV2-T DCT'),
              ('MobileNetV2','MobileNetV2','MobileNetV2-DCT24'),
              ('MobileNetV2','MobileNetV2-DCT32','MobileNetV2-DCT32'),
              ('ResNet-50','ResNet-50-DCT24','ResNet-50-DCT24'),
              ('ResNet-50','ResNet-50-DCT64','ResNet-50-DCT64'),
              ('EfficientNet-B0 reference','eFUN-DCT','eFUN-DCT (reference)')]
    systems=['PyTorch','DALI','CoorDL','FFCV']
    data=[]
    for gpu in ('RTX 4090','H100'):
        for rgb,dct,label in variants:
            baseline=[next(r for r in training if (r['gpu'],r['model'],r['pipeline'])==(gpu,rgb,p)) for p in systems]
            assert all(r.get('implementation_cohort','').startswith('training_expansion_') and int(r['trials'])==3 for r in baseline)
            native=next(r for r in training if (r['gpu'],r['model'],r['pipeline'])==(gpu,dct,'FastLens'))
            rates=[float(r['median_images_per_s']) for r in baseline]
            speed=float(native['median_images_per_s'])
            data.append([tex(gpu),tex(label),*[fmt(v,0) for v in rates],fmt(speed,0),fmt(speed/max(rates),2)])
    text=r"""\paragraph{Training deployment windows.}
Table~\ref{tab:training-window} compares seven DCT workloads with five
RGB model families on RTX~4090 and H100. Each trial measures 65,536 image
executions and 64 optimizer updates, after 32 excluded warm-up updates.
The microbatch is 64 and gradient accumulation is 16, giving an effective
batch of 1,024. ViT-Ti uses FP32; the other workloads use BF16 autocast.
Each configuration is measured three times with warm input caches and no
profiler. A shared RGB family is measured once and reused for its DCT variants.

Within each RGB family, the four systems share initialization, sample
order, crop and flip decisions, optimizer, precision, and update count.
All resize crops to $224\times224$ with bilinear interpolation without
antialiasing and apply identical normalization. DALI and CoorDL use native
mixed JPEG decoding; we check subsequent transformations against CPU
postprocessing of the decoded pixels. FFCV uses a keyed-crop adapter to
reproduce the prescribed augmentation sequence, with crop and bilinear
resizing parallelized across images.

The DCT path uses closed-pool crops, RandAugment, and Mixup, and retains
its configured model and input representation. Each window consumes
65,536 distinct samples; skipped DCT pool-tail microbatches are recorded.
These RGB/DCT differences make the reported ratios deployment ratios,
not causal input-system speedups. The eFUN comparison additionally uses
EfficientNet-B0 as a reference architecture rather than an exact counterpart.
"""
    text+='\n'+table('Training throughput on RTX 4090 and H100 (images/s), median of three 65,536-image windows. RGB columns use the corresponding family model; DCT variants share those RGB observations. Ratio = FastLens DCT / fastest RGB median. ViT-Ti uses FP32; other rows use BF16 autocast. EfficientNet-B0 is the eFUN reference.',
        'tab:training-window',['GPU','DCT workload',*systems,r'\sys{}','Deployment ratio'],data,True,textcols=2,group_first=True)
    text+=r"""
Training deployment rankings depend on the model and DCT profile.
ViT-Ti is near parity with the fastest RGB median on both GPUs.
SwinV2-T and MobileNetV2-DCT32 exceed that median, while
MobileNetV2-DCT24 and both ResNet-50 variants remain below it.
The fastest RGB system is selected independently for each architecture
and GPU. The DCT variants share those RGB measurements, so a change in
channel profile changes the DCT deployment while retaining the same
reference. Throughput must be read alongside checkpoint accuracy and
input representation, particularly for the MobileNetV2 profiles.

Small median differences should be read in light of trial variability.
For example, RTX~4090 MobileNetV2-DCT24 spans 1,440--2,146 images/s,
and H100 RGB SwinV2-T/DALI has a 26.78\% coefficient of variation.
FFCV also exhibits variability, reaching 11.80\% for H100 MobileNetV2.
The released trial ranges accompany the medians; near-parity results
do not establish a stable ranking. These windows measure execution cost
under each deployment's training recipe, rather than time to a target accuracy.

"""
    return text


def training_epoch_section(epochs):
    # Display verified full-epoch pairs; leave unavailable comparisons blank.
    models=('ViT-Ti DCT','SwinV2-T','MobileNetV2-DCT24','MobileNetV2-DCT32',
            'ResNet-50-DCT24','ResNet-50-DCT64','eFUN-DCT')
    verified={(r['gpu'],r['model'],r['pipeline']):r for r in epochs
              if r['valid']=='True' and r['strict_matched']=='True'}
    data=[];ratios={}
    for gpu in ('RTX 4090','H100'):
        for model in models:
            control=verified.get((gpu,model,'CPU-DCT'))
            native=verified.get((gpu,model,'FastLens'))
            if control is None or native is None:
                data.append([tex(gpu),tex(model),'--','--','--'])
                continue
            assert control['initial_model_hash']==native['initial_model_hash']
            assert all(int(r['samples'])==1281167 and int(r['optimizer_updates'])==1252 for r in (control,native))
            ratio=float(control['elapsed_s'])/float(native['elapsed_s'])
            ratios[gpu,model]=ratio
            data.append([tex(gpu),tex(model),fmt(control['elapsed_s'],0),fmt(native['elapsed_s'],0),fmt(ratio,2)])
    text=r'''\paragraph{Complete matched training epochs.}
Table~\ref{tab:full-training} compares CPU-DCT and \sys{} over a complete training epoch,
with one measured epoch per model, input path, and GPU.
Each epoch consumes 1,281,167 images and performs 1,252 optimizer
updates, including partial batches. Initialization, sample sequence,
logical crop and augmentation decisions, optimizer, and effective-batch
semantics are matched between CPU-DCT and \sys{}. Deterministic sample
comparisons verify numerical agreement of model inputs and parameter
updates.
The short excluded warm-up restores initialization and does not consume
a complete training epoch. ViT-Ti uses FP32; the other reported models
use BF16 autocast.
'''
    text+=f'Measured system speedups range from {min(ratios.values()):.2f} to {max(ratios.values()):.2f}'+r'''$\times$.

'''
    text+='\n'+table('Full matched DCT epoch times, excluding warm-up; one measured epoch per configuration. System speedup = CPU-DCT time / FastLens time. A dash denotes an unreported comparison.',
        'tab:full-training',['GPU','DCT model',r'\shortstack{CPU-DCT\\(s)}',r'\shortstack{\sys{}\\(s)}',r'\shortstack{System\\speedup}'],
        data,False,textcols=2,group_first=True)
    text+=r'''
The larger gains for MobileNetV2-DCT32 than for ResNet-50-DCT24
show that input preparation contributes differently to these workloads.
The full epochs test whether the matched input-path benefit persists
through an entire dataset traversal, including partial batches.
Checkpoint accuracy and training throughput answer separate questions;
these epochs do not measure convergence or time to quality.
'''
    return text


def main(*, training_only=False):
    milestone=json.loads((R/'paper/milestone_a.json').read_text())
    assert milestone['status']=='COMPLETE' and all(milestone['requirements'].values())
    path=PAPER/'sections/07_evaluation.tex'
    original=path.read_text()
    bibliography=(PAPER/'bibliography/references.bib').read_bytes()
    citations=lambda text:set(key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',text) for key in group.split(','))
    before=citations(original)
    if training_only:
        start=original.index(r'\paragraph{Training deployment windows.}')
        end=original.index(r'\paragraph{Complete matched training epochs.}',start)
        updated=original[:start]+training_window_section(rows('p0/training_window_summary.csv'))+'\n\n'+original[end:]
        assert before<=citations(updated), 'An existing citation would be removed'
        path.write_text(updated)
        print('Updated training deployment windows only; other manuscript sections preserved.')
        return
    deployment=rows('p0/deployment_ratios.csv')
    deployment=[r for r in deployment if r['dct_top1']]
    deployment.sort(key=lambda r:('RTX 4090','H100').index(r['gpu']))
    inference=rows('p0/deployment_inference_summary.csv')
    quality=rows('quality/accuracy_deltas.csv')
    matched=[r for r in rows('p0/matched_dct_inference.csv') if r['correctness']=='PASS']
    training=rows('p0/training_window_summary.csv')
    epochs=rows('p1/full_epoch_training.csv')
    efficiencies=rows('analysis/inference_pipeline_efficiency.csv')
    h2d=rows('resources/h2d.csv'); memory=rows('resources/memory.csv'); storage=rows('resources/storage.csv')
    crop=rows('ablation/crop_pushdown.csv'); direct=rows('ablation/direct_vs_grid.csv')
    systems=['PyTorch','DALI','CoorDL','FFCV','FastLens']
    main_deploy=[float(r['deployment_ratio']) for r in deployment if r['relationship']!='reference_only']
    match_min=min(float(r['matched_system_speedup']) for r in matched)
    match_max=max(float(r['matched_system_speedup']) for r in matched)
    measured_above=sum(v>1 for v in main_deploy)
    deployment_finding=(f'Across the verified architecture pairs, {measured_above} of {len(main_deploy)} '
        f'model/GPU deployment observations exceed the fastest RGB median; ratios range from '
        f'{min(main_deploy):.3f} to {max(main_deploy):.3f}'+r'$\times$.'+
        ' Small observed differences do not establish a reliable performance ranking.')
    body=[r'''\subsection{End-to-End Performance and Model Quality}
\label{sec:eval-e2e}

\paragraph{Checkpoint quality.}
Table~\ref{tab:model-quality} reports checkpoint-normal evaluation on the
50,000-image ImageNet-512 validation view. The RGB reference accuracy is
measured once per architecture; the per-loader observations retain their
own measured accuracy. Positive and negative deltas are reported in
percentage points. The DCT checkpoints improve top-1 accuracy for the
two Transformer families, while both MobileNetV2 variants lose accuracy.
ResNet-50-DCT24 improves accuracy but DCT64 does not: a larger channel
profile alone does not guarantee higher quality. EfficientNet-B0 is a
reference architecture for eFUN rather than an exact RGB counterpart.

''']
    body.append(table('ImageNet-512 checkpoint accuracy. Delta is DCT minus RGB in percentage points. The eFUN row uses an RGB reference architecture.',
        'tab:model-quality',['DCT workload',r'RGB (\%)',r'DCT (\%)',r'$\Delta$ (pp)'],
        [[tex(r['dct_model']),fmt(r['rgb_top1'],3),fmt(r['dct_top1'],3),fmt(r['accuracy_delta_pp'],3)] for r in quality],False))
    body.append(r'''\paragraph{Deployment inference.}
Table~\ref{tab:deployment-inference} and Figure~\ref{fig:deployment-inference}
compare complete RGB and DCT deployments. The four RGB systems execute
the corresponding RGB checkpoint; \sys{} executes the DCT checkpoint.
The deployment ratio divides \sys{} throughput by the fastest measured
RGB system for that architecture and GPU.
'''+deployment_finding+'\n')
    body.append(r'''ViT-Ti and SwinV2-T lie near the RGB throughput frontier on both GPUs.
In contrast, the MobileNetV2 and ResNet-50 DCT deployments remain slower
than the fastest RGB systems. MobileNetV2 also incurs a checkpoint
accuracy loss, whereas SwinV2-T improves accuracy. The useful comparison
is therefore the joint throughput--quality outcome, rather than a
throughput ratio alone. The matched DCT controls below separate input
processing from the model execution cost behind these deployment results.
''')
    body.append(deployment_inference_table(deployment, matched))
    body.append(figure('01_deployment_inference',r'Inference throughput with matched DCT controls. RGB and PyTorch/CPU-DCT bars show medians with min--max whiskers; \sys{} bars show measured throughput. CPU-DCT and \sys{} use the same GPU model. RGB--DCT comparisons concern complete deployments. Accuracy appears in Table~\ref{tab:model-quality}; eFUN uses an RGB reference architecture. RGB results are reused across DCT variants. MV2 and RN50 denote MobileNetV2 and ResNet-50; suffixes identify DCT channel counts.','fig:deployment-inference'))
    body.append(r'''\paragraph{Matched DCT inference.}
Table~\ref{tab:matched-inference} holds the DCT model, checkpoint, sample
identities, batch size, precision, and logical input transformation fixed.
We compare all 50,000 predictions between the two paths; representative
model-input comparisons check their numerical transformation.
These ratios hold model semantics fixed and compare the two input paths.
PyTorch/CPU-DCT performs coefficient preparation on the CPU; both DCT
paths execute the model on the same GPU. The control covers all seven
DCT workloads on both GPUs, making input-system effects distinguishable
from the RGB--DCT model differences in the deployment comparison.
'''+f'The measured system ratios span {match_min:.3f}--{match_max:.3f}'+r'$\times$.'+'\n')
    body.append(r'''The large MobileNetV2 ratios indicate a substantial CPU-DCT input
bottleneck, yet its deployment remains below the fastest RGB system.
ResNet-50 shows the opposite boundary: changing the input path has little
effect on same-model throughput, while the RGB deployment remains much
faster. Thus, removing CPU coefficient preparation does not by itself make
a DCT architecture the fastest deployable design. Attributing gains to
individual input stages requires fixed-input mechanism experiments.
''')
    resnet=next(r for r in matched if (r['gpu'],r['dct_model'])==('H100','ResNet-50-DCT24'))
    resnet_dct=next(r for r in efficiencies if (r['gpu'],r['model'],r['pipeline'])==('H100','ResNet-50-DCT24','FastLens'))
    resnet_rgb=next(r for r in efficiencies if (r['gpu'],r['model'],r['pipeline'])==('H100','ResNet-50 RGB','PyTorch'))
    body.append('On H100, ResNet-50-DCT24 achieves '+fmt(resnet['cpu_dct_images_per_s'],0)+
        ' images/s with CPU-DCT and '+fmt(resnet['imagelanes_images_per_s'],0)+r' with \sys{}, compared with '+
        fmt(resnet_rgb['end_to_end_images_per_s'],0)+' for PyTorch RGB. The resident-input DCT model reaches '+
        fmt(resnet_dct['model_only_images_per_s'],0)+' images/s, versus '+fmt(resnet_rgb['model_only_images_per_s'],0)+
        r' for RGB; \sys{} sustains '+fmt(100*float(resnet_dct['pipeline_efficiency']),1)+r'''\% of the DCT rate.
The DCT network uses stride one in its second residual stage, whereas
the RGB network uses stride two. Its larger downstream feature maps
increase convolutional work. These measurements indicate that model
execution, rather than exposed input preparation, dominates this workload;
reducing input traffic alone therefore offers limited end-to-end benefit.
''')
    models=list(dict.fromkeys(r['dct_model'] for r in matched))
    matched_table=[]
    for gpu in ('RTX 4090','H100'):
        for model in models:
            r=next(r for r in matched if r['gpu']==gpu and r['dct_model']==model)
            matched_table.append([tex(gpu),tex(model),fmt(r['cpu_dct_images_per_s'],0),fmt(r['imagelanes_images_per_s'],0),fmt(r['matched_system_speedup'],3)])
    body.append(table('Matched DCT inference throughput (images/s). Ratio = FastLens / CPU-DCT, with model and checkpoint fixed.',
        'tab:matched-inference',['GPU','DCT model','CPU-DCT',r'\sys{}','Ratio'],matched_table,False,textcols=2,group_first=True))
    body.append(training_window_section(training))
    body.append(training_epoch_section(epochs))
    body.append(r'''\paragraph{Resident-input reference and GPU scaling.}
To distinguish model execution cost from input-path overhead, we measure
each model with resident GPU inputs, preserving its checkpoint, batch
size, and precision. Figure~\ref{fig:pipeline-efficiency} reports
$E_{\mathrm{pipe}}=T_{\mathrm{end\text{-}to\text{-}end}}/T_{\mathrm{resident}}$.
Each RGB or DCT pipeline is normalized to its own model. This measured
reference is not a strict hardware upper bound; independently timed
ratios can lie slightly above one.
''')
    data=[next(r for r in efficiencies if (r['gpu'],r['model'],r['pipeline'])==(g,'MobileNetV2-DCT24','FastLens')) for g in ('RTX 4090','H100')]
    scaling=float(data[1]['model_only_images_per_s'])/float(data[0]['model_only_images_per_s'])
    endscaling=float(data[1]['end_to_end_images_per_s'])/float(data[0]['end_to_end_images_per_s'])
    body.append(f'For MobileNetV2-DCT24, the H100/4090 resident-input ratio is {scaling:.2f}'+r'$\times$'+
        f', compared with {endscaling:.2f}'+r'$\times$'+
        f' end to end. Normalized throughput falls from {100*float(data[0]["pipeline_efficiency"]):.1f}'+r'\%'+
        f' to {100*float(data[1]["pipeline_efficiency"]):.1f}'+r'''\%.
The input path absorbs part of the benefit of faster model execution,
so the faster GPU does not yield a proportional deployment gain.
''')
    body.append(figure('02_pipeline_efficiency','Inference throughput relative to each deployment\'s own resident-input model reference. FastLens uses DCT; the other systems use RGB. Mobile denotes MobileNetV2-DCT24 and its RGB counterpart.','fig:pipeline-efficiency'))
    body.append(r'''\subsection{Data Movement and Resource Costs}
\label{sec:eval-cost}

\paragraph{Storage and preparation.}
Table~\ref{tab:storage-cost} compares persistent representations, grouped
by dataset split and sample count. The FFCV training file contains
98,560 images for the bounded windows: 32,768 warm-up images, 65,536
measured images, and 256 additional images for reader prefetch.
The full-epoch experiments compare CPU-DCT with \sys{} and do not use
this FFCV subset. Amplification is relative to the
512-pixel JPEGs for exactly that split or subset. Source coefficients,
prepared access indexes, and FFCV materialization have different storage
costs; the prepared \sys{} rows include both coefficients and indexes.
The compact representation used by SwinV2 inference is reported
separately from the block-major representation used by the other workloads.

Table~\ref{tab:preparation-cost} separates preparation from verification.
Its timings cover the operations named in the table, rather than a
uniform conversion benchmark. For a measured complete conversion, the
break-even number of passes is
$T_{\mathrm{conversion}}/(T_{\mathrm{baseline}}-T_{\sys{}})$,
provided that the denominator is positive. The baseline must match the
comparison of interest: a CPU-DCT saving does not imply amortization
against an RGB deployment.
''')
    labels={'jpeg':'JPEG','imagelanes_source':r'\sys{} source',
        'imagelanes_prepared':r'\sys{} prepared','imagelanes_compact_v3':r'\sys{} compact',
        'ffcv_beton':'FFCV Beton (raw RGB)'}
    persistent=[r for r in storage if r['dataset_view']!='original' and r['representation'] in labels]
    split_order=('val','train','train_p0_selection')
    persistent.sort(key=lambda r:split_order.index(r['split']))
    split_labels={'val':'Validation (50,000)','train':'Training (1,281,167)',
                  'train_p0_selection':'Subset (98,560)'}
    storage_table=[]
    for r in persistent:
        storage_table.append([split_labels[r['split']],labels[r['representation']],
            fmt(float(r['payload_bytes'])/1e9,3),fmt(float(r['metadata_bytes'])/1e6,2),
            fmt(float(r['index_bytes'])/1e6 if r['index_bytes'] else None,2),
            fmt(float(r['representation_bytes'])/1e9,3),fmt(r['storage_amplification'],2)])
    body.append(table(r'Persistent storage in decimal GB/MB. Total includes file padding. Amplification uses the corresponding 512-pixel JPEG sample set; the FFCV training subset is not a full training representation. CoorDL runtime caching is excluded.',
        'tab:storage-cost',['Dataset (images)','Representation',r'\shortstack{Payload\\GB}',
        r'\shortstack{Metadata\\MB}',r'\shortstack{Index\\MB}',r'\shortstack{Total\\GB}','Ampl.'],
        storage_table,textcols=2,group_first=True))
    prep=[]
    for representation,split,operation in [
        ('imagelanes_prepared','val',r'\sys{} preparation'),
        ('imagelanes_prepared','train',r'\sys{} compression only'),
        ('ffcv_beton','val','FFCV conversion'),
        ('ffcv_beton','train_p0_selection','FFCV writer only')]:
        r=next(r for r in storage if (r['representation'],r['split'])==(representation,split))
        prep.append([operation,fmt(r['samples'],0),fmt(r['conversion_s'],1),fmt(r['verification_s'],1)])
    body.append(table(r'Offline preparation and verification times (s). Training compression excludes access-index construction. FFCV subset time covers its writer; verification checks all labels and ordinals and samples pixels. A dash denotes an unavailable timing.',
        'tab:preparation-cost',['Operation','Images','Prep.','Verify'],prep,False))
    body.append(r'''\paragraph{H2D and memory.}
Table~\ref{tab:resource-cost} reports 16,384-image training windows on the
RTX~4090. CUPTI captures run separately from throughput measurements.
We distinguish application-issued storage bytes from CUDA H2D bytes;
neither is a measurement of physical NVMe traffic. GPU residency is
sampled every 100~ms throughout the unprofiled memory window. The main
process RSS and process-tree RSS are reported separately because loaders
may use worker processes; the latter can double-count shared pages.
Framework allocation/reservation, native pools, and pinned-memory
counters are retained separately from the process-level peaks. We do not
sum independent high-water marks. Input submission and GPU idle intervals
provide timing context, but an idle interval alone does not identify
exposed input wait.
''')
    resource_table=[]
    for model in ('swinv2','mobilenet'):
        for s in systems:
            h=next(r for r in h2d if (r['model'],r['pipeline'])==(model,s.lower()))
            m=next(r for r in memory if (r['model'],r['pipeline'])==(model,s.lower()))
            resource_table.append([tex('SwinV2-T' if model=='swinv2' else 'MobileNetV2'),tex(s),fmt(float(h['h2d_bytes_per_image'])/1e6,3),fmt(float(m['gpu_peak_process_bytes'])/2**30,2),fmt(float(m['host_rss_peak_bytes'])/2**30,2),fmt(float(m['process_tree_rss_peak_bytes'])/2**30,2),fmt(float(m['pinned_peak_bytes'])/2**20,2) if m['pinned_peak_bytes'] else '--'])
    body.append(table('RTX 4090 resource costs for 16,384-image training windows. H2D uses a separate profile; memory uses an unprofiled run. PyTorch pinned allocation excludes untracked loader/native allocations.',
        'tab:resource-cost',['Model','System',r'\shortstack{H2D\\MB/image}',r'\shortstack{GPU\\GiB}',r'\shortstack{Main RSS\\GiB}',r'\shortstack{Tree RSS\\GiB}',r'\shortstack{Torch pinned\\MiB}'],resource_table,textcols=2,group_first=True))
    cache=next(r for r in storage if r['representation']=='coordl_jpeg_cache')
    body.append(r'CoorDL also uses a tmpfs JPEG cache: the measured '+fmt(cache['samples'],0)+
        f'-image cache occupies {float(cache["representation_bytes"])/1e9:.3f}'+r'''~GB.
This runtime cache is separate from persistent representation size and
from process RSS.

For both workloads, \sys{} transfers 0.078 MB/image, below the RGB
systems in Table~\ref{tab:resource-cost}. Its whole-process GPU peak is
higher, however, reflecting a different tradeoff between transfer volume
and resident buffers. Lower H2D traffic therefore does not imply lower
GPU memory demand. These deployment costs include differences in model
representation; the fixed-input experiments below isolate system mechanisms.
''')
    body.append(r'''\subsection{Matched Mechanism Experiments}
\label{sec:eval-ablation}

The following controls follow the input path from spatial selection,
through compressed transfer and decoding, to tensor construction.
They use MobileNetV2-DCT24 on the RTX~4090 and a fixed 16,384-image
physical-prefix request. All cases use the same checkpoint, FP32,
batch size 64, centered $448\times448$ source crop, all 64 source
frequencies, and final model tensor. Crop pushdown compares selective
reads with full-source reads followed by the identical crop. Tensor
construction compares direct output with full component grids followed
by gathering, while holding issued input bytes fixed. Full prediction
sequences agree, and representative model tensors agree exactly.
''')
    c,n=crop
    body.append(r'\paragraph{Spatial pushdown.} '+
        f'The control issues {float(c["issued_read_bytes"])/1e9:.3f} GB and the optimized path issues {float(n["issued_read_bytes"])/1e9:.3f} GB. '+
        f'H2D traffic changes from {float(c["h2d_bytes"])/1e9:.3f} to {float(n["h2d_bytes"])/1e9:.3f} GB, '+
        f'and unprofiled elapsed time from {float(c["elapsed_s"]):.3f} to {float(n["elapsed_s"]):.3f} s. '+
        'Spatial selection reduces issued bytes but does not yield a wall-clock gain for this request.\n')
    body.append(figure('03_crop_pushdown',r'Matched spatial pushdown, MobileNetV2-DCT24/4090: 16,384 images, K64, 76.56\% retained area. Both paths execute the same crop. Read denotes issued storage bytes; H2D is profiled separately from elapsed time.','fig:crop-pushdown'))
    body.append(figure('04_core_ablations','Matched controls on RTX 4090: spatial pushdown, direct construction, and compressed transfer. Lower ratios mean fewer bytes or less time. Direct construction holds reads fixed; compressed transfer compares FastLanes with raw int16.','fig:core-ablations'))
    compression_path=R/'ablation/fastlanes_vs_raw_int16.csv'
    if compression_path.exists():
        compression=rows('ablation/fastlanes_vs_raw_int16.csv')
        raw=next(r for r in compression if r['case']=='raw_int16')
        packed=next(r for r in compression if r['case']=='compressed')
        body.append(r'\paragraph{Compressed versus raw coefficients.} '+
            'The matched 16,384-image control preserves coefficients, grouping, crop, frequency selection, and model output. '+
            f'Raw int16 takes {float(raw["elapsed_s"]):.3f} s and FastLanes takes {float(packed["elapsed_s"]):.3f} s. '+
            f'Application-issued reads decrease from {float(raw["issued_read_bytes"])/1e9:.3f} to {float(packed["issued_read_bytes"])/1e9:.3f} GB; '+
            f'H2D traffic decreases from {float(raw["h2d_bytes"])/1e9:.3f} to {float(packed["h2d_bytes"])/1e9:.3f} GB. '+
            'This control measures the joint effect of compressed representation and decoding on this request.\n')
    c,n=direct
    body.append(r'\paragraph{Direct tensor construction.} '+
        f'Both paths issue {float(c["issued_read_bytes"])/1e9:.3f} GB. '+
        f'Direct output reduces elapsed time from {float(c["elapsed_s"]):.3f} to {float(n["elapsed_s"]):.3f} s '+
        f'and whole-process GPU residency from {float(c["gpu_peak_process_bytes"])/2**30:.2f} to {float(n["gpu_peak_process_bytes"])/2**30:.2f} GiB. '+
        f'H2D traffic is nearly unchanged ({float(c["h2d_bytes"])/1e9:.3f} versus {float(n["h2d_bytes"])/1e9:.3f} GB). '+
        r'''The gain comes from avoiding full component-grid materialization
and the subsequent gather, rather than reading fewer selected inputs.
''')
    body.append(r'''These controls separate physical input reduction from intermediate
materialization cost. For the measured crop, fewer storage bytes yield
little H2D reduction and no elapsed-time benefit, whereas direct output
reduces time without changing issued reads. Their scope is a fixed
spatial request and $K=64$; frequency selectivity and decompression
placement are not varied. Because pipeline stages overlap, the component
ratios are not additive fractions of the matched-system speedup.

\subsection{Discussion}
\label{sec:eval-discussion}

The results connect three determinants of deployment performance:
checkpoint quality, model execution cost, and input preparation.
\sys{} substantially reduces the CPU-DCT bottleneck for MobileNetV2,
but the resulting deployment still trails the fastest RGB system and
has lower checkpoint accuracy. ResNet-50 instead exposes a model-execution
limit, while the Transformer deployments approach the RGB throughput
frontier with higher measured checkpoint accuracy.

The mechanism experiments explain why byte savings alone are an
incomplete performance argument. Eliminating intermediate tensors can
matter even when reads are unchanged, and spatial selection may save
storage traffic without shortening execution. These findings concern
warm-cache execution and the evaluated crop and frequency request.
Training results characterize execution cost under the stated recipes;
long-horizon quality and time to quality require paired convergence measurements.
''')
    start=original.index(r'\subsection{End-to-End Performance and Model Quality}')
    updated=original[:start]+'\n\n'.join(body)+'\n'
    headline=f'On ImageNet-512, matched DCT inference throughput ratios range from {match_min:.2f} to {match_max:.2f}'+r'$\times$'+f'. Across verified RGB/DCT architecture pairs, deployment ratios range from {min(main_deploy):.2f} to {max(main_deploy):.2f}'+r'$\times$'+r'. ViT-Ti and SwinV2-T approach the fastest RGB deployments, whereas MobileNetV2 and ResNet-50 remain below them. These deployment differences include model execution and input processing, and checkpoint accuracy changes with the DCT variant.'
    section_updates={}
    for name in ('00_abstract.tex','01_introduction.tex','09_conclusion.tex'):
        section=PAPER/'sections'/name
        source=section.read_text()
        result,count=re.subn(r'On ImageNet-512, matched DCT inference (?:throughput )?ratios range from .*?checkpoint accuracy changes with the DCT variant\.',lambda m:headline,source,flags=re.S)
        assert count==1, f'Expected one measured headline in {section}'
        if name=='09_conclusion.tex':
            start=result.index('The fixed-input controls')
            result=result[:start]+'The fixed-input controls distinguish storage selection, tensor construction, and compressed representation. The results support evaluating compressed input systems together with model quality, model execution cost, and recurring data movement.\n'
        assert citations(source)<=citations(result), f'A citation would be removed from {section}'
        section_updates[section]=result
    assert before<=citations(updated),'An existing citation would be removed'
    assert (PAPER/'bibliography/references.bib').read_bytes()==bibliography
    assert r'\texttt{TBD}' not in updated
    target=PAPER/'figures/milestone_a'
    target.mkdir(parents=True,exist_ok=True)
    for source in sorted((R/'figures/milestone_a').glob('*.pdf')): shutil.copyfile(source,target/source.name)
    path.write_text(updated)
    for section,result in section_updates.items(): section.write_text(result)
    print('Updated included evaluation, result-bearing abstract/introduction/conclusion, and four vector figures; citations preserved.')


if __name__=='__main__': main()
