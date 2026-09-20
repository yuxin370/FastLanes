"""Summarize completed CNN artifacts without treating running jobs as results."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT/'galp/data/system_rgbnomore/e2e_v3'
CONFIGS = [('MobileNetV2 DCT-24', 'dctnet_mobilenet24', 'rtx4090_io_pushdown_20260914'),
           ('MobileNetV2 DCT-32', 'dctnet_mobilenet32', 'rtx4090_cnn_complete_20260914'),
           ('ResNet-50 DCT-24', 'dctnet_static24', 'rtx4090_io_pushdown_20260914'),
           ('ResNet-50 DCT-64', 'dctnet_static64', 'rtx4090_cnn_complete_20260914')]
PATHS = [('RGB PyTorch', 'RGB_pytorch_50000.json'), ('RGB DALI', 'RGB_dali_50000.json'),
         ('R', 'R_50000.json'), ('O', 'O_50000.json'),
         *[(f'N {layout} {pushdown}', f'native_{layout}_{pushdown}/N_50000.json')
           for layout in ('grid', 'projected') for pushdown in ('off', 'on')]]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def number(value, places=3):
    return '未测' if value is None else f'{value:.{places}f}'


def contended(run, directory, suffix):
    profile = directory.replace('dctnet_static', 'resnet').replace('dctnet_', '')
    path = DATA/'runs/dctnet_mobilenet24'/run/'queue'/f'{profile}_{suffix}.gpu_occupancy.json'
    return path.exists() and bool(json.loads(path.read_text())['competing_pids'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DATA/'runs/dctnet_mobilenet24/rtx4090_cnn_complete_20260914')
    p.add_argument('--inference-run', default='rtx4090_cnn_memory_20260914')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference, training, probes, memory, data_only, traces = [], [], [], [], [], []
    for name, directory, run in CONFIGS:
        for route, relative in PATHS:
            path = DATA/'runs'/directory/run/'full'/relative
            measured = DATA/'runs'/directory/args.inference_run/'full'/relative
            metric = relative.split('/')[0] if '/' in relative else relative.removesuffix('_50000.json')
            metric_path = measured.parents[1] / f'memory_{metric}.json' if '/' in relative else measured.parent / f'memory_{metric}.json'
            if measured.exists() and metric_path.exists() and not contended(args.inference_run, directory, 'inference_memory'):
                m = json.loads(metric_path.read_text())
                if m['returncode'] == 0:
                    path = measured
                    memory.append(dict(model=name, route=route, nvml_process_peak_mib=m['nvml_process_peak_mib'],
                        main_rss_bytes=m['cpu_peaks']['main_rss_bytes'], tree_rss_sum_bytes=m['cpu_peaks']['tree_rss_sum_bytes'], result=str(metric_path)))
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            shards = r.get('native_shards', [])
            requested = sum(s['compressed_payload_bytes_read'] for s in shards) if shards else r.get('logical_read_bytes', r.get('source_bytes'))
            inference.append(dict(model=name, route=route, samples=r['samples'], top1=r['top1'], top5=r['top5'], ce=r['ce'],
                e2e_seconds=r['e2e_seconds'], images_per_second=r['samples']/r['e2e_seconds'],
                input_wait_seconds=r.get('input_wait_seconds'), model_seconds=r['model_seconds'],
                requested_file_bytes=requested, h2d_payload_bytes=sum(s['workset_upload_dma_bytes'] for s in shards) if shards else None,
                torch_peak_allocated_bytes=r.get('peak_cuda_allocated_bytes'),
                native_peak_in_use_bytes=max((s['galp_native_device_peak_in_use_bytes'] for s in shards), default=None),
                agreement_with_R=r.get('prediction_agreement_with_R'), source_version='imagenet_512',
                result=str(path), checkpoint=r['checkpoint']))
        folders = [('B6', DATA/'runs'/directory/'rtx4090_native_training_optimized_20260914/B6_full_m4_inplace'),
                   ('A0 JPEG', DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914/jpeg_full')]
        if directory.endswith('24'):
            folders += [(backend, DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914'/f'{backend}_full')
                        for backend in ('rgb_pytorch', 'rgb_d2', 'rgb_d3')]
        for route, folder in folders:
            final = json.loads((folder/'training.json').read_text()) if (folder/'training.json').exists() else {}
            for epoch_file in sorted(folder.glob('epoch_*.json')):
                r = json.loads(epoch_file.read_text())
                v = r.get('validation', {})
                training.append(dict(model=name if not route.startswith('rgb_') else name.split(' DCT-')[0]+' RGB',
                    route=route, epoch=r['epoch']+1, samples=r['samples'], unique_samples=r['unique_samples'],
                    full_epoch=r['full_epoch'], e2e_seconds=r['seconds'], images_per_second=r['images_per_second'],
                    input_wait_seconds=r['input_wait_seconds'], model_stream_seconds=r['model_stream_seconds'],
                    training_ce=r['ce'], validation_samples=v.get('samples'), top1=v.get('top1'), top5=v.get('top5'), ce=v.get('ce'),
                    nvml_process_peak_mib=final.get('nvml_process_peak_mib'),
                    torch_peak_allocated_bytes=final.get('torch_peak_allocated'), cpu_peak_rss_kib=final.get('cpu_peak_rss_kib'),
                    full_run_complete=bool(final), result=str(epoch_file)))
        probe_path=DATA/'runs'/directory/'rtx4090_native_training_optimized_20260914/B6_probe_m4_inplace/training.json'
        if probe_path.exists():
            r=json.loads(probe_path.read_text())
            probes.append(dict(model=name, images=r['epochs'][0]['samples'], nvml_process_peak_mib=r['nvml_process_peak_mib'],
                torch_peak_allocated_bytes=r['torch_peak_allocated'], torch_peak_reserved_bytes=r['torch_peak_reserved'],
                cpu_peak_rss_kib=r['cpu_peak_rss_kib'], result=str(probe_path)))
        for path in sorted((DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914').glob('*_data/training.json')):
            r = json.loads(path.read_text()); epoch = r['epochs'][0]
            data_only.append(dict(model=name, route=path.parent.name.removesuffix('_data'), samples=epoch['samples'],
                seconds=epoch['seconds'], images_per_second=epoch['images_per_second'], input_wait_seconds=epoch['input_wait_seconds'],
                nvml_process_peak_mib=r['nvml_process_peak_mib'], result=str(path)))
        for path in sorted((DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914').glob('*_profile/breakdown.json')):
            r = json.loads(path.read_text())
            traces.append(dict(model=name, workload='training', route=path.parent.name.removesuffix('_profile'), samples=r['captured_images'],
                window_seconds=r['window_seconds'], gpu_input_seconds=r['groups'].get('input', {}).get('union_seconds', 0),
                gpu_model_seconds=r['groups'].get('model', {}).get('union_seconds', 0), gpu_idle_seconds=r['gpu_idle_seconds'],
                overlap_seconds=r['input_model_overlap_seconds'], result=str(path)))
        for path in sorted((DATA/'runs'/directory/args.inference_run/'profiles').glob('*/breakdown.json')):
            if contended(args.inference_run, directory, f'inference_trace_{path.parent.name}'):
                continue
            for r in json.loads(path.read_text()).values():
                traces.append(dict(model=name, workload='inference', route=path.parent.name, samples=r['captured_images'],
                    window_seconds=r['window_seconds'], gpu_input_seconds=r['gpu'].get('input', {}).get('union_seconds', 0),
                    gpu_model_seconds=r['gpu'].get('model', {}).get('union_seconds', 0), gpu_idle_seconds=r['gpu_idle_seconds'],
                    overlap_seconds=r['model_input_overlap_seconds'], result=str(path)))
    write_csv(args.output_dir/'cnn_inference.csv', inference)
    write_csv(args.output_dir/'cnn_training.csv', training)
    write_csv(args.output_dir/'cnn_training_memory_probes.csv', probes)
    write_csv(args.output_dir/'cnn_inference_memory.csv', memory)
    write_csv(args.output_dir/'cnn_data_only.csv', data_only)
    write_csv(args.output_dir/'cnn_trace_breakdown.csv', traces)
    text = ['# CNN inference + training 实测结果', '',
        '主表使用 RTX 4090。仅汇总已写完的结果；PRO 6000 后端正确性短测不混入性能表。', '',
        f'本轮推理补测目录：`{args.inference_run}`。检测到其他 GPU 计算进程竞争的补测不进入主表；'
        '未取得合格补测时，推理 CSV 的 result 列仍指向先前结果，不算作本轮测量完成。', '',
        '## 1. Models and Experimental Setup', '',
        'ResNet-50 DCT-24/64，MobileNetV2 DCT-24/32。推理 FP32、TF32 off、batch 64、官方 checkpoint；'
        'RGB 使用对应官方 RGB 网络和权重。所有输入来自同一 imagenet_512 版本。', '',
        '训练从零初始化，配对 seed 11997733，BF16 autocast + Inductor，64×16=1024 有效 batch，'
        '完整训练集 1,281,167 张，保留尾批。沿用 SwinV2 的 300-epoch 配方及 10,000-update warmup，'
        '本轮停止于 2 epoch。RGB 复用 Transformer 的 PyTorch/DALI 增强，不作为 DCT 收敛控制。', '',
        '## 2. Inference End-to-End Results', '',
        '| 模型配置 | 路径 | Top-1 % | Top-5 % | CE | E2E s | 输入等待 s | 模型 s | 请求 GB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in inference:
        text.append('| '+ ' | '.join([r['model'], r['route'],number(r['top1']),number(r['top5']),number(r['ce']),
            number(r['e2e_seconds']),number(r['input_wait_seconds']),number(r['model_seconds']),
            number(r['requested_file_bytes']/1e9 if r['requested_file_bytes'] is not None else None)])+' |')
    text += ['', '以上每条推理路径均为 50,000 张。N 的 grid/projected × off/on 均按完整 shard 读取，'
        '随后按模型 batch 运行；所有完成的 N 预测与同配置 R 一致。GB 为十进制。请求字节表示文件读取范围，'
        '不是绕过 page cache 的 SSD 实读。各计时项有不同重叠边界，不相加解释 E2E。', '',
        '## 3. Training End-to-End Results', '',
        '| 模型 | 路径 | epoch | 图像数 | E2E s | images/s | 训练 CE | val Top-1 % | val Top-5 % | val CE | 两轮任务完成 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in training:
        text.append('| '+' | '.join([r['model'],r['route'],str(r['epoch']),str(r['samples']),number(r['e2e_seconds']),
            number(r['images_per_second']),number(r['training_ce']),number(r['top1']),number(r['top5']),number(r['ce']),
            '是' if r['full_run_complete'] else '否'])+' |')
    text += ['', '表中只列已完成 epoch；缺少的模型/路径仍未完成。epoch 1 包含首次编译及初始化，验证单独计时。'
        '准确率来自每轮 50K 验证，不把 8192 张短测当作收敛结果。', '',
        '## 4. Model-only and Data-only Ceilings', '',
        '已完成 BF16 驻留张量训练校准：MobileNetV2 RGB/DCT24/DCT32 分别为 3201.5/3315.4/3244.5 images/s；'
        'ResNet-50 RGB/DCT24/DCT64 为 1876.5/754.3/752.7 images/s。它们不含读取或在线增强。', '',
        'MobileNetV2 DCT-24 native data-only，8192 张短测：原配置 19.696 s（416 images/s）；'
        '原位输出、大 kernel 批量、grouped crop 调度复用后 8.807 s（930 images/s）。这是数据端短测，'
        '不等于完整训练加速倍数。', '',
        '## 5. Pipeline Breakdown', '',
        'B6：读取源 DCT → 保留 resize 全部源频率依赖 → fused projected grid → 原位 RandAugment/标准化/Mixup → model。'
        '完整浮点 192 通道 pool 和第二个输出 float pool 均未物化；仅 microbatch 使用 int16 scratch。', '',
        'A0：同源 JPEG 的多进程 DataLoader → 每图 DCT crop/resize → 同一 GPU 增强代码 → model。'
        '全局随机读取 premixed GALP 的诊断在 1024 张耗时 214 s、请求约 36.60 GB，已停止；'
        '该不合理的访问方式不作为标准 baseline。', '',
        'RGB：复用现有 PyTorch、DALI D2（planned ROI）和 D3（native crop/shuffle）训练适配器。'
        'D3 增强决策与 D2 不等价，单独呈现。worker 累计时间不加到 wall time。', '',
        '## 6. GPU Critical Path', '',
        'MobileNetV2 DCT-24 同为 16,384 张的 Nsight 训练窗口，首个 compile 已排除：', '',
        '| 配置 | 窗口 s | GPU input 活跃 s | model 活跃 s | GPU idle s | input/model 重叠 s |',
        '|---|---:|---:|---:|---:|---:|']
    capture_root=DATA/'runs/dctnet_mobilenet24/rtx4090_native_training_20260914'
    for label, file in [('优化前','nsys_breakdown_before_batching.json'),('优化后','nsys_breakdown_optimized.json')]:
        r=json.loads((capture_root/file).read_text())
        text.append('| '+' | '.join([label,number(r['window_seconds']),number(r['groups']['input']['union_seconds']),
            number(r['groups']['model']['union_seconds']),number(r['gpu_idle_seconds']),number(r['input_model_overlap_seconds'])])+' |')
    text += ['', '窗口内 GPU 活跃时间为区间并集，不能解释为 SM 利用率。优化前约 90.3 万次网格 kernel launch，'
        'CPU launch 和重复几何调度占据关键路径；扩大每次任务量并共享同几何的所有权计算后，'
        '窗口约快 1.99×。两次 trace 的预取边界不同，不将其 H2D 字节差解释为等量输入工作的减少。', '',
        '## 7. System Efficiency and Dataflow', '',
        'ResNet DCT 推理的模型部分约 63 s/50K，本身远高于 RGB ResNet 的约 23 s，因此 N 输入加速难以显著改变总时间。'
        'MobileNet DCT 的模型部分约 8 s，输入优化可以明显接近 RGB DALI。'
        '这些模型计算图不同，不能把网络计算差归因于 GALP。', '',
        '## 8. Resource Footprint / Memory', '',
        '以下为同合同 8192 张训练 + 1000 张验证短测的进程生命周期峰值，不能当作完整 epoch 峰值。'
        'NVML 以 100 ms 采样，包含 native 非 Torch 分配；主进程 RSS 不包含全部 workers。', '',
        '| 模型 | NVML 进程峰值 MiB | Torch allocated GiB | Torch reserved GiB | 主进程 RSS GiB |',
        '|---|---:|---:|---:|---:|']
    for r in probes:
        text.append('| '+' | '.join([r['model'],str(r['nvml_process_peak_mib']),number(r['torch_peak_allocated_bytes']/2**30),
            number(r['torch_peak_reserved_bytes']/2**30),number(r['cpu_peak_rss_kib']/2**20)])+' |')
    text += ['', 'MobileNetV2 DCT-24 原双 float pool 短测进程峰值 22550 MiB，原位整理后为 13142 MiB，降低约 41.7%。'
        '完整训练 memory 数据见已完成任务 JSON；推理 CSV 分别保留 Torch 和 native allocator 峰值，二者不直接相加。', '',
        '## 9. Pushdown and Materialization Ablations', '',
        '推理 off/on 实际改变文件请求范围、上传和解码；R→N 还包含离线预计算收益。训练 resize 混合频率，'
        '保留完整源依赖后在输出处选择通道，不把输出通道直接当作可删除的源列。', '',
        '## 10. Convergence: A0 versus B6', '',
        '仅对完成的配对 A0/B6 结果比较；未完成项不作不劣声明。即使两轮都完成，约 2504 次更新仍在 '
        '10000 次 warmup 内，只能说明早期行为，不能证明最终收敛非劣。', '',
        '## 11. Artifacts and Remaining Experiments', '',
        f'详细推理表：`{args.output_dir/"cnn_inference.csv"}`。', '',
        f'完整 epoch 表：`{args.output_dir/"cnn_training.csv"}`。', '',
        f'训练入口：`{HERE/"native_training_matrix.sh"}`、`{HERE/"cnn_training_baselines.sh"}`。', '',
        'B6 命令：`bash galp/experiments/dct_pushdown_inference/native_training_matrix.sh full rtx4090_native_training_optimized_20260914 B6 4 32768`。', '',
        '标准 baseline 命令：`bash galp/experiments/dct_pushdown_inference/cnn_training_baselines.sh full rtx4090_cnn_training_baselines_20260914`。', '',
        f'完成覆盖：推理 {len(inference)}/32；训练 epoch {len(training)}/28；完整推理 memory {len(memory)}/32；'
        f'data-only {len(data_only)}/14；训练 trace {sum(r["workload"] == "training" for r in traces)}/14；'
        f'推理 trace {sum(r["workload"] == "inference" for r in traces)}/32。缺项均为未完成，不作估算填补。', '',
        '### Full-run Memory', '',
        '| 模型 | 路径 | NVML MiB | 主进程 RSS GiB | 进程树 RSS 总和 GiB |',
        '|---|---|---:|---:|---:|']
    for r in memory:
        text.append(f'| {r["model"]} | {r["route"]} | {number(r["nvml_process_peak_mib"])} | {number(r["main_rss_bytes"]/2**30)} | {number(r["tree_rss_sum_bytes"]/2**30)} |')
    text += ['', 'NVML 为进程生命周期 100 ms 采样峰值；RSS 总和重复计算共享页，不是 PSS。', '',
        '| 训练模型 | 路径 | NVML MiB | Torch allocated GiB | 主进程 RSS GiB |', '|---|---|---:|---:|---:|']
    for r in training:
        if r['epoch'] == 2 and r['full_run_complete']:
            text.append(f'| {r["model"]} | {r["route"]} | {number(r["nvml_process_peak_mib"])} | {number(r["torch_peak_allocated_bytes"]/2**30)} | {number(r["cpu_peak_rss_kib"]/2**20)} |')
    text += ['', '### Data-only / Matched Trace', '',
        '| 模型 | 输入路径 | 样本数 | data-only s | images/s |', '|---|---|---:|---:|---:|']
    for r in data_only:
        text.append(f'| {r["model"]} | {r["route"]} | {r["samples"]} | {number(r["seconds"])} | {number(r["images_per_second"])} |')
    text += ['', 'Data-only 为 5 个 pool 的有界测量，包含首次数据初始化；不冒充整轮训练速度。', '',
        '| 模型 | workload | 路径 | 样本 | 窗口 s | GPU input s | model s | idle s | input/model overlap s |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in traces:
        text.append('| '+' | '.join([r['model'], r['workload'], r['route'], str(r['samples']),
            *[number(r[k]) for k in ('window_seconds','gpu_input_seconds','gpu_model_seconds','gpu_idle_seconds','overlap_seconds')]])+' |')
    text += ['', '### Paired Early Validation', '',
        '| 模型 | epoch | B6 − A0 Top-1 pp | B6 − A0 CE |', '|---|---:|---:|---:|']
    for name, _, _ in CONFIGS:
        for epoch in (1, 2):
            paired = {r['route']:r for r in training if r['model'] == name and r['epoch'] == epoch}
            if 'B6' in paired and 'A0 JPEG' in paired:
                a, b = paired['A0 JPEG'], paired['B6']
                text.append(f'| {name} | {epoch} | {number(b["top1"]-a["top1"])} | {number(b["ce"]-a["ce"])} |')
    text += ['', '这是单 seed、warmup 内的两轮比较，不构成最终收敛非劣证明。', '']
    (args.output_dir/'CNN_E2E_RESULTS.md').write_text('\n'.join(text))
    print(json.dumps(dict(inference_rows=len(inference), completed_epoch_rows=len(training), output=str(args.output_dir))))


if __name__ == '__main__':
    main()
