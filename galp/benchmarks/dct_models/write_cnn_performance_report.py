"""Build the presentation report from the completed CNN experiment artifacts."""
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from galp.benchmarks.dct_models.report_cnn_suite import CONFIGS, DATA, HERE, ROOT

RUN = 'rtx4090_cnn_memory_isolated_20260915'
TRAIN = 'rtx4090_cnn_training_baselines_20260914'
SUMMARY = DATA/'runs/dctnet_mobilenet24/rtx4090_cnn_complete_20260914'
DOCS = ROOT/'galp/benchmarks/system_dct_major/docs'
REPORT = DOCS/'CNN_SYSTEM_PERFORMANCE_REPORT_2026-09-15.md'
ASSETS = DOCS/'assets/cnn_20260915'
ARCHIVED_DATA = ROOT/'galp/data/compressed/backup/offline_model_input'


def read(path):
    return json.loads(Path(path).read_text())


def rows(name):
    with (SUMMARY/name).open() as f:
        return list(csv.DictReader(f))


def f(value, digits=2):
    return '未测' if value is None or value == '' else f'{float(value):,.{digits}f}'


def link(path, label=None):
    return f'[{label or Path(path).name}]({Path(path).resolve()})'


def table(headers, body):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in body]])


def model_name(name):
    return name.replace('MobileNetV2', 'MobileNetV2').replace(' DCT-', ' DCT')


def main():
    status = read(DATA/'runs/dctnet_mobilenet24'/RUN/'queue/status.json')
    assert status['state'] == 'complete' and len(status['jobs']) == 40
    occupancy = list((DATA/'runs/dctnet_mobilenet24'/RUN/'queue').glob('*.gpu_occupancy.json'))
    assert len(occupancy) == 40 and all(not read(p)['competing_pids'] for p in occupancy)
    inference, training, memory, traces, data = [rows(name) for name in
        ('cnn_inference.csv', 'cnn_training.csv', 'cnn_inference_memory.csv', 'cnn_trace_breakdown.csv', 'cnn_data_only.csv')]
    assert list(map(len, (inference, training, memory, traces, data))) == [32, 28, 32, 46, 14]
    assert all(RUN in r['result'] and int(r['samples']) == 50000 for r in inference)
    assert all(r['full_run_complete'] == 'True' and r['samples'] == r['unique_samples'] == '1281167'
               and r['validation_samples'] == '50000' for r in training)
    assert all(float(r['agreement_with_R']) == 1 for r in inference if r['route'].startswith('N '))
    idx = {(r['model'], r['route']):r for r in inference}
    ti = {(r['model'], r['route'], int(r['epoch'])):r for r in training}
    mi = {(r['model'], r['route']):r for r in memory}
    models, profiles, native_epochs, sums = {}, {}, {}, {}
    for name, directory, _ in CONFIGS:
        models[name] = read(DATA/'runs'/directory/RUN/'models.json')
        native_epochs[name] = read(DATA/'runs'/directory/'rtx4090_native_training_optimized_20260914/B6_full_m4_inplace/epoch_1.json')
        values = native_epochs[name]['pools']
        sums[name] = {key:sum(p['native'].get(key, 0) for p in values) for key in (
            'compressed_payload_bytes_read', 'selected_compressed_payload_bytes', 'full_compressed_payload_bytes',
            'workset_upload_dma_bytes', 'selected_vector_count', 'full_vector_count', 'fixed_transform_source_block_count',
            'duplicate_physical_read_count', 'planning_ms', 'active_output_planning_ms', 'decode_ms', 'fixed_transform_ms')}
        profiles[name] = read(idx[name, 'N projected on']['result'])['model_profile']
    warm_data = []
    for r in data:
        result = read(r['result']); pools = result['epochs'][0]['pools']
        seconds = pools[-1]['completed_epoch_seconds']-pools[0]['completed_epoch_seconds']
        warm_data.append(dict(model=r['model'], route=r['route'], cold=float(r['images_per_second']),
            warm=16384/seconds, seconds=seconds, result=r['result']))
    model_only = {}
    for name, directory, _ in CONFIGS:
        for domain in (('rgb', 'dct') if directory.endswith('24') else ('dct',)):
            model_only[name, domain] = read(DATA/'runs'/directory/'rtx4090_training_steady_20260914/model_only'/f'{domain}_model_only.json')
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_figures(inference, training, memory, traces)
    for name in ('inference_e2e', 'inference_payload', 'early_validation'):
        for extension in ('png', 'pdf'):
            shutil.copyfile(SUMMARY/f'figures/{name}.{extension}', ASSETS/f'{name}.{extension}')
    text = []
    def add(value):
        text.append(value.strip()+'\n')
    def image(name, caption):
        add(f'![{caption}](assets/cnn_20260915/{name}.png)\n\n{caption}。同目录提供可导出的 PDF。')

    add('''# GALP on CNN：端到端训练、推理与 I/O 性能报告

日期：2026-09-15
硬件：NVIDIA GeForce RTX 4090（24 GiB）；双路 Xeon Gold 5318Y，96 个逻辑 CPU
软件：PyTorch 2.11.0+cu128，torchvision 0.26.0+cu128，CUDA 12.8
数据集：ImageNet-1K，训练集 1,281,167 张，验证集 50,000 张
模型：MobileNetV2 DCT24/32、ResNet-50 DCT24/64，以及对应 RGB 模型

## 摘要

GALP 的 block-major 数据路径已在四个官方 CNN DCT 配置上完成推理和两轮从零训练。
推理采用同一配置的同一官方 DCT checkpoint 比较参考路径 R、新目标表示 N 与旧数据适配 O；
训练采用相同初始化、优化器和 DCT 增强数学，比较标准逐图 crop/global shuffle 的 A0 与
grouped crop/delayed shuffle 的 B6。两条训练路径有意保留各自的 crop/order 策略，不能称为逐样本增强完全等价。

本次推理复测的 40 个任务全部通过，包括 32 条完整 50K 路径、4 组 model-only 和 32 个 Nsight 窗口。
训练完成 14 条路径 × 2 epoch，以及 14 组 data-only 和 14 个训练 Nsight 窗口。
推理复测开始前等待 4090 无其他计算进程，运行中约每秒采样，未记录到其他计算进程竞争。
上轮 OOM 或竞争时段的推理补测不进入本报告主表；训练及其诊断没有同等粒度的独占监控，主机 page cache 也未受控。

主要结论是：**MobileNetV2 获得显著输入路径收益，ResNet-50 更接近模型计算上限；收益不能仅由压缩率解释。**''')
    summary_rows = []
    for name, _, _ in CONFIGS:
        a, b = ti[name, 'A0 JPEG', 2], ti[name, 'B6', 2]
        r, n = idx[name, 'R'], idx[name, 'N projected on']
        summary_rows.append([model_name(name), f(b['images_per_second']), f(float(b['images_per_second'])/float(a['images_per_second']), 3)+'×',
            f(float(b['top1'])-float(a['top1']), 3)+' pp', f(n['images_per_second']), f(float(r['e2e_seconds'])/float(n['e2e_seconds']), 3)+'×', '50K 完全一致'])
    add(table(['配置','B6 训练 img/s','B6/A0 训练加速','E2 Top-1：B6−A0','N 推理 img/s','R/N 推理加速','N 与 R 预测'], summary_rows))
    add('''N 默认指 **projected + coefficient I/O pushdown on**。MobileNetV2 DCT24 的 N 为 9.97 s/50K，
RGB DALI 为 10.26 s；DCT32 的 N 为 11.69 s，RGB DALI 为 10.23 s。两者属于不同输入域和 checkpoint，
这里只比较完整系统，不把比值当作 reader 或压缩器的纯收益。

两轮训练仍处于 10,000-update warmup 内。本报告支持“当前早期训练未观察到明显发散”，
**不支持最终收敛不劣、统计等价或论文最终精度复现**。''')
    add('''## 1. 实验问题与证据结构

1. GALP 能否在保持官方 DCT 输入与预测的同时，消除在线目标 DCT 构造成本？
2. 固定频率的读取下推与直接生成所需布局，分别降低了多少 I/O、显存和端到端时间？
3. B6 的 grouped crop 与 delayed shuffle 能否驱动 CNN 训练，早期精度与标准 A0 有何差异？
4. 不同 CNN 的瓶颈是 CPU 输入、GPU DCT 变换、内存流量，还是模型计算？
''')
    add(table(['证据','范围','可支持的解释','限制'],[
        ['训练 E2E','14 路 × 完整 E1/E2','同样本数训练吞吐；E2 为主要 warm observation','单次 epoch，非长期性能分布'],
        ['收敛对照','4 个 DCT 配置 × A0/B6 × 2 epoch','共同数据与初始化下的早期行为','crop/order 不同；单 seed；仍在 warmup'],
        ['推理 E2E','4 配置 × 8 路 × 50K','相同 DCT checkpoint 下的 R/N/O；RGB 系统参照','每条主表只有一次完整运行，不报告 repeat CV 或 p95'],
        ['训练/推理 Nsight','16,384 / 4,096 张窗口','GPU 活动、idle、overlap 和 copy bytes','profiler 开启；不能替代未插桩的完整 E2E'],
        ['进程内存','全部完整训练与推理','NVML 含 native/DALI 的进程峰值','采样峰值；RSS 总和重复计算共享页'],
    ]))
    add('''## 2. CNN 模型与数据路径

### 2.1 模型输入与计算量

模型是 **MobileNetV2，不是 MobileNetV3**。DCT 通道数是 Y、Cb、Cr 合计；以下仅覆盖本轮固定配置，
不是对可用架构或所有可能通道预算的穷举。MACs 按真实前向的 Conv/Linear 输出形状累计，不含其他算子。''')
    body=[]
    for name, _, _ in CONFIGS:
        p=profiles[name]; m=models[name]['DCT']
        body.append([model_name(name),'×'.join(map(str,p['shape'])),'/'.join(str(len(v)) for v in p['indices']),
            f(m['parameters']/1e6,3),f(m['conv_linear_macs_per_image']/1e9,3),f(m['resident_batch64_gpu_ms'],3)])
    for name in ('MobileNetV2 DCT-24','ResNet-50 DCT-24'):
        m=models[name]['RGB'];body.append([name.split(' DCT')[0]+' RGB','3×224×224','RGB',f(m['parameters']/1e6,3),f(m['conv_linear_macs_per_image']/1e9,3),f(m['resident_batch64_gpu_ms'],3)])
    add(table(['模型','输入 C×H×W','Y/Cb/Cr 通道','参数 M','Conv/Linear GMAC/image','驻留输入 FP32 ms/batch64'],body))
    add('''ResNet DCT 的官方计算图在更大空间网格上执行后续层，约 13.56 GMAC/image，RGB 约 4.09 GMAC。
因此同属 ResNet-50 不代表同一计算工作量；固定通道选择不会自动降低后续主干 MAC。

RGB 使用 DCTNet 仓库引用的官方 RGB 权重。实验 wrapper 为仓库中缺少 forward 的基类补上标准 RGB forward，
并恢复 RGB stride；严格加载全部权重。DCT 模型保持官方结构。它不是把 DCT checkpoint 接到 RGB 网络。

### 2.2 推理 R、N、O 与 RGB

```mermaid
flowchart LR
    JPG[同源 512 JPEG] --> R[官方像素 resize/crop/upscale\n重新编码 Q100 / DCT 提取]
    R --> INPUT[固定频率 / 分量顺序 / mean/std]
    INPUT --> DCT[CNN DCT checkpoint]
    JPG --> OFFLINE[离线生成完整目标 DCT]
    OFFLINE --> N[GALP 目标数据\n整 shard 读取与选择性解码]
    N --> PROJECT[直接生成指定 NCHW\n融合反量化与标准化]
    PROJECT --> DCT
    OLD[旧 GALP 源 DCT] --> O[反量化 / 中心 crop\nDCT 域上采样]
    O --> INPUT
    JPG --> RGB[PyTorch 或 DALI\nRGB decode / resize / crop]
    RGB --> RGBMODEL[CNN RGB checkpoint]
```

R 复用官方 OpenCV BGR loader 和参考算子：ResNet Resize 512、crop 448；MobileNet Resize 1024、crop 896；
均使用 bilinear upscale、JPEG quality 100、4:2:0 重新编码。Y 来自原 crop 分支，Cb/Cr 来自 2× 分支，
最终三个分量分别都是 56×56 或 112×112 blocks。

N 保存三个分量各完整 64 个自然频率的量化 int16 及正确量化表，位于模型标准化之前。
本数据的 Q100 量化表全为 1；仍明确只反量化一次，GALP zigzag 列索引显式映射至自然序 `u×8+v`。
`grid` 先形成完整 192 通道等价浮点网格再整理；`projected` 直接写选定通道的标准化 NCHW。
每个 shard 激活一次，49 个 shard 后按 batch64 推理，保留原 ordinal，最多一个当前 shard 和一个预取 shard。

O 解码旧 512/4:2:0 数据：Y 64² 中心裁至 56²，Cb/Cr 32² 裁至 28²，再在 DCT 域上采样至 56²；
MobileNet 继续把三个分量从 56² 放大至 112²。此适配器不做 round/clamp，不运行 RGB-no-more 224 profile 或 patch 换基。
它与官方像素放大再重新编码不同，R 与 O 不要求逐值相等。

### 2.3 训练 A0 与 B6

A0 从同源 JPEG 提取已反量化 DCT，以 16 个 DataLoader worker 执行逐图 crop/resize 和 global shuffle，
之后使用与 B6 共享的 GPU RandAugment/标准化/Mixup 实现。B6 从现有 premixed block-major 源 DCT 出发，
使用 G=1024 的 grouped crop、M=4 的 4096-image closed pool、delayed shuffle 和双 context lookahead。
训练 resize 需要源频率混合，保留全部源依赖，再输出固定模型需要的通道；不能将输出频率索引直接用于删除源频率。

通用优化包括：同几何调度复用、一次处理整组、增大每次 kernel 工作量、投影输出、原位增强与标准化。
输出网格、通道索引和 mean/std 来自 profile，不在 native runtime 中按 CNN 类名分支。
训练只为增强依赖保留所需的额外转置频率，使用 microbatch int16 scratch，避免完整 192 通道 float pool 和第二份 float 输出 pool。''')
    add('''## 3. 实验设置与公平性

### 3.1 训练设置''')
    add(table(['属性','设置'],[
        ['初始化','随机初始化，seed 11997733；构造官方模型后重置所有参数层；无 fine-tuning'],
        ['训练长度','每条路径完整 2 epoch；每轮 1,281,167 个不同样本，尾批保留；每轮后 50K validation'],
        ['Batch / optimizer','64 × accumulation16 = 1024；每轮 1252 次更新；AdamW＋独立 weight decay'],
        ['Schedule / precision','沿用 SwinV2 300-epoch 配方、10,000-update warmup；BF16 autocast＋Inductor；TF32 off'],
        ['计时','E1 包含首次编译；E2 为主要 warm observation；每轮验证另计'],
        ['数值与增强','DCT resize 后 round 到整数并限制 int16；RandAugment clamp [-1024,1016]；标准化后 Mixup'],
        ['CPU / native','A0 与 RGB PyTorch 16 workers，worker 内 torch1线程；DALI4线程；模型主进程 torch8线程'],
        ['B6','M4 双 context，transform blocks/launch 32768；不重新压缩训练集'],
        ['Audit','首100次更新同步检查；后续递延有限性检查；epoch 末检查参数和样本覆盖'],
    ]))
    add('''A0/B6 共享模型、seed、源图像、频率与标准化、增强数学、optimizer/scheduler，但 crop/order 有意不同。
因此二者的准确率差衡量完整训练配置，不是仅替换 reader 的等语义 A/B。
RGB PyTorch/DALI D2 复用 Transformer 的计划 crop/flip；D3 使用 DALI-native crop/order。
RGB 训练是 224 RGB、mean/std 0.5、hard labels，不使用 DCT RandAugment/Mixup，不作为 DCT 收敛控制。

### 3.2 推理与测量边界

推理 batch64，FP32、TF32 off，每路完整50K，逐配置的 R/N/O 严格加载同一官方 DCT checkpoint；RGB 使用对应 RGB checkpoint。
R/O/RGB PyTorch 使用64个输入 workers；DALI的 workers 参数为其内部线程数，不能视为64个同类进程。
N 使用 native shard reader、4个 rowgroup prefetch workers、64-rowgroup decode batching 和512 MiB workset。
推理主表为单次完整运行，模型 warmup 排除，输入文件到 logits 的 E2E 包括 reader/loader 在线阶段；
进程启动、导入和模型构造另在 process wall/memory 口径中体现。

推理 Nsight 在16,384张 warmup 后捕获4,096张；训练在首个4096-image pool 后捕获16,384张，
这些训练窗口仍包含首100次更新的严格 audit。GPU 活跃时间使用 kernel/copy 区间并集，idle 不是 SM 利用率。
各内部阶段存在重叠，不能相加或从 wall 中相减得到模型常数。

page cache 未清空；其结果是当前机器与缓存状态下的系统观察。独立队列只管理自己的任务，
运行前检查和约1秒的进程采样不能构成硬件独占锁，也不能排除亚秒竞争或其他GPU任务造成的共享CPU/I/O影响。''')
    add('## 4. 训练端到端性能\n\n### 4.1 完整 Epoch 2')
    e2=[r for r in training if r['epoch']=='2']
    add(table(['模型','Pipeline','E2 wall s','img/s','E1 img/s','输入等待 s','模型 stream s'],[
        [model_name(r['model']),r['route'],f(r['e2e_seconds']),f(r['images_per_second']),f(ti[r['model'],r['route'],1]['images_per_second']),
         f(r['input_wait_seconds']),f(r['model_stream_seconds'])] for r in e2]))
    image('training_e2e','图1：完整第二轮训练吞吐，DCT 同域对照与 RGB 系统参照分开呈现')
    add('''MobileNetV2 的 B6 输入等待约占 E2 的64%；相对于 A0，吞吐提高46%–67%，但仍低于 RGB DALI。
ResNet B6 的输入等待只有约1.5秒/epoch，模型 stream 时间接近整个epoch，因此更快 reader 只能带来有限 E2E 改善。

`model stream` 是 CUDA event 覆盖的流时间，不是 kernel 活跃时间；它包含主线程发射、audit 和流上其他工作的影响。
A0 的输入等待与模型流时间可以重叠，例如 ResNet DCT24 两项相加超过 wall，不能据此判断计时错误或相加分配百分比。

### 4.2 Model-only 与 data-only

训练 model-only 是固定64张 GPU 驻留输入的独立校准：先105次更新 warmup，再120次更新、122,880次样本呈现，
包含 forward/backward/optimizer/audit。**该校准从官方预训练权重开始，更新后丢弃；完整两轮训练则从零开始。**
它只用于计算工作诊断，不是同一训练状态的严格上限。

Data-only 每路20,480张。下表同时列完整短测和剔除首pool后的4个pool；后者避免把 DataLoader/DALI 启动成本误作稳态瓶颈。
它仍是短窗口，不是完整epoch，也不能用不同窗口的 E2E 减去 data-only。''')
    add(table(['模型','路径','data-only 全5 pool img/s','后4 pool img/s','训练 model-only img/s'],[
        [model_name(r['model']).split(' DCT')[0]+' RGB' if r['route'].startswith('rgb_') else model_name(r['model']),r['route'],f(r['cold']),f(r['warm']),
         f(model_only[r['model'],'rgb' if r['route'].startswith('rgb_') else 'dct']['images_per_second'])] for r in warm_data]))
    add('''MobileNet B6 data-only 后4 pool 约958–966 img/s，与完整 E2 的950–957 img/s 接近，
而独立模型校准约3245–3315 img/s，说明数据端仍是主要优化方向。
ResNet B6 data-only 约3257–3389 img/s，模型校准约753–754 img/s，完整训练约709–716 img/s，
进一步压缩输入时间的绝对空间较小。以上不是全局最优配置证明。

### 4.3 训练 Nsight critical path''')
    train_traces=[r for r in traces if r['workload']=='training']
    add(trace_table(train_traces))
    image('training_gpu_breakdown','图2：16,384张训练窗口的GPU活动区间分解；重叠单独列出，灰色为无GPU活动')
    add('''B6 MobileNet 的 GPU input 约7.0–7.1秒，model 活跃约3.7–3.8秒，仍有约5.6秒 idle。
ResNet B6 的 model 活跃约22.3秒，input/model 重叠约3.8–4.1秒，idle 约1.1–1.2秒。
模型越重，越容易隐藏下一pool的准备；这与完整epoch输入等待的差异一致。

DALI 表中的 input 只统计其 GPU kernel 活动；nvJPEG、H2D 和其他 copy 另有区间，不能把很小的 input 值解释为零预处理成本。
图中保留其他GPU活动，避免把未归入 input/model 的工作误作 idle。''')
    add('### 4.4 Native pool 准备与共享调度')
    add(table(['配置','计划累计 s','active-output 调度 s','DCT transform累计 s','pool准备累计 s','暴露 activation wait s'],[
        [model_name(n),f(sums[n]['planning_ms']/1000),f(sums[n]['active_output_planning_ms']/1000),f(sums[n]['fixed_transform_ms']/1000),
         f((native_epochs[n]['native_prefetch']['prepare_plan_ms']+native_epochs[n]['native_prefetch']['prepare_materialize_ms'])/1000),
         f(native_epochs[n]['native_prefetch']['activation_wait_ms']/1000)] for n,_,_ in CONFIGS]))
    add('''内部累计计时可能嵌套，表中不得求和。四个配置均激活313个pool并使用最多两个context。
MobileNet112网格的调度和DCT变换成本明显高于ResNet56网格；当前共享几何复用已消除大量重复工作，仍没有消除真实的上采样计算。
此前 MobileNet DCT24 的同16,384张 trace 从31.31秒降至15.71秒，主要来自扩大 kernel 工作批量和调度复用；
这是历史优化前后诊断，不混入本次各路径主表。''')
    add('''## 5. 存储、I/O 下推与数据移动

### 5.1 离线目标数据与存储成本

训练复用现有106.946 GB premixed FLS源布局；源训练JPEG inventory约50.601 GB，
FLS约为2.114倍。本轮没有重新压缩训练集。
推理N母数据分别保存三个分量的完整64频率，50K各49个shard；量化系数和量化表回读检查通过，样本映射一致。
下面是实际生成任务记录，生成 worker 时间和编码时间为跨任务累计，不与wall相加。''')
    generation=[]
    for label, d in [('ResNet：3×64×56×56','imagenet512_val_resnet56_block_major'),('MobileNet：3×64×112×112','imagenet512_val_mobilenet112_block_major')]:
        g=read(ARCHIVED_DATA/d/'generation.json'); parts=[read(p) for p in (ARCHIVED_DATA/d).glob('shard_*.json')]
        generation.append([label,g['samples'],g['completed_shards'],f(g['invocation_wall_seconds']),f(g['images_per_second']),
            f"{g['workers']}/{g['shard_workers']}/{g['encoding_threads']}",f(sum(p['generation_worker_seconds'] for p in parts)),
            f(sum(p['encoding_seconds'] for p in parts)),f(g['total_disk_bytes']/1e9,3)])
    add(table(['目标','样本','shard','生成wall s','img/s','生成worker/shard并行/编码线程','生成worker累计 s','编码累计 s','生成记录磁盘 GB'],generation))
    add('''源validation JPEG inventory为2.052 GB。N专用目标表示的存储代价很大，尤其是112网格；
生成记录的磁盘大小不等于后续新增access sidecar后的所有文件总和。完整频率母数据便于固定模型复用，
但不是所有模型通用输入，不能把其存储代价与GALP无损编码收益混在一起。
目前只有有限并发校准与实际任务记录，没有证明编码配置达到全机最优；大网格的生成/编码仍是主要离线成本。

### 5.2 B6 训练 crop pushdown

四个CNN配置的第二轮源读取计数完全一致，说明源数据选择不依赖backbone：''')
    s=sums[CONFIGS[0][0]]
    add(table(['计数层次','E2 实测','解释'],[
        ['完整源blocks',f(1281167*6144,0),'512/4:2:0，每图6144个源block'],
        ['进入transform的源blocks',f(s['fixed_transform_source_block_count'],0),f(s['fixed_transform_source_block_count']/(1281167*6144)*100,3)+'% retention'],
        ['选择/触及rowgroup完整vectors',f(s['selected_vector_count'],0)+' / '+f(s['full_vector_count'],0),f(s['selected_vector_count']/s['full_vector_count']*100,3)+'% retention'],
        ['触及rowgroup完整payload',f(s['full_compressed_payload_bytes']/1e9,3)+' GB','已排除未触及rowgroup'],
        ['精确选择payload',f(s['selected_compressed_payload_bytes']/1e9,3)+' GB','范围选择下界'],
        ['实际payload范围读取',f(s['compressed_payload_bytes_read']/1e9,3)+' GB','约1.0448×相对精确选择；非SSD设备流量'],
        ['Workset DMA',f(s['workset_upload_dma_bytes']/1e9,3)+' GB','包含压缩数据与执行metadata'],
        ['重复物理读取计数',f(s['duplicate_physical_read_count'],0),'native报告无重复读取'],
    ]))
    add('''B6相对触及rowgroup的完整payload少读22.40%，源block工作减少46.80%。
读取下推、输出通道投影和输出网格大小是不同层次；本轮训练不能按最终通道数直接删掉resize所需的源频率。

### 5.3 推理固定频率 I/O 下推与物化消融''')
    body=[]
    for name,_,_ in CONFIGS:
        off,on=idx[name,'N projected off'],idx[name,'N projected on']
        body.append([model_name(name),f(float(off['requested_file_bytes'])/1e9,3),f(float(on['requested_file_bytes'])/1e9,3),
            f((1-float(on['requested_file_bytes'])/float(off['requested_file_bytes']))*100)+'%',
            f(float(off['h2d_payload_bytes'])/1e9,3),f(float(on['h2d_payload_bytes'])/1e9,3),
            f(float(off['e2e_seconds'])/float(on['e2e_seconds']),3)+'×'])
    add(table(['配置','off请求 GB','on请求 GB','请求减少','off DMA GB','on DMA GB','projected off/on E2E加速'],body))
    image('inference_payload','图3：同一完整目标数据上的固定频率I/O下推，统计实际请求范围')
    add('''下推已到文件payload范围层，不只是选择性解码或少上传。当前列布局按三个分量所需频率的并集选择，
不能把模型合计24/32/64通道直接理解为只读24/32/64个独立物理分量列。
on/off在同一母数据上产生相同最终输入和50K预测；ResNet中少读字节没有等比例转换为吞吐，因为其模型已占主要时间。

`requested bytes`是reader发出的文件范围；没有清空page cache，不等于NVMe流量。
本轮进程采样另记录`read_bytes/read_chars`，但边界覆盖进程启动和IPC，且短命worker可能漏采，
不将它们外推为精确SSD收益。R→N包含离线预计算；只有同N数据的off/on才用于隔离固定列下推收益。''')
    add('''## 6. 训练收敛

### 6.1 四个 DCT 配置的两轮配对结果

每个数字来自该轮后完整50K验证；验证统一使用对应N目标表示。训练DCT域crop/resize与官方参考像素变换并不等价，
但A0和B6共享验证路径，因此比较口径一致。''')
    body=[]
    for name,_,_ in CONFIGS:
        for epoch in (1,2):
            a,b=ti[name,'A0 JPEG',epoch],ti[name,'B6',epoch]
            body.append([model_name(name),epoch,f(a['top1'],3),f(b['top1'],3),f(float(b['top1'])-float(a['top1']),3),
                f(a['top5'],3),f(b['top5'],3),f(a['ce'],4),f(b['ce'],4)])
    add(table(['模型','Epoch','A0 Top-1%','B6 Top-1%','Δ pp','A0 Top-5%','B6 Top-5%','A0 CE','B6 CE'],body))
    image('early_validation','图4：单seed前两轮50K验证精度，仍处于学习率warmup阶段')
    add('''第二轮B6−A0 Top-1差分别为+0.468、+0.324、−0.170和+2.496 pp。
ResNet DCT24的Top-1略低而CE略好；ResNet DCT64的早期优势也不能直接推断为最终优势。
完成两轮只能证明训练可运行、数值稳定和当前早期表现，尚不能满足“最终收敛不劣”的强要求。

### 6.2 RGB 训练参照

RGB路径的第二轮Top-1、Top-5、CE列在下表，只用于检查其训练进展，不与DCT增强recipe混作收敛控制。''')
    add(table(['模型','RGB路径','Top-1%','Top-5%','CE'],[[r['model'],r['route'],f(r['top1'],3),f(r['top5'],3),f(r['ce'],4)] for r in e2 if r['route'].startswith('rgb_')]))
    add('''## 7. 推理端到端性能

### 7.1 完整50K、全部baseline

以下均为本次无已检测GPU竞争的重测；每个配置中的RGB行使用相同RGB checkpoint重复运行，
不是额外的RGB通道预算。吞吐、精度与计时按同一行的实际结果对应。''')
    for name,_,_ in CONFIGS:
        add('#### '+model_name(name))
        add(table(['路径','wall s','img/s','Top-1%','Top-5%','CE','输入等待 s','模型 s'],[
            [r['route'],f(r['e2e_seconds']),f(r['images_per_second']),f(r['top1'],3),f(r['top5'],3),f(r['ce'],4),f(r['input_wait_seconds']),f(r['model_seconds'])]
            for r in inference if r['model']==name]))
    image('inference_e2e','图5：完整50K推理吞吐，包含RGB、官方参考、旧适配和四种N执行方式')
    add('''MobileNet R的输入等待约58秒，而N projected on降至1.8–3.3秒；这是其大幅E2E收益的直接表现。
ResNet R已经主要被约63秒模型时间限制，N projected on为63.6–63.9秒，因此相对R只提高约9%–14%。
RGB ResNet的模型时间约23秒，不能要求GALP通过输入优化补偿DCT模型约3.3倍的Conv/Linear计算量。

O不重写旧数据，但当前CPU适配链路耗时明显：MobileNet O的输入等待162–203秒，ResNet为65–82秒。
其Top-1相对R分别为−1.788、−0.908、+0.332、−1.100 pp。O可作为兼容已有存储的路径，
当前实现不适合作为追求吞吐的默认路径；也不支持“所有模型精度都不降”的结论。

### 7.2 GPU critical path 与完整网格物化''')
    image('inference_gpu_breakdown','图6：4,096张推理窗口的GPU活动分解；选择代表性路径解释输入与模型瓶颈')
    add('''MobileNet DCT24从grid on切换projected on，GPU input活跃从1.345秒降至0.065秒/4096张，
model活跃维持约0.57秒，窗口从2.382秒降至0.715秒。DCT32对应input从1.344秒降至0.072秒。
优化消除了完整192通道浮点网格和后续gather/permute/normalization的主要流量，直接生成模型所需输出布局。

ResNet projected on的model活跃约5.11秒，窗口约5.19–5.22秒，GPU idle仅约0.065–0.069秒。
此时进一步下推仍节省I/O与中间存储，吞吐提升自然较小。这不是GALP没有做下推，也不是已证明全局最优。

R/O的GPU input列为0，表示输入主要在CPU执行；不能理解为它们没有预处理成本。
固定模型的model-only应当相近，但在线model事件可受并发输入kernel和流竞争影响，grid与projected间的model计时差不能当作网络MAC变化。

### 7.3 全路径推理 Nsight 数值''')
    add(trace_table([r for r in traces if r['workload']=='inference']))
    add('''### 7.4 进程内存与显存

推理与训练均采用NVML进程显存约100 ms采样；该值包含Torch之外的native/DALI分配。
Torch allocated/reserved是其他口径，不能直接相加。下列推理RSS为完整进程生命周期峰值，
包括启动与预取；进程树RSS总和重复计算共享页，不是独占物理内存或PSS。''')
    add(table(['模型','路径','NVML GiB','主进程RSS GiB','进程树RSS总和 GiB'],[
        [model_name(r['model']),r['route'],f(float(r['nvml_process_peak_mib'])/1024,3),f(float(r['main_rss_bytes'])/2**30,3),f(float(r['tree_rss_sum_bytes'])/2**30,3)] for r in memory]))
    image('inference_memory','图7：推理NVML进程显存峰值；projected输出避免完整192通道浮点网格')
    add('''MobileNet grid消耗约21–22 GiB，projected on约3.56/4.43 GiB，下降约80%–83%。
ResNet grid约7.4–8.3 GiB，projected on约3.10/4.22 GiB。
这说明只看Torch allocator会漏掉大量native内存，也说明完整网格物化确实是重要成本。

训练的完整两轮进程峰值如下，包含各轮验证，与早先8192张probe的峰值不是同一口径。''')
    body=[]
    for r in e2:
        result=read(Path(r['result']).parent/'training.json')
        body.append([model_name(r['model']),r['route'],f(result['nvml_process_peak_mib']/1024,3),f(result['torch_peak_allocated']/2**30,3),
            f(result['torch_peak_reserved']/2**30,3),f(result['cpu_peak_rss_kib']/2**20,3)])
    add(table(['训练模型','路径','NVML GiB','Torch allocated GiB','Torch reserved GiB','主进程RSS GiB'],body))
    add('''B6完整训练NVML峰值约12.5–17.4 GiB，明显高于A0的约4.7–9.9 GiB，是维持大pool、预取和GPU变换的成本。
原位投影已经减少额外float pool，但它不意味着B6内存小于逐图CPU baseline。

### 7.5 正确性

16条N执行路径各自50K预测与同配置R完全一致，Top-1/Top-5一致；CE只存在约10⁻⁹量级归约差异。
原始目标数据生成记录确认量化系数与量化表无损回读，抽样输入/logits检查另保留原始验证产物。
这支持“本合同下GALP保存和读取没有改变参考预测”，不外推为任意JPEG/DCT接口的完全等价。
O使用不同DCT域几何，不要求输入或预测与R一致；RGB PyTorch/DALI也有decoder/resize实现差异。''')
    add('''## 8. 与 ViT-Ti、SwinV2-T 的关系

本报告沿用SwinV2报告的证据结构，补齐CNN的训练、推理、data-only、Nsight和进程内存。
共同的可迁移边界是block-major存储、crop-before-materialize、压缩workset上传、native调度和profile驱动的输出投影，
不依赖ViT token拓扑、Swin窗口结构或CNN类名。

四个CNN配置的B6第二轮均读取57.865 GB payload、源block retention 53.202%，与SwinV2的同源布局计数相符。
这体现数据选择规则的可迁移性，不意味着性能也必须相同。CNN采用DCTNet的56/112分量网格，
SwinV2训练采用28/14分量网格；数据输出大小与模型计算窗口均不同。

SwinV2报告中的严格DCT pair共享crop/order；本CNN A0/B6有意比较global/per-image与grouped/delayed策略。
SwinV2的15-epoch配对收敛证据也强于本CNN的2-epoch warmup证据。因此不把跨报告绝对吞吐或非劣结论直接移植。

## 9. 可展示的结论与限制

### 9.1 证据支持的结论

- 四个官方DCT CNN配置均可由GALP驱动；完整训练和推理没有样本丢失或重复映射。
- N保持同checkpoint的50K预测，与R相比MobileNet获6.60–7.54×推理加速，ResNet获1.09–1.14×。
- 固定频率下推已减少实际文件请求范围约19.8%–58.0%；这与预计算收益可分别观察。
- projected布局显著减少GPU输入kernel工作和进程显存；MobileNet约80%–83%的grid显存峰值下降是直接证据。
- B6的完整E2训练相对A0提高1.02–1.67×；MobileNet仍受数据供给约束，ResNet主要受模型约束。

### 9.2 尚不能支持的结论

- 两轮warmup不能证明最终收敛不劣；单seed不能证明统计等价。
- 不同RGB/DCT计算图和checkpoint的比值不是纯reader/codec加速比。
- 有限并发校准、单次50K及单epoch观察不能证明全局最优、稳定p95或跨机器泛化。
- reader范围字节不等于SSD流量；NVML采样峰值不是理论最大峰值；RSS总和不是PSS。
- kernel/copy区间的idle不是SM利用率；累计阶段计时不相加。
- 推理复测未观察到其他GPU计算进程，仍不能排除亚秒竞争或共享CPU/I/O影响；训练没有同等独占审计。

### 9.3 下一步优先级

若目标是部署吞吐，优先采用N projected＋固定接口I/O下推；O保留为不重写旧数据的兼容方案。
若目标是CNN训练系统优化，优先处理MobileNet112网格的CPU调度和GPU DCT变换供给，
而不是继续压缩已被ResNet计算隐藏的几毫秒输入等待。
若目标是正式声明收敛不劣，应先扩大A0/B6共同训练前缀至warmup之后；本次报告不以两轮结果替代这一证据。

## 10. 原始证据与可复现性

### 10.1 实际路径与运行入口''')
    paths=[('报告对应推理运行',DATA/'runs/dctnet_mobilenet24'/RUN),('完整CSV与原始汇总',SUMMARY),
        ('同源图像',DATA/'imagenet_512'),('训练/验证manifest',DATA/'training_manifests_official_v3'),
        ('56网格目标数据',ARCHIVED_DATA/'imagenet512_val_resnet56_block_major'),('112网格目标数据',ARCHIVED_DATA/'imagenet512_val_mobilenet112_block_major'),
        ('源训练premix',Path('/home/tangyuxin/gfastlanes/FastLanes/galp/data/compressed/imagenet512_train_block_major_premixed')),
        ('官方checkpoint',DATA.parent/'e2e_v2/checkpoints'),('实验代码',HERE)]
    add(table(['对象','实际路径'],[[n,link(p)] for n,p in paths]))
    add('''模型目录依次为`dctnet_mobilenet24`、`dctnet_mobilenet32`、`dctnet_static24`、`dctnet_static64`。
以上不是新的数据管理体系：继续使用`e2e_v3/runs/<模型>/<运行名>`、原ordinal和manifest映射。
原始数据和旧失败运行均保留；报告只选定完成且通过竞争检查的本次推理复测。

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
# 已完成任务按原入口复用结果，部分训练可由已保存epoch checkpoint接续
bash galp/benchmarks/dct_models/cnn_training_baselines.sh full cnn_training "mobilenet24:native mobilenet32:native resnet24:native resnet64:native"
bash galp/benchmarks/dct_models/cnn_training_baselines.sh full rtx4090_cnn_training_baselines_20260914
# 复用本次推理结果；新测量应指定独立memory-run名，保留现有产物
MEASURE_PROCESS_MEMORY=1 bash galp/benchmarks/dct_models/run_inference.sh "$CUDA_VISIBLE_DEVICES" cnn_memory
# 重新汇总与生成报告；只处理现有结果，不启动GPU实验
MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python galp/benchmarks/dct_models/report_cnn_suite.py --inference-run rtx4090_cnn_memory_isolated_20260915
MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python galp/benchmarks/dct_models/plot_cnn_suite.py
MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python galp/benchmarks/dct_models/write_cnn_performance_report.py
```

### 10.2 数值合同、数据表和时间线''')
    for name in ('cnn_inference.csv','cnn_training.csv','cnn_inference_memory.csv','cnn_trace_breakdown.csv','cnn_data_only.csv'):
        add('- '+link(SUMMARY/name))
    add('逐模型N结果中的`model_profile`完整保存频率列表、分量顺序、mean/std和checkpoint路径；以下链接可直接核对，而不根据DCT编号猜测合同。')
    add(table(['配置','输入合同/完整N结果','B6 E2原始结果','A0 E2原始结果','B6训练时间线'],[
        [model_name(n),link(idx[n,'N projected on']['result'],'model_profile'),link(ti[n,'B6',2]['result'],'epoch_1.json'),
         link(ti[n,'A0 JPEG',2]['result'],'epoch_1.json'),link(DATA/'runs'/d/TRAIN/'native_profile/timeline.png','PNG / 同目录PDF与nsys')]
        for n,d,_ in CONFIGS]))
    add('''训练profile目录为`<模型>/rtx4090_cnn_training_baselines_20260914/<backend>_profile`，
推理profile目录为`<模型>/rtx4090_cnn_memory_isolated_20260915/profiles/<route>`；各自保留`.nsys-rep`、SQLite与breakdown JSON。
报告图表来自这些实际文件，不使用估计值填补未测数据。本次没有新增训练或新的GPU性能实验。

## 11. 结论

CNN结果表明GALP的收益受模型与目标DCT网格共同决定。对MobileNetV2，官方参考输入构造成本远大于模型计算，
离线目标表示、文件范围下推和直接投影可以把完整推理降低到约10–12秒/50K。
对ResNet-50 DCT，主干本身约63秒/50K，输入优化可以减少I/O和显存，却不能抵消官方DCT计算图的较大MAC。

训练中，B6以更大的GPU内存预算换取并行准备和源crop下推。四个配置都完成了两轮全量训练，
相对A0的吞吐改善从2%到67%不等。MobileNet仍有明确的数据端优化空间，ResNet已主要由模型决定。
目前最稳妥的展示结论是：**推理语义保持、输入执行优化和早期训练可行性均有实测证据；最终收敛不劣仍待更长配对训练验证。**''')
    REPORT.write_text('\n'.join(text))
    print(json.dumps(dict(report=str(REPORT), figures=str(ASSETS), inference_rows=len(inference), training_epochs=len(training),
                         traces=len(traces), inference_finished=datetime.fromtimestamp(status['finished']).astimezone().isoformat()), ensure_ascii=False))


def trace_table(values):
    body = []
    for r in values:
        raw = read(r['result'])
        if r['workload'] == 'training':
            h2d = raw['copy_bytes'].get('1', 0)
        else:
            h2d = next(iter(raw.values()))['copies'].get('Host-to-Device', {}).get('bytes', 0)
        body.append([model_name(r['model']), r['route'], *[f(r[k],3) for k in
            ('window_seconds','gpu_input_seconds','gpu_model_seconds','gpu_idle_seconds','overlap_seconds')], f(h2d/2**20,1)])
    return table(['模型','路径','窗口 s','GPU input s','model活跃 s','idle s','input/model重叠 s','窗口H2D MiB'], body)


def make_figures(inference, training, memory, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(ASSETS/f'{name}.png',dpi=180)
        fig.savefig(ASSETS/f'{name}.pdf')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
    for ax,backbone in zip(axes,('MobileNetV2','ResNet-50')):
        selected=[r for r in training if r['epoch']=='2' and r['model'].startswith(backbone)]
        labels=[r['model'].replace(backbone+' ','')+' / '+r['route'].replace('A0 JPEG','A0').replace('rgb_','') for r in selected]
        colors=['#29956a' if r['route']=='B6' else '#e1812c' if r['route']=='A0 JPEG' else '#679cc0' for r in selected]
        bars=ax.barh(labels,[float(r['images_per_second']) for r in selected],color=colors)
        ax.bar_label(bars,fmt='%.0f',padding=3);ax.invert_yaxis();ax.set(title=backbone,xlabel='Images/s, complete epoch 2',xlim=(0,1400))
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('CNN training E2E — RTX 4090, BF16, effective batch 1024')
    save(fig,'training_e2e')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(model,_,_) in zip(axes.flat,CONFIGS):
        selected=[r for r in memory if r['model']==model]
        bars=ax.barh([r['route'] for r in selected],[float(r['nvml_process_peak_mib'])/1024 for r in selected],color='#679cc0')
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.invert_yaxis();ax.set(title=model,xlabel='NVML process peak (GiB)',xlim=(0,25))
    fig.suptitle('50K inference process GPU memory — includes native/DALI allocations')
    save(fig,'inference_memory')
    for workload in ('training','inference'):
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,(model,_,_) in zip(axes.flat,CONFIGS):
            selected=[r for r in traces if r['workload']==workload and r['model']==model and
                      (workload=='training' or r['route'] in ('rgb_dali','R','O','grid_on','projected_on'))]
            names=[r['route'] for r in selected]; values=[]
            for r in selected:
                i,m,o,idle,w=[float(r[k]) for k in ('gpu_input_seconds','gpu_model_seconds','overlap_seconds','gpu_idle_seconds','window_seconds')]
                values.append([m-o,o,i-o,w-idle-m-i+o,idle])
            left=np.zeros(len(values))
            for column,label,color in zip(np.array(values).T,['Model only','Model + input','Input only','Other GPU / copies','GPU idle'],['#29956a','#a478b6','#e1812c','#679cc0','#dddddd']):
                assert np.min(column)>-1e-7
                ax.barh(names,column,left=left,label=label,color=color);left+=column
            ax.invert_yaxis();ax.set(title=model,xlabel='Seconds per profiled window');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
        fig.suptitle(f'GPU activity interval union — {workload}, '+('16,384' if workload=='training' else '4,096')+' images\nProfiler enabled; interval activity is not SM utilization')
        save(fig,f'{workload}_gpu_breakdown')


if __name__=='__main__':
    main()
