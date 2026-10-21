"""Summarize the block-descriptor frequency sweep and its separate CUPTI capture."""
import csv
import json
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'results/compact_projection_20261001'
CANONICAL = ROOT / 'results/selectivity_4090_20261001'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    for k in (8, 16, 24, 32, 48, 64):
        for arm in ('selected', 'late'):
            paths = [OUT / 'frequency' / f'k{k}_{arm}_trial{i}.json' for i in (1, 2, 3)]
            runs = [json.loads(p.read_text()) for p in paths]
            times = [r['elapsed_s'] for r in runs]
            for r in runs:
                assert r['samples'] == 4096 and r['coefficients_exact']
                assert sum(b['decoded_gather_item_count'] for b in r['native_batches']) == 4096 * 6144
                assert sum(b['decoded_projection_item_count'] for b in r['native_batches']) == 0
            def total(name):
                values = [sum(b[name] for b in r['native_batches']) for r in runs]
                assert len(set(values)) == 1
                return values[0]
            rows.append(dict(k=k, arm=arm, median_s=statistics.median(times), min_s=min(times), max_s=max(times),
                cv=statistics.stdev(times)/statistics.mean(times),
                read_bytes=total('rowgroup_storage_bytes_read'),
                workset_h2d_bytes=total('workset_upload_dma_bytes'),
                planning_median_s=statistics.median(sum(b['planning_ms'] for b in r['native_batches'])/1000 for r in runs),
                requested_values=4096*6144*k, correctness_checked_images=32,
                block_descriptors=total('decoded_gather_item_count'),
                raw_results=';'.join(str(p.relative_to(ROOT)) for p in paths)))
    write_csv(OUT / 'frequency_summary.csv', rows)
    write_csv(CANONICAL / 'frequency_summary.csv', rows)

    native = json.loads((OUT / 'diagnostic/native.json').read_text())
    transfers = []
    with sqlite3.connect(OUT / 'diagnostic/capture.sqlite') as db:
        for arm in ('selected', 'late'):
            (start, end), = db.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s "
                "ON n.textId=s.id WHERE COALESCE(n.text,s.value)=?", (f'frequency.{arm}',)).fetchall()
            copies = db.execute('SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
            assert all(start <= a <= b <= end for a, b, _ in copies)
            kernel_ns, = db.execute('SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL '
                'WHERE start>=? AND end<=?', (start, end)).fetchone()
            batches = native['native_batches'][arm]
            workset = sum(b['workset_upload_dma_bytes'] for b in batches)
            total = sum(size for _, _, size in copies)
            transfers.append(dict(arm=arm, k=8, samples=4096, total_h2d_bytes=total,
                workset_h2d_bytes=workset, h2d_copy_count=len(copies),
                h2d_cumulative_ms=sum(b-a for a,b,_ in copies)/1e6,
                projection_items=sum(b['decoded_projection_item_count'] for b in batches),
                gather_items=sum(b['decoded_gather_item_count'] for b in batches),
                kernel_cumulative_ms=kernel_ns/1e6, other_h2d_bytes=total-workset))
    write_csv(OUT / 'diagnostic/h2d.csv', transfers)
    summary = json.loads((CANONICAL / 'summary.json').read_text())
    summary.update(compact_frequency=rows, frequency_h2d=transfers,
                   frequency_correctness_images=32,
                   frequency_result_source=str(OUT.relative_to(ROOT)))
    summary['scope_notes'] = [s.replace('Compact frequency timing runs checked 8 images;',
        'Compact frequency timing runs checked 32 images;') for s in summary['scope_notes']]
    (CANONICAL / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), constrained_layout=True)
    for arm, label in [('selected', 'Early selection'), ('late', 'Late selection')]:
        values = [r for r in rows if r['arm'] == arm]
        xs = [r['k'] for r in values]
        ys = [r['median_s'] for r in values]
        axes[0].errorbar(xs, ys, yerr=[[r['median_s']-r['min_s'] for r in values],
            [r['max_s']-r['median_s'] for r in values]], marker='o', capsize=3, label=label)
        axes[1].plot(xs, [r['read_bytes']/1e9 for r in values], marker='o', label=label)
    for ax in axes:
        ax.set_xlabel('Selected frequencies K'); ax.set_xticks([8,16,24,32,48,64]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Preparation time (s)'); axes[0].legend()
    axes[1].set_ylabel('Issued payload (GB)')
    fig.savefig(CANONICAL / 'frequency.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 2.7), constrained_layout=True)
    payload = [r['workset_h2d_bytes']/1e9 for r in transfers]
    ax.bar(['Early', 'Late'], payload, label='Compressed workset')
    ax.bar(['Early', 'Late'], [r['other_h2d_bytes']/1e9 for r in transfers], bottom=payload,
           label='Placement and bindings')
    ax.set_ylabel('Total H2D (GB), K=8'); ax.legend()
    fig.savefig(CANONICAL / 'frequency_h2d.png', dpi=180); plt.close(fig)

    early = next(r for r in rows if r['k'] == 8 and r['arm'] == 'selected')
    late = next(r for r in rows if r['k'] == 8 and r['arm'] == 'late')
    section = r'''\paragraph{Frequency selectivity and placement cost.}
A separate frequency control prepares contiguous int16 coefficient tensors
for 4,096 images, without resize or a model. It compares early column
selection with reading all 64 columns and selecting the same $K$ on the
GPU. Coefficient tensors agree exactly on 32 sampled images at each
frequency budget. This generic compact output interface uses a different
materialization path from the model-profile experiments above;
``compact'' here denotes the output tensor, not the image-major storage
layout. Both arms use the same block-major source dataset.

Table~\ref{tab:compact-frequency} separates payload savings from preparation
time. At $|K|=8$, early selection takes '''
    section += f"{early['median_s']:.3f}~s versus {late['median_s']:.3f}~s for late selection,\n"
    section += f"while issued payload falls from {late['read_bytes']/1e9:.3f} to {early['read_bytes']/1e9:.3f}~GB.\n"
    section += 'At $|K|=48$ and $64$, the observed timing ranges overlap.\n'
    section += r'''
\begin{table}[t]
\centering
\caption{Generic compact frequency projection on RTX~4090, 4,096 images.
Times are three-trial medians after one warm-up pass. Late selection reads
all 64 columns, then constructs the same contiguous output; reader startup
is excluded. Both arms use block-level placement descriptors.}
\label{tab:compact-frequency}
\fontsize{8}{9}\selectfont\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rrrr@{}}
\toprule
$|K|$ & Early (s) & Late (s) & Issued bytes, early/late \\
\midrule
'''
    for k in (8, 16, 24, 32, 48, 64):
        e = next(r for r in rows if r['k'] == k and r['arm'] == 'selected')
        l = next(r for r in rows if r['k'] == k and r['arm'] == 'late')
        section += f"{k} & {e['median_s']:.3f} & {l['median_s']:.3f} & {e['read_bytes']/l['read_bytes']:.3f} " + '\\\\\n'
    section += r'''\bottomrule
\end{tabular}
\end{table}

The compact gather represents each source block with one 16-byte
placement descriptor and shares coefficient bindings within a rowgroup.
The GPU computes the selected coefficient slot rather than receiving a
separate placement record for each value. For 4,096 images with 6,144
component blocks each, block descriptors occupy 0.403~GB independently
of $|K|$. Enumerating one 32-byte record per selected coefficient would
instead require 6.442~GB even at $|K|=8$.
'''
    e, l = transfers
    section += f"A separate CUPTI capture at $|K|=8$ measures total H2D of {e['total_h2d_bytes']/1e9:.3f}~GB\n"
    section += f"for early selection versus {l['total_h2d_bytes']/1e9:.3f}~GB for late selection; compressed-workset\n"
    section += f"transfers account for {e['workset_h2d_bytes']/1e9:.3f} and {l['workset_h2d_bytes']/1e9:.3f}~GB, respectively.\n"
    planning = [r['planning_median_s'] for r in rows]
    section += f"CPU planning still takes {min(planning):.2f}--{max(planning):.2f}~s across the measured configurations.\n"
    section += r'''Block-level planning and placement therefore impose costs that frequency
selection does not remove. The reduction in compressed bytes is larger
than the improvement in preparation time; these input-only measurements
do not establish an end-to-end model speedup.

'''
    paper = ROOT / 'fastlens/docs/paper/sections/07_evaluation.tex'
    text = paper.read_text()
    start = text.index(r'\paragraph{Descriptor cost in generic compact projection.}')
    end = text.index(r'\subsection{Read Coalescing and Cold-Cache Access}', start)
    text = text[:start] + section + text[end:]
    text = text.replace('generic compact projection shows that per-value placement metadata can\n'
        'overwhelm compressed-data savings.',
        'the frequency study shows that block-level planning and placement\n'
        'remain significant even when compressed payload shrinks.')
    paper.write_text(text)
    print(json.dumps(dict(frequency=rows, h2d=transfers), indent=2))


if __name__ == '__main__':
    main()
