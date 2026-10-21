"""Plot matched unprofiled throughput and overlap-preserving Nsight timelines."""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
from pathlib import Path
import sqlite3
import statistics

from fastlens.benchmarks.profiling.input_motivation import MODELS, ARMS
from fastlens.benchmarks.profiling.analyze_training_nsys import _merge

LABELS = dict(mobilenet="MobileNetV2", resnet="ResNet-50", vitti="ViT-Ti", swinv2="SwinV2-T",
              efficientnet="EfficientNet-B0")
LABELS.update({"mobilenet24": "MobileNetV2 DCT-24", "mobilenet32": "MobileNetV2 DCT-32",
               "resnet24": "ResNet-50 DCT-24", "resnet64": "ResNet-50 DCT-64",
               "vitti-dct": "ViT-Ti DCT", "swinv2-dct": "SwinV2-T DCT", "efun": "eFUN",
               "efun-l": "eFUN-L", "efun-s": "eFUN-S", "efun-s-plus": "eFUN-S+"})
DOMAIN = "rgb"
LANES = ("CPU submission", "CPU input API / wait", "H2D", "GPU preprocessing",
         "Model kernels", "Other GPU activity", "GPU idle*")
COLORS = ("#888888", "#bd7954", "#8064a2", "#2589ae", "#499260", "#b4a76b", "#d7d7d7")


def clip(intervals, start, end):
    return [(max(a, start), min(b, end)) for a, b in intervals if a < end and b > start]


def duration(intervals):
    return sum(b - a for a, b in _merge(intervals))


def intersection(a, b):
    return duration(a) + duration(b) - duration(a + b)


def complement(intervals, start, end):
    result, cursor = [], start
    for a, b in _merge(clip(intervals, start, end)):
        if cursor < a:
            result.append((cursor, a))
        cursor = b
    if cursor < end:
        result.append((cursor, end))
    return result


def contains(index, timestamp):
    intervals, starts = index
    position = bisect.bisect_right(starts, timestamp) - 1
    return position >= 0 and timestamp < intervals[position][1]


def read_trace(path):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    nvtx = connection.execute(
        "SELECT coalesce(n.text,s.value),n.start,n.end,n.globalTid "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL"
    ).fetchall()
    captures = [(a, b, tid) for name, a, b, tid in nvtx if name == "motivation.capture"]
    if len(captures) != 1:
        raise ValueError(f"Expected one measured capture: {path}")
    start, end, main_tid = captures[0]
    threads = dict(connection.execute(
        "SELECT t.globalTid,s.value FROM ThreadNames t JOIN StringIds s ON t.nameId=s.id"))
    runtime = {(tid >> 24, corr): (a, b, tid, name) for corr, a, b, tid, name in connection.execute(
        "SELECT r.correlationId,r.start,r.end,r.globalTid,s.value "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id")}
    if any("GraphLaunch" in name for _, _, _, name in runtime.values()):
        raise ValueError(f"CUDA Graph replay is forbidden in this experiment: {path}")
    scopes = {}
    for category, prefixes in (("model", ("training.model.", "training.optimizer.")),
                               ("input", ("training.loader.", "training.input_handoff"))):
        intervals = _merge([(a, b) for name, a, b, tid in nvtx
                            if tid == main_tid and name and name.startswith(prefixes)])
        scopes[category] = intervals, [a for a, b in intervals]
    lanes = {name: [] for name in LANES}
    lanes["CPU input API / wait"] = scopes["input"][0]
    lanes["CPU submission"] = [(a, b) for a, b, tid, name in runtime.values()
                                if (tid >> 24) == (main_tid >> 24)
                                and any(token in name for token in ("Launch", "Memcpy", "Memset"))]
    kernel_count = other_count = graph_count = 0
    for a, b, pid, corr, name in connection.execute(
        "SELECT k.start,k.end,k.globalPid,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id WHERE k.start<? AND k.end>?", (end, start)):
        category = "Other GPU activity"
        launch = runtime.get((pid >> 24, corr))
        if launch:
            graph_count += "GraphLaunch" in launch[3]
            submitted, _, tid, _ = launch
            thread = threads.get(tid, "")
            if "[DALI]" in thread or "dali::" in name or "nvjpeg::" in name:
                category = "GPU preprocessing"
            elif tid == main_tid and contains(scopes["input"], submitted):
                category = "GPU preprocessing"
            elif thread.startswith("pt_autograd") or (tid == main_tid and contains(scopes["model"], submitted)):
                category = "Model kernels"
        lanes[category].append((a, b))
        kernel_count += 1
        other_count += category == "Other GPU activity"
    copy_names = dict(connection.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
    h2d_bytes = 0
    for a, b, kind, size in connection.execute(
        "SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?", (end, start)):
        category = "H2D" if copy_names[kind] == "Host-to-Device" else "Other GPU activity"
        lanes[category].append((a, b))
        if category == "H2D":
            h2d_bytes += size
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone():
        lanes["Other GPU activity"] += connection.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start<? AND end>?", (end, start)).fetchall()
    for name in LANES[:-1]:
        lanes[name] = _merge(clip(lanes[name], start, end))
    gpu = [interval for name in LANES[2:-1] for interval in lanes[name]]
    lanes["GPU idle*"] = complement(gpu, start, end)
    if duration(gpu) + duration(lanes["GPU idle*"]) != end - start:
        raise AssertionError("GPU union and complement do not cover the capture")
    connection.close()
    return dict(start=start, end=end, lanes=lanes, h2d_bytes=h2d_bytes,
                kernels=kernel_count, unclassified_kernels=other_count, graph_kernels=graph_count)


def write_csv(path, rows):
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def throughput(root, output, plt):
    raw, aggregate = [], []
    modes = set()
    for model in MODELS:
        results = {}
        for arm in ARMS:
            results[arm] = []
            for trial in range(3):
                path = root / f"{model}-{arm}-r{trial}" / "result.json"
                result = json.loads(path.read_text())
                assert result.get("domain", "rgb") == DOMAIN
                modes.add(result.get("mode", "training"))
                assert (result["model"], result["arm"], result["trial"], result["profile"]) == (model, arm, trial, False)
                assert result["cache_before"]["resident_fraction"] == 1
                assert result["cache_after"]["resident_fraction"] == 1
                assert result["cuda_graphs"] is False
                results[arm].append(result)
        keys = ("batch", "accumulation", "warmup_updates", "updates", "seed", "precision", "compile",
                "cuda_graphs", "tf32", "gpu", "torch", "cuda", "model_parameters", "training")
        reference = results["model-only"][0]
        assert all(all(row[k] == reference[k] for k in keys) for rows in results.values() for row in rows)
        for arm in ARMS:
            ratios = []
            for trial, result in enumerate(results[arm]):
                ratio = result["images_per_second"] / results["model-only"][trial]["images_per_second"]
                ratios.append(ratio)
                raw.append(dict(model=LABELS[model], arm=arm, trial=trial,
                                images_per_second=result["images_per_second"], normalized=ratio,
                                seconds=result["seconds"], images=result["images"], precision=result["precision"],
                                source=str(root / f"{model}-{arm}-r{trial}" / "result.json")))
            aggregate.append(dict(model=LABELS[model], arm=arm,
                                  images_per_second_median=statistics.median(r["images_per_second"] for r in results[arm]),
                                  normalized_median=statistics.median(ratios), normalized_min=min(ratios), normalized_max=max(ratios)))
    assert len(modes) == 1
    mode = modes.pop()
    write_csv(output / "throughput_trials.csv", raw)
    write_csv(output / "throughput_summary.csv", aggregate)
    fig, ax = plt.subplots(figsize=(12, 3.2) if DOMAIN == "dct" else (6.8, 2.6), layout="constrained")
    draw_bars(ax, aggregate, mode)
    title = "DCT models; DALI not integrated; CUDA Graph disabled" if DOMAIN == "dct" else "CUDA Graph disabled"
    ax.set_title(title, loc="left", fontsize=8, pad=23)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"throughput.{ext}", dpi=200)
    plt.close(fig)
    return aggregate, mode


def draw_bars(ax, aggregate, mode):
    arm_labels = ("Model only", "PyTorch (JPEG → DCT)") if DOMAIN == "dct" else ("Model only", "PyTorch", "DALI")
    for j, (arm, label, color) in enumerate(zip(ARMS, arm_labels, ("#bfc5cb", "#d89150", "#3489ad"))):
        rows = [r for r in aggregate if r["arm"] == arm]
        centers = [i + (j - (len(ARMS) - 1) / 2) * .24 for i in range(len(MODELS))]
        heights = [r["normalized_median"] for r in rows]
        bars = ax.bar(centers, heights, width=.22, label=label, color=color,
                      yerr=([h-r["normalized_min"] for h,r in zip(heights,rows)],
                            [r["normalized_max"]-h for h,r in zip(heights,rows)]),
                      error_kw={"elinewidth": .8, "capsize": 2})
        if DOMAIN != "dct" or arm != "model-only":
            labels = [f"{h:.4f}" if h < .02 else f"{h:.2f}" for h in heights]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.axhline(1, color="#777777", ls="--", lw=.6)
    ax.set(xticks=range(len(MODELS)), xticklabels=[LABELS[m].replace(" DCT", "\nDCT") for m in MODELS], ylim=(0, max(1.16, max(r["normalized_max"] for r in aggregate) + .16)),
           ylabel=f"{mode.capitalize()} throughput / model only")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.20), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def timelines(root, output, plt, mode):
    trace_models = ("mobilenet24", "vitti-dct") if DOMAIN == "dct" else ("mobilenet", "vitti")
    trace_arms = ("pytorch",) if DOMAIN == "dct" else ("pytorch", "dali")
    fig, axes = plt.subplots(len(trace_arms), 2, figsize=(7.4, 3.1 if DOMAIN == "dct" else 5.3),
                             sharex=True, squeeze=False, layout="constrained")
    fig.suptitle(f"{DOMAIN.upper()} {mode} pipeline activity (CUDA Graph disabled)", fontsize=9)
    summaries, panels = [], []
    with gzip.open(output / "timeline_intervals.csv.gz", "wt") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "arm", "lane", "start_ns", "end_ns"])
        for col, model in enumerate(trace_models):
            for row, arm in enumerate(trace_arms):
                source = root / f"{model}-{arm}-trace" / "trace.sqlite"
                trace = read_trace(source)
                if arm == trace_arms[-1]:
                    panels.append((model, arm, trace))
                start, end, lanes = trace["start"], trace["end"], trace["lanes"]
                # Fixed selection rule: 200 ms centered in the complete capture.
                center = (start + end) // 2
                left, right = center - 100_000_000, center + 100_000_000
                if left < start or right > end:
                    raise ValueError("Capture shorter than the fixed 200 ms display window")
                ax = axes[row, col]
                for i, (name, color) in enumerate(zip(LANES, COLORS)):
                    for a, b in lanes[name]:
                        writer.writerow([model, arm, name, a, b])
                    displayed = clip(lanes[name], left, right)
                    ax.broken_barh([((a-left)/1e6, (b-a)/1e6) for a,b in displayed],
                                   (i-.33, .66), color=color, linewidth=0, rasterized=True)
                    summaries.append(dict(model=LABELS[model], arm=arm, lane=name,
                                          capture_ms=(end-start)/1e6, union_ms=duration(lanes[name])/1e6,
                                          fraction=duration(lanes[name])/(end-start),
                                          overlap_model_ms=intersection(lanes[name], lanes["Model kernels"])/1e6,
                                          overlap_idle_ms=intersection(lanes[name], lanes["GPU idle*"])/1e6,
                                          display_union_ms=duration(displayed)/1e6,
                                          capture_start_ns=start, capture_end_ns=end,
                                          display_start_ns=left, display_end_ns=right,
                                          h2d_bytes=trace["h2d_bytes"], kernels=trace["kernels"],
                                          graph_kernels=trace["graph_kernels"],
                                          unclassified_kernels=trace["unclassified_kernels"], source=str(source)))
                ax.set(title=f"{LABELS[model]} / {'DALI' if arm == 'dali' else 'PyTorch'}",
                       yticks=range(len(LANES)), yticklabels=LANES if col == 0 else [],
                       xlim=(0,200), ylim=(len(LANES)-.5,-.75))
                ax.tick_params(axis="y", length=0, labelsize=7)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.grid(axis="x", alpha=.15)
                if row == len(trace_arms) - 1:
                    ax.set_xlabel("Elapsed time in middle 200 ms (ms)")
    fig.supxlabel("* GPU idle: no recorded kernels, copies, or memsets from this job. Lanes overlap.", fontsize=7)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"timeline.{ext}", dpi=300)
    plt.close(fig)
    write_csv(output / "timeline_union.csv", summaries)
    return panels


def combined(aggregate, mode, panels, output, plt):
    fig = plt.figure(figsize=(12, 5.4) if DOMAIN == "dct" else (7.4, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.15))
    ax = fig.add_subplot(grid[0, :])
    draw_bars(ax, aggregate, mode)
    title = "(a) Warm-cache throughput (CUDA Graph disabled)"
    if DOMAIN == "dct":
        title += "; DCT models; DALI not integrated"
    ax.set_title(title, loc="left", fontsize=8, pad=23)
    for column, (model, arm, trace) in enumerate(panels):
        ax = fig.add_subplot(grid[1, column])
        center = (trace["start"] + trace["end"]) // 2
        left, right = center - 100_000_000, center + 100_000_000
        for i, (name, color) in enumerate(zip(LANES, COLORS)):
            events = clip(trace["lanes"][name], left, right)
            ax.broken_barh([((a-left)/1e6, (b-a)/1e6) for a,b in events],
                           (i-.33, .66), color=color, linewidth=0, rasterized=True)
        ax.set(yticks=range(len(LANES)), yticklabels=LANES if column == 0 else [],
               xlim=(0,200), ylim=(len(LANES)-.5,-.75), xlabel="Elapsed time (ms)")
        ax.set_title(f"({'b' if column == 0 else 'c'}) {LABELS[model]} / {'DALI' if arm == 'dali' else 'PyTorch'}", loc="left", fontsize=8)
        ax.tick_params(axis="y", length=0, labelsize=7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
    fig.supxlabel("* No recorded job GPU activity. Timeline lanes overlap; they are not additive.", fontsize=7)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"motivation.{ext}", dpi=300)
    plt.close(fig)


def main():
    global MODELS, ARMS, DOMAIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--domain", choices=("rgb", "dct"), default="rgb")
    args = parser.parse_args()
    DOMAIN = args.domain
    if DOMAIN == "dct":
        from fastlens.benchmarks.profiling.input_motivation import DCT_MODELS, DCT_ARMS
        MODELS, ARMS = DCT_MODELS, DCT_ARMS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    output = args.root / "figures"
    output.mkdir(exist_ok=True)
    aggregate, mode = throughput(args.root, output, plt)
    panels = timelines(args.root, output, plt, mode)
    combined(aggregate, mode, panels, output, plt)


if __name__ == "__main__":
    main()
