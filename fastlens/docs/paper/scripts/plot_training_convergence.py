#!/usr/bin/env python3
"""Plot four completed crop-policy runs without smoothing."""
import json
import statistics

from chart_style import RESULTS, WIDTH, BLUE, GREEN, ORANGE, plt, read, save

SOURCE = RESULTS / "section_6_2_5/shared.csv"
INDEPENDENTS = {
    "swinv2": RESULTS / "section_6_2_5/swinv2_independent.jsonl",
    "mobilenet24": RESULTS / "section_6_2_5/mobilenet24_independent.jsonl",
}
FIGSIZE = (WIDTH, 1.45)


def main():
    rows = read(SOURCE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=.16, right=.96, bottom=.32, top=.96)
    for model, label, shared_color, independent_color in (
        ("swinv2", "SwinV2-T", BLUE, "#9b86b5"),
        ("mobilenet24", "MV2-DCT24", GREEN, ORANGE),
    ):
        shared = [r for r in rows if r["model"] == model]
        assert [int(r["epoch"]) for r in shared] == list(range(1, 301))
        assert all(r["correctness"] == "PASS" for r in shared)
        source = INDEPENDENTS[model]
        independent = [json.loads(line) for line in source.read_text().splitlines()]
        assert [r["epoch"] for r in independent] == list(range(300))
        assert all(r[k] == "PASS" for r in independent for k in ("coverage", "finite", "replica_state"))
        assert all(r["validation"]["samples"] == 50000 for r in independent)
        summary = {}
        for policy, values, color, linestyle, sources in (
            ("shared", [float(r["top1"]) for r in shared], shared_color, "-",
             sorted({r["raw_result"] for r in shared})),
            ("independent", [r["validation"]["top1"] for r in independent], independent_color, "--", [str(source)]),
        ):
            policy_label = "Partition-Shared Crop" if policy == "shared" else "Independent Crop"
            ax.plot(range(1, 301), values, color=color, linestyle=linestyle, label=f"{label}: {policy_label}")
            summary[policy] = dict(
                epochs=len(values), final_top1=values[-1], best_top1=max(values),
                best_epoch=values.index(max(values)) + 1,
                mean_absolute_epoch_change_pp=statistics.mean(abs(b-a) for a, b in zip(values, values[1:])),
                raw_results=sources,
            )
        summary["shared_minus_independent_final_pp"] = summary["shared"]["final_top1"] - summary["independent"]["final_top1"]
        summary["scope"] = "Descriptive, post-hoc summary of all299 adjacent changes; one seed, no equivalence margin. " + (
            "Swin fixes eight ranks, initial model, rank sample orders, LR and contract except crop policy."
            if model == "swinv2" else "MobileNet uses different GPU-count configurations; not a crop-only causal stability test."
        )
        # Rendering must not modify the original experiment records.
        for policy in ('shared', 'independent'):
            values = summary[policy]
            print(model, policy, 'epochs=', values['epochs'],
                  'final_top1=', values['final_top1'], 'best_top1=', values['best_top1'])
    ax.set(xlim=(0, 300), ylim=(0, 85), xticks=[0, 100, 200, 300], yticks=[0, 40, 80])
    ax.set_xlabel("Epoch", fontsize=9.5, labelpad=2)
    ax.set_ylabel("Top-1 (%)", fontsize=9.5, labelpad=3)
    ax.tick_params(labelsize=8.5, pad=2)
    ax.grid(axis="y", alpha=.15, linewidth=.5)
    ax.legend(loc="lower right", frameon=False, labelspacing=.05, ncol=1, fontsize=8,
              handlelength=1.1, handletextpad=.3, borderpad=.15, borderaxespad=.3)
    save(fig, "evaluation/05_training_convergence")


if __name__ == "__main__":
    main()
