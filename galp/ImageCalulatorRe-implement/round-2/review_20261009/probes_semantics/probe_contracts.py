"""Review probes: execute selected pure functions from the uploaded source.

No Torch, GALP, model, JPEG, or GPU implementation is replaced or simulated.
AST extraction avoids their unavailable imports; results cover only the shown
pure-Python contract and batching logic, not tensor/model correctness.
"""
import ast
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1] / "dct_baselines"


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(), filename=filename)
    kept = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in kept} == set(names)
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(ROOT / filename), "exec"), namespace)


ns = {"VERSION": 1, "json": json, "math": math, "Path": Path}
load_functions("contracts.py", {"frequency_set", "validate_profile", "load_trace"}, ns)
load_functions("training.py", {"update_windows"}, ns)

profile = {
    "version": 1, "representation": "jpeg_b8", "id": "fixture",
    "source_frequencies": [list(range(64))] * 3,
    "dependency_derivation": "full source", "transform": "cnn_source512",
    "output_grids": [56, 56, 56],
}
base_row = {
    "occurrence_id": "0:0:0", "logical_slot": 0, "image_id": "image0",
    "profile_id": "fixture", "augmentation_parameters": {},
    "mixup_partner_occurrence": None, "mixup_lambda": 1.,
    "crop_box": [32, 32, 448, 448], "resize_geometry": [56, 56, 56],
    "target_parameters": {}, "flip": False, "epoch": 0, "view_id": 0,
    "microbatch_id": 0, "target_label": 0,
}

label_results = []
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
    path = Path(directory) / "trace.jsonl"
    for label in (-1, 1000, 999):
        path.write_text(json.dumps(dict(base_row, target_label=label)) + "\n")
        loaded = ns["load_trace"](path, profile, {"image0": {"label": 0}})
        label_results.append({"source_label": 0, "trace_target_label": label,
                              "load_trace_accepted": len(loaded) == 1})

schedule_results = []
for population, batch, pool, accumulation in [(8, 2, 2, 4), (8, 2, 8, 4), (100, 4, 12, 4)]:
    rows = []
    for start in range(0, population, batch):
        for offset in range(min(batch, population - start)):
            rows.append({"epoch": 0, "microbatch_id": start // batch,
                         "closed_pool_id": f"0:{start // pool}"})
    windows = ns["update_windows"](rows, accumulation)
    estimated = math.ceil(population / (batch * accumulation))
    schedule_results.append({"population": population, "batch_size": batch,
        "pool_size": pool, "accumulation": accumulation,
        "prepare_updates_per_epoch_formula": estimated,
        "actual_update_windows_per_epoch": len(windows),
        "actual_samples_per_update": [len(w) for w in windows],
        "prepare_total_for_300_epochs": estimated * 300,
        "actual_total_for_300_epochs": len(windows) * 300})

# Profile order is appended literally by prepare; raw/finish_batch are inspected
# independently below. This section reports the source statements for auditing.
main_tree = ast.parse((ROOT / "__main__.py").read_text())
order_mutations = [ast.unparse(n) for n in ast.walk(main_tree) if isinstance(n, ast.AugAssign)
                   and "operation_order" in ast.unparse(n.target)]

print(json.dumps({"scope": "pure-Python AST probes only; no model/tensor verification",
                  "schedule": schedule_results, "inference_target_validation": label_results,
                  "training_profile_order_append": order_mutations}, ensure_ascii=False, indent=2))
