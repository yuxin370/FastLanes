"""Export durable review fixtures and real CPU evidence; never a performance run.

Run from the repository root. Temporary-directory unittest discovery remains a
separate host command; exported fixture cases deliberately persist their inputs.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new persistent evidence directory")
    parser.add_argument("--models", nargs="*", default=["mobilenet24", "resnet24", "vitti"])
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import PIL
    from PIL import Image
    from galp.benchmarks.dct_baselines.contracts import artifact, write_json
    from galp.benchmarks.dct_baselines.source import freeze_source, source_images
    from galp.benchmarks.dct_baselines.store import build_store
    from galp.benchmarks.dct_baselines.tests.test_training import (
        AccumulatedTrainingTests, FrozenAugmentationTests, NativeSourceTests, UnifiedTrainingTests,
        FrozenTraceTests, CompatibilityReportTests,
    )
    from galp.benchmarks.dct_baselines.tests.test_baselines import (
        StoreTests, NativeTests, TrainingTests, MaterializationTests, ReplayTests, JpegTests,
    )
    from galp.benchmarks.dct_baselines.tests.test_round2 import (
        ScheduleTests, GoldenOrganization, StorageRegressions, TraceLabels, WindowAndReport,
    )
    write_json(root/"environment.json", dict(python=sys.version, executable=sys.executable,
        torch=torch.__version__, cuda_build=torch.version.cuda, numpy=np.__version__, pillow=PIL.__version__,
        repository_revision=subprocess.check_output(["git","rev-parse","HEAD"], cwd=REPO, text=True).strip(),
        cuda_gate="not_run", scope="CPU correctness and isolated regressions; no performance qualification"))
    loader, log = unittest.TestLoader(), io.StringIO()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in
        (TrainingTests, AccumulatedTrainingTests, FrozenAugmentationTests, NativeSourceTests, ScheduleTests, GoldenOrganization))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (root/"readonly-tests.log").write_text(log.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    exported = []
    for cls in (StoreTests, NativeTests, MaterializationTests, ReplayTests, JpegTests,
                FrozenTraceTests, CompatibilityReportTests, StorageRegressions,
                TraceLabels, WindowAndReport, UnifiedTrainingTests):
        for method in loader.getTestCaseNames(cls):
            fixture = cls(method)
            fixture.root = root/"regressions"/method
            fixture.root.mkdir(parents=True)
            try:
                # These are explicit persistent fixture exports, not redirected
                # TemporaryDirectory tests. Assertions execute original code.
                getattr(fixture, method)()
            except Exception:
                (fixture.root/"failure.log").write_text(traceback.format_exc())
                raise
            exported.append(dict(test=f"{cls.__name__}.{method}", status="passed", path=str(fixture.root)))
    write_json(root/"regressions.json", dict(status="passed", fixture_exports=exported,
        readonly_tests=result.testsRun,
        readonly_skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
        full_temporary_directory_suite="run separately on host"))
    print("CPU regressions and persistent JPEG/NPZ/trace fixtures passed", flush=True)
    if not args.models:
        return

    sources = root/"source"
    sources.mkdir()
    rng, entries = np.random.default_rng(11997733), []
    for i in range(3):
        path = sources/f"{i}.jpg"
        Image.fromarray(rng.integers(0,256,(512,512,3),dtype=np.uint8)).save(path, quality=82, subsampling=2)
        entries.append(dict(path=str(path), logical_sample_id=f"synthetic/n{i:08d}/{i}.JPEG", label=i))
    selection, recipe = sources/"selection.json", sources/"recipe.json"
    write_json(selection, dict(split="train", samples=entries))
    write_json(recipe, dict(resize="none; generated 512x512 RGB fixture", encoder="Pillow", encoder_version=PIL.__version__,
                           quality=82, subsampling=2, provenance="this reproduce.py; synthetic, not ImageNet"))
    source_path = sources/"source.json"
    source = freeze_source(selection, source_path, json.loads(recipe.read_text()))
    stores = {}
    for method in ("ic_matched_shared", "ic_matched_batched", "fs_matched_b8"):
        path = root/"stores"/method
        path.parent.mkdir(exist_ok=True)
        build_store(source_images(source), path, method, group_size=3, source_artifact=artifact(source_path))
        stores[method] = path

    def command(directory, name, arguments):
        completed = subprocess.run([sys.executable,"-B","-m","galp.benchmarks.dct_baselines",*arguments],
                                   cwd=REPO, text=True, capture_output=True)
        (directory/f"{name}.log").write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; inspect {directory/name}.log")

    for model in args.models:
        directory = root/"models"/model
        directory.mkdir(parents=True)
        contract = directory/"contract"
        command(directory,"prepare",["prepare","--selection",str(selection),"--source-recipe",str(recipe),
            "--model",model,"--upstream",str(Path.home()/"RGB-no-more"),"--batch-size","2","--pool-size","2",
            "--workers","1","--torch-threads","1","--seed","11997733","--training","--epochs","10",
            "--calibration-count","1","--total-updates","20","--warmup-updates","2","--output",str(contract)])
        config_path = contract/"matched.json"
        config = json.loads(config_path.read_text())
        config["source_manifest"] = artifact(source_path)
        write_json(config_path,config)
        initial = directory/"initial.pt"
        command(directory,"snapshot",["snapshot","--config",str(config_path),"--output",str(initial)])
        for method, store in stores.items():
            backend = "fs" if method == "fs_matched_b8" else "ic"
            calibration, verified = directory/f"{method}.calibration.json", directory/f"{method}.verified.json"
            common=["--config",str(config_path),"--snapshot",str(initial),"--backend",backend,"--root",str(store),
                    "--device","cpu","--updates","10"]
            command(directory,f"{method}.calibrate",["calibrate",*common,"--input-atol","0","--model-atol","0",
                "--state-atol","0","--rtol","0","--output",str(calibration)])
            command(directory,f"{method}.verify",["verify-training",*common,"--calibration",str(calibration),"--output",str(verified)])
        print(f"{model}: IC / IC-b / FS each passed 10 real CPU model updates",flush=True)
    write_json(root/"completion.json",dict(status="passed", models=args.models, updates_per_backend=10,
        cpu_evidence="models/*/*.verified.json", cuda_gate="not_run", headline_eligible=False))


if __name__ == "__main__":
    main()
