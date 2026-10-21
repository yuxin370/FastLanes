#!/usr/bin/env python3
"""Capture local artifact discovery and machine evidence, without benchmarks."""
import ast
import dataclasses
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/inventory"


def capture(name, command):
    if shutil.which(command[0]) is None:
        result = dict(command=command, returncode=None, stdout="", stderr="executable missing")
    else:
        run = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
        result = dict(command=command, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
    path = OUT / "raw" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return dict(raw_log=str(path.relative_to(ROOT)), **result)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    commands = {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
        "gpu": ["nvidia-smi", "-q", "-x"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "cpu": ["lscpu", "--json"],
        "numa": ["numactl", "--hardware"],
        "memory": ["cat", "/proc/meminfo"],
        "storage": ["lsblk", "--json", "-b", "-o", "NAME,MODEL,ROTA,SIZE,FSTYPE,MOUNTPOINT"],
        "filesystem": ["findmnt", "--json", "-T", str(ROOT)],
        "os": ["cat", "/etc/os-release"],
        "cuda_toolkit": ["nvcc", "--version"],
        "nsight": ["nsys", "--version"],
        "coordl_commit": ["git", "-C", str(Path.home() / "tmp/coordl-baseline-source"), "rev-parse", "HEAD"],
        "coordl_api": [str(Path.home() / "tmp/coordl-build-cu128/coordl-python"), "-c",
                       "from nvidia.dali import backend; assert 'cache_size' in backend.GetSchema('FileReader').GetArgumentNames(); print('CoorDL FileReader available')"],
    }
    evidence = {name: capture(name, command) for name, command in commands.items()}
    import torch
    import torchvision
    versions = {}
    for package in ("nvidia-dali-cuda120", "nvidia-dali-cuda130", "ffcv", "numpy", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    software = dict(python=sys.version, executable=sys.executable, pytorch=torch.__version__,
                    torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), packages=versions)
    cuda_devices = [dict(cuda_ordinal=index, name=torch.cuda.get_device_properties(index).name,
                         uuid=str(torch.cuda.get_device_properties(index).uuid))
                    for index in range(torch.cuda.device_count())]
    write("environment.json", dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          hostname=platform.node(), kernel=platform.release(), software=software, evidence=evidence,
          cuda_device_order=cuda_devices,
          platform_policy="Enumerate actual GPUs in gpu raw XML; both requested GPUs share this host if present."))

    from fastlens.benchmarks.baselines.system_rgbnomore.inference import model_factory as inference
    from fastlens.benchmarks.training_pls import model_registry as training
    from fastlens.benchmarks.dct_models.online_crop_suite import PROFILES
    from fastlens.benchmarks.system_dct_major.common import PIPELINES
    models = []
    for task, registry in (("inference", inference), ("training", training)):
        for identifier in registry.MODEL_IDS:
            spec = registry.resolve_model(identifier)
            config = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "builder"}
            models.append(dict(id=identifier, task=task, configuration=config,
                               source=str(Path(registry.__file__).relative_to(ROOT))))
    for profile in PROFILES:
        module = "efun_backend" if profile == "efun" else "backend"
        code = (f"import os,json; os.environ['DCTNET_PROFILE']={profile!r}; "
                f"from fastlens.benchmarks.dct_models import {module} as backend; print(json.dumps(backend.profile()))")
        probe = capture(f"model_{profile}", [sys.executable, "-c", code])
        config = json.loads(probe["stdout"]) if probe["returncode"] == 0 else None
        models.append(dict(id=profile, domain="dct", task="inference,training", configuration=config,
                           discovery_log=probe["raw_log"],
                           checkpoint_exists=Path(config["checkpoint"]).is_file() if config else None,
                           source="fastlens/benchmarks/dct_models/online_crop_suite.py",
                           status="configured; runtime correctness gate pending"))
    for name in ("MobileNetV2", "ResNet-50", "EfficientNet-B0"):
        models.append(dict(id=name, domain="rgb", task="inference,training",
                           source="fastlens/benchmarks/dct_models/rgb.py",
                           comparison="deployment comparison against DCT architectures"))
    # Discover declarations and experiment configurations beyond the primary registries.
    files = subprocess.check_output(["rg", "--files", "--hidden", "fastlens/benchmarks", "fastlens/examples",
                                    "fastlens/profiles", "fastlens/torch"], cwd=ROOT, text=True).splitlines()
    declarations = []
    locations = dict(registries=[], configs=[], launch_scripts=[], dataset_readers=[], benchmarks=[])
    for relative in files:
        path = ROOT / relative
        if path.suffix == ".py":
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    names = [x.id for x in node.targets if isinstance(x, ast.Name)]
                    if any("MODEL" in n or n in ("PROFILES", "PROFILE", "_SPECS") for n in names):
                        declarations.append(dict(source=relative, line=node.lineno,
                                                 names=names, expression=ast.unparse(node.value)))
            if any(x in path.name for x in ("registry", "model_factory", "backend")):
                locations["registries"].append(relative)
            if any(x in relative for x in ("dataset/", "reader", "pipeline.py", "ffcv_dataset")):
                locations["dataset_readers"].append(relative)
            if "/tests/" not in relative:
                locations["benchmarks"].append(relative)
        if path.suffix == ".sh":
            locations["launch_scripts"].append(relative)
        if path.suffix in (".json", ".yaml", ".yml"):
            locations["configs"].append(relative)
    extra = []
    for relative in locations["configs"]:
        if "/dct_retokenization/configs/" in relative:
            extra.append(dict(source=relative, configuration=json.loads((ROOT / relative).read_text()),
                              status="configured model variant; runtime/checkpoint gate pending",
                              comparison="model/system tradeoff, not a causal pipeline control"))
    write("models.json", dict(models=models, additional_configurations=extra,
          discovered_declarations=declarations, locations=locations,
          notes=["eFUN/eFUN-DCT name one DCT model; its RGB deployment baseline is EfficientNet-B0.",
                 "resnet64 is an additional configured DCT model.",
                 "Swin inference 256/window8 and training 224/window7 are distinct configurations."]))
    combinations = []
    for model in ("ViT-Ti", "SwinV2-T", *PROFILES, "MobileNetV2", "ResNet-50", "EfficientNet-B0"):
        for pipeline in ("pytorch_rgb", "cpu_dct", "dali", "coordl", "ffcv", "imagelanes"):
            rgb = model in ("MobileNetV2", "ResNet-50", "EfficientNet-B0")
            dct = model in PROFILES
            if (rgb and pipeline in ("cpu_dct", "imagelanes")) or (dct and pipeline in ("pytorch_rgb", "dali", "coordl", "ffcv")):
                status, reason = "UNSUPPORTED", "input domain incompatible; use separate RGB/DCT deployment comparison"
            elif pipeline in ("coordl", "ffcv") and model != "ViT-Ti":
                status, reason = "UNSUPPORTED", "existing CoorDL/FFCV runner is ViT-specific"
            else:
                status, reason = "CONFIGURED", "execution and correctness gate pending"
            combinations.append(dict(model=model, pipeline=pipeline, status=status, reason=reason,
                primary_training_status="UNSUPPORTED" if pipeline in ("coordl", "ffcv") else status,
                training_note="CoorDL/FFCV have a validation-set update workload, not a full ImageNet train-epoch adapter" if pipeline in ("coordl", "ffcv") else
                "CNN B6 versus CPU A0 changes crop and shuffle; RGB/DCT training also differs in augmentation; deployment comparisons only" if model in PROFILES else
                "Transformer native B6 and standard-DCT B6 share logical schedule/augmentation; RGB remains a deployment comparison"))
    write("pipelines.json", dict(registered_dct_major_pipelines=PIPELINES, combinations=combinations,
          implementations={"imagelanes": ["fastlens/torch/direct_dct.py", "fastlens/benchmarks/dct_models/evaluate_shards.py"],
          "pytorch_rgb_cpu_dct_dali": ["fastlens/benchmarks/baselines/system_rgbnomore/inference/pipeline.py", "fastlens/benchmarks/dct_models/evaluate.py", "fastlens/benchmarks/dct_models/evaluate_rgb.py"],
          "coordl_ffcv": ["fastlens/benchmarks/system_dct_major/pipeline.py", "fastlens/benchmarks/system_dct_major/ffcv_dataset.py"]},
          dataset_note="User-confirmed primary source: existing ImageNet-512 view, not original-resolution ImageNet; both GPUs share host and data paths. Offline model-input backups are excluded from primary online results."))
    print("Inventory written to", OUT)


if __name__ == "__main__":
    main()
