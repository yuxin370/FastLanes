"""Actual Conv/Linear MAC counts and resident-input model-only timings."""
import argparse
import json
import time
from pathlib import Path

import torch

import galp.benchmarks.dct_models.backend as B
import galp.benchmarks.dct_models.rgb as rgb
from galp.benchmarks.dct_models.evaluate import samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    entry = samples(1)
    result = {}
    with torch.inference_mode():
        for name, factory, sample in [("RGB", rgb.model, rgb.Inputs(entry)[0][0]),
                                      ("DCT", B.model, B.Reference()(entry[0]["path"]))]:
            net = factory()
            layers, stages, handles = [], {}, []
            for key, module in net.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    def hook(m, inputs, output, key=key):
                        factor = (m.in_channels//m.groups*m.kernel_size[0]*m.kernel_size[1]
                                  if isinstance(m,torch.nn.Conv2d) else m.in_features)
                        layers.append(dict(name=key, output_shape=list(output.shape),
                                           macs=output.numel()*factor))
                    handles.append(module.register_forward_hook(hook))
                if key in ("layer1", "layer2", "layer3", "layer4", "model.0", "model.1", "model.2", "model.3"):
                    handles.append(module.register_forward_hook(
                        lambda m,inputs,output,key=key: stages.update({key:list(output.shape)})))
            logits = net(sample[None])
            for h in handles:
                h.remove()
            assert logits.shape==(1,1000) and torch.isfinite(logits).all()
            net = net.cuda()
            x = sample[None].repeat(64,1,1,1).cuda()
            for _ in range(8):
                net(x)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            for _ in range(32):
                net(x)
            end.record()
            end.synchronize()
            elapsed = time.perf_counter()-wall
            result[name] = dict(input_shape=list(sample.shape), stages=stages,
                                conv_linear_macs_per_image=sum(l['macs'] for l in layers), layers=layers,
                                parameters=sum(p.numel() for p in net.parameters()),
                                resident_batch64_gpu_ms=start.elapsed_time(end)/32,
                                resident_batch64_wall_ms=elapsed/32*1000,
                                warmup_batches=8, measured_batches=32,
                                checkpoint=str(rgb.CHECKPOINT if name=='RGB' else B.CHECKPOINT),
                                dtype="float32", tf32=False, device=torch.cuda.get_device_name(),
                                timing_scope="resident real image tensor repeated to batch64; no I/O, no metrics")
            del net, x
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='layers'} for k,v in result.items()},indent=2))


if __name__=='__main__':
    main()
