"""Deterministic matched-DCT update gate, excluded from performance results.

Thresholds fixed before expansion timing: FP32 loss rtol=1e-3/atol=1e-4;
gradients and updated parameters rtol=1e-3/atol=1e-4. Reader gates retain
their existing, tighter input tolerances. Failure blocks the affected pair.
"""
import torch
from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import capture_rng_state, restore_rng_state
from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer


def check_update(model, inputs_a, targets_a, inputs_b, targets_b):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model.train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_hash=tensor_state_sha256(initial)
    rng=capture_rng_state()
    observations=[]
    reference=None
    for inputs,targets in ((inputs_a,targets_a),(inputs_b,targets_b)):
        model.load_state_dict(initial,strict=True)
        restore_rng_state(rng)
        optimizer,decay,scheduler=build_published_optimizer(model,total_updates=375600)
        optimizer.zero_grad(set_to_none=True)
        lr=scheduler.prepare_next_update()
        logits=model(*inputs)
        loss=torch.nn.functional.cross_entropy(logits,targets)
        if not bool(torch.isfinite(logits).all() & torch.isfinite(loss)):
            raise FloatingPointError('Non-finite initial logits/loss')
        loss.backward()
        gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise FloatingPointError('Missing or non-finite gradients')
        if not any(bool(g.count_nonzero()) for g in gradients.values()):
            raise ValueError('All gradients are zero')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();decay.step(lr);scheduler.complete_update()
        parameters={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
        if not all(bool(torch.isfinite(p).all()) for p in parameters.values()):
            raise FloatingPointError('Non-finite updated parameters')
        if not any(not torch.equal(initial[name],p) for name,p in parameters.items()):
            raise ValueError('No parameter updated')
        current=(loss.detach().cpu(),gradients,parameters)
        if reference is None:
            reference=current
        else:
            torch.testing.assert_close(current[0],reference[0],rtol=1e-3,atol=1e-4)
            for actual,expected in zip(current[1:],reference[1:]):
                torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-4)
        observations.append(dict(initial_loss=float(loss),optimizer_updates=scheduler.completed_updates,
            samples=len(targets),gradients_finite=True,parameters_changed=True))
    return dict(initial_model_hash=initial_hash,precision='fp32',observations=observations,
                update_atol=1e-4,update_rtol=1e-3,scope='one diagnostic update on each matched first microbatch; no performance timing')
