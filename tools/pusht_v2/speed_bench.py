"""Training- and inference-step speed of v2's IBC policy under precision and compilation settings (LeRobot env).

    python speed_bench.py OUT_JSON [--steps N]           every setting, each in its own process
    python speed_bench.py OUT_JSON --setting NAME        one setting (Accelerate fixes its precision per process)

The step is lerobot-train's: the plugin's IBCPolicy (v2's protocol arguments) prepared by an Accelerator with the
setting's mixed precision, AdamW at the protocol's learning rate, batch 256, inputs in the normalized box. Settings:
fp32_tf32 (today: lerobot-train enables TF32 matmuls), bf16 and fp16 (Accelerate mixed precision; --policy.use_amp is
fp16), and compile (torch.compile of the energy network) with fp32_tf32 and bf16. Each setting reports seconds per
training step and per inference call (50 observations, the evaluator's batch), and, on one fixed batch, parameters and
negatives, the relative difference of its loss and gradients from strict FP32 (TF32 off). Timings are wall clock on a
shared GPU: compare settings with each other, not with other hosts or runs.
"""
import argparse
import json
import time

import torch
from accelerate import Accelerator
from lerobot.configs.types import FeatureType, PolicyFeature

import lerobot_policy_ibc  # noqa: F401  (registers the plugin)
from lerobot_policy_ibc.configuration_ibc import IBCConfig
from lerobot_policy_ibc.modeling_ibc import IBCPolicy

FEATURES = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(32,))}
SETTINGS = {"fp32_tf32": ("no", False), "bf16": ("bf16", False), "fp16": ("fp16", False),
            "compile_fp32_tf32": ("no", True), "compile_bf16": ("bf16", True)}


def policy(seed=0):
    torch.manual_seed(seed)
    cfg = IBCConfig(input_features=FEATURES, output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
                    device="cuda")
    return IBCPolicy(cfg).cuda()


def batch(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return {"observation.state": torch.rand(n, 4, device="cuda", generator=g) * 2 - 1,
            "observation.environment_state": torch.rand(n, 32, device="cuda", generator=g) * 2 - 1,
            "action": torch.rand(n, 1, 2, device="cuda", generator=g) * 2 - 1}


def grads(p, acc, b, neg):
    """Loss and flat gradient of the InfoNCE loss on a fixed batch and fixed negatives, as one deterministic function:
    eval mode (no dropout), no positive jitter, and an unscaled backward (no fp16 loss scale)."""
    p.eval()
    p.zero_grad()
    with acc.autocast():
        loss = p.infonce(p.observation(b), b["action"][:, 0], neg).mean()
    loss.backward()
    p.train()
    return loss.detach().float(), torch.cat([q.grad.flatten().float() for q in p.parameters()])


def one(name, steps) -> dict:
    """One setting's timings and numerics, in this process."""
    mp, comp = SETTINGS[name]
    acc = Accelerator(mixed_precision=mp)
    b, neg = batch(256, 1), torch.rand(256, 1024, 2, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2)) * 2 - 1
    torch.backends.cuda.matmul.allow_tf32 = False  # the strict FP32 reference, without autocast
    ref_p = policy()
    ref_p.eval()
    with torch.autocast("cuda", enabled=False):
        ref_p.zero_grad()
        ref_loss = ref_p.infonce(ref_p.observation(b), b["action"][:, 0], neg).mean()
        ref_loss.backward()
    ref_loss, ref_grad = ref_loss.detach(), torch.cat([q.grad.flatten() for q in ref_p.parameters()])
    torch.backends.cuda.matmul.allow_tf32 = True  # as lerobot-train and lerobot-eval set it
    p = policy()
    if comp:
        p.energy = torch.compile(p.energy)
    opt = torch.optim.AdamW(p.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    p, opt = acc.prepare(p, opt)
    inner = acc.unwrap_model(p)
    loss, grad = grads(inner, acc, b, neg)
    numerics = {"loss_rel_diff": float(abs(loss - ref_loss) / abs(ref_loss)),
                "grad_rel_diff": float((grad - ref_grad).norm() / ref_grad.norm())}
    data = [batch(256, 10 + i) for i in range(8)]
    for i in range(steps + 20):
        if i == 20:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with acc.autocast():
            l, _ = p(data[i % 8])
        acc.backward(l)
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    train = (time.perf_counter() - t0) / steps
    inner.eval()
    obs = batch(50, 99)
    with torch.no_grad(), acc.autocast():
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            inner.predict_action_chunk(obs)
    torch.cuda.synchronize()
    infer = (time.perf_counter() - t0) / 20
    return {"train_s_per_step": train, "inference_s_per_call": infer, **numerics}


def main(out, steps):
    import subprocess
    import sys
    results = {}
    for name in SETTINGS:
        part = f"{out}.{name}.json"
        subprocess.run([sys.executable, "-W", "ignore", __file__, part, "--steps", str(steps), "--setting", name],
                       check=True)
        results[name] = json.load(open(part))
        print(name, json.dumps(results[name]), flush=True)
    base = results["fp32_tf32"]
    for r in results.values():
        r["train_speedup"] = base["train_s_per_step"] / r["train_s_per_step"]
        r["inference_speedup"] = base["inference_s_per_call"] / r["inference_s_per_call"]
    json.dump({"steps": steps, "batch": 256, "results": results,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--setting", choices=list(SETTINGS))
    a = ap.parse_args()
    if a.setting:
        json.dump(one(a.setting, a.steps), open(a.out, "w"))
    else:
        main(a.out, a.steps)
