"""Held-out action error of every saved checkpoint of one training unit; runs in the LeRobot environment, read-only.

    python lerobot_select.py SOURCE_ROOT HELD_OUT_JSON CHECKPOINTS_JSON OUT_JSON

For each checkpoint (a pretrained_model directory), the policy and its own saved pre/post-processors predict the
action of every frame of the held-out real episodes, one observation each; the score is the mean L2 distance from
the recorded action, in raw action units. Nothing is written but OUT_JSON.
"""
import json
import sys

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

source, held, checkpoints, out = sys.argv[1], json.loads(open(sys.argv[2]).read()), json.loads(open(sys.argv[3]).read()), sys.argv[4]
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = LeRobotDataset("local/pusht-source", root=source, episodes=sorted(held))
loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
scores = {}
for name, path in checkpoints.items():
    policy = ACTPolicy.from_pretrained(path).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            target = batch["action"].to(torch.float64)
            pred = post(policy.predict_action_chunk(pre(batch))[:, 0]).to("cpu", torch.float64)
            total += float(torch.linalg.vector_norm(pred - target, dim=1).sum())
            count += len(target)
    scores[name] = total / count
json.dump({"frames": count, "episodes": sorted(held), "mean_l2": scores, "device": device}, open(out, "w"), indent=1)
