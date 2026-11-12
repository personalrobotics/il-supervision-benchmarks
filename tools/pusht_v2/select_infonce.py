"""Held-out InfoNCE of every saved checkpoint of one IBC unit; runs in the LeRobot environment with the plugins, read-only.

    python select_infonce.py SOURCE_ROOT HELD_OUT_JSON CHECKPOINTS_JSON OUT_JSON

For each checkpoint (a pretrained_model directory), the policy in eval mode and its own saved pre-processor score
every frame of the held-out episodes: the cross-entropy of the recorded action against 1024 negatives uniform over the
normalized action box. The negatives are drawn once, from a fixed seed, and shared by every checkpoint, so scores
compare across checkpoints and arms. The score is the mean over frames (lower is better). The sampled action's mean L2
error from the recorded one is also recorded, as a diagnostic: it does not select. Nothing is written but OUT_JSON.
"""
import json
import sys

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402

N_NEG, SEED = 1024, 0
source, held, checkpoints, out = sys.argv[1], json.loads(open(sys.argv[2]).read()), json.loads(open(sys.argv[3]).read()), sys.argv[4]
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = LeRobotDataset("local/pusht-source", root=source, episodes=sorted(held))
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
g = torch.Generator().manual_seed(SEED)
negatives = None
infonce, l2 = {}, {}
for name, path in checkpoints.items():
    cfg = PreTrainedConfig.from_pretrained(path)
    policy = get_policy_class(cfg.type).from_pretrained(path).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
    if negatives is None:
        negatives = torch.rand(len(ds), N_NEG, policy.d_act, generator=g) * 2 - 1
    tot_ce, tot_l2, count = 0.0, 0.0, 0
    torch.manual_seed(SEED)
    with torch.no_grad():
        for batch in loader:
            raw = batch["action"].to(torch.float64)
            nb = pre(batch)
            obs, act = policy.observation(nb), nb["action"][:, 0] if nb["action"].dim() == 3 else nb["action"]
            neg = negatives[count:count + len(obs)].to(device)
            tot_ce += float(policy.infonce(obs, act, neg).sum())
            pred = post(policy.predict_action_chunk(nb)[:, 0]).to("cpu", torch.float64)
            tot_l2 += float(torch.linalg.vector_norm(pred - raw.reshape(pred.shape), dim=1).sum())
            count += len(obs)
    infonce[name], l2[name] = tot_ce / count, tot_l2 / count
json.dump({"frames": count, "episodes": sorted(held), "negatives": N_NEG, "seed": SEED, "infonce": infonce,
           "mean_l2_diagnostic": l2, "device": device}, open(out, "w"), indent=1)
