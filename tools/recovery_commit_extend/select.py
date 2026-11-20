"""#85 fresh selection (recovery-commit-extend.md "Fresh starts"), from committed records and seeds alone.

    python3 tools/recovery_commit_extend/select.py > tools/recovery_commit_extend/selection.json

An episode is eligible if it was not used to fit model 0 and no recovery controller ever ran from it; #82's anchor
and control episodes are now excluded too. The selection
seed is derived from the sha256 of the eligible list, so it is a function of the inventory alone. No demonstration
state is read and no perturbed start is generated here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "docs/studies/evidence"
EPISODES = [str(i) for i in range(100)]
H, BUDGET, LAST = 64, 127, 499
T_MAX = LAST - H - BUDGET  # 308: #82's fixed-H comparator keeps H = 64 through a full recovery budget
ANCHOR_EPISODES, CONTROL_EPISODES, ANCHORS, CONTROLS_PER_EPISODE = 15, 5, 50, 4


def rng(*k):
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in k]))


model0 = json.loads((EV / "recovery-learned/study/identity.json").read_text())["identity"]["models"]["0"]["episodes"]
anchors76 = {str(a[0]) for a in json.loads((EV / "recovery-receding/study/record.json").read_text())["identity"]["anchors"]}
i79 = json.loads((EV / "recovery-switch/study/identity.json").read_text())["identity"]
s82 = json.loads((ROOT / "tools/recovery_fixed_horizon/selection.json").read_text())
excluded = {"model0_fit": sorted(model0, key=int),
            "recovery_from_starts_#70_#73_#76_#79": sorted(anchors76, key=int),
            "recovery_from_controls_#79_study": sorted(i79["control_episodes"], key=int),
            "recovery_from_controls_#79_smoke": ["0", "1"],
            "recovery_from_starts_#82": sorted(s82["anchor_episodes"], key=int),
            "recovery_from_controls_#82": sorted(s82["control_episodes"], key=int)}
assert set(anchors76) <= set(model0)  # every anchor episode is a training episode
eligible = sorted(set(EPISODES) - set().union(*map(set, excluded.values())), key=int)
digest = hashlib.sha256(json.dumps(eligible).encode()).hexdigest()
order = [eligible[i] for i in rng(171, 7, 0, int(digest[:16], 16)).permutation(len(eligible))]
anchor_eps, control_eps = order[:ANCHOR_EPISODES], order[ANCHOR_EPISODES:ANCHOR_EPISODES + CONTROL_EPISODES]
per = [ANCHORS // ANCHOR_EPISODES + (i < ANCHORS % ANCHOR_EPISODES) for i in range(ANCHOR_EPISODES)]
anchors = [[e, int(t)] for e, k in zip(anchor_eps, per)
           for t in sorted(rng(171, 7, 1, int(e)).choice(T_MAX + 1, k, replace=False))]
controls = [[e, int(t)] for e in control_eps
            for t in sorted(rng(171, 7, 2, int(e)).choice(T_MAX + 1, CONTROLS_PER_EPISODE, replace=False))]
assert len(eligible) == ANCHOR_EPISODES + CONTROL_EPISODES, "the remaining inventory cannot supply the population"
assert len(anchors) == ANCHORS and len({a[0] for a in anchors}) == ANCHOR_EPISODES
assert not {a[0] for a in anchors} & set(control_eps)
print(json.dumps({"rule": "eligible = not used to fit model 0, and no recovery controller ever ran from the episode",
                  "excluded": excluded, "eligible": eligible, "eligible_sha256": digest,
                  "seed": "SeedSequence([171, 7, 0, int(eligible_sha256[:16], 16)]).permutation(eligible)",
                  "t_max": T_MAX, "anchor_episodes": anchor_eps, "anchors_per_episode": per,
                  "control_episodes": control_eps, "anchors": anchors, "controls": controls}, indent=1))
