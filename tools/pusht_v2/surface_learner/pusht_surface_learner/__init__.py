"""The PushT surface study's IBC learner (ilsb#5), run in a LeRobot 0.6.1 environment with the two plugins.

    python -m pusht_surface_learner train UNIT_DIR      one unit's whole life: 20k, then 40k and 80k while selected
    python -m pusht_surface_learner infonce UNIT_DIR CHECKPOINT OUT_JSON   re-score a saved checkpoint on D_V

The training step is lerobot-train's own (lerobot.scripts.lerobot_train.update_policy, the policy, processors,
optimizer, scheduler, precision and checkpoint format all built by LeRobot's factories); only the minibatch source is
this package's: every batch holds exactly b_s = round(alpha * B) synthetic rows and B - b_s real rows (sampler.py).
"""
