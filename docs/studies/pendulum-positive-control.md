# Pendulum missing-state positive control: predeclaration, version 1.1 (#12)

> **Historical record (2026-11-03).** The current account of the Pendulum investigation is [pendulum.md](pendulum.md). The text below is unchanged. The release asset named under "Running and artifacts" was not published; the run directory is retained on the run host, and `artifact-manifest.json` records the sha256 of its policies and state sets.

**Status: version 1.1 (a label-check amendment, below); version 1 was pushed before any policy in this document is trained.** The outcome is reported as it falls. If it is ambiguous, the result is "positive control not established". Replication is not increased.

## Result (version 1.1: run 20261101T230010Z from `b3f1114`)

**Decision: not established.** The relabel arm fits the demonstrator at held-out off-support states. This experiment did not establish a closed-loop improvement over BC or the duplication control: both occupancy intervals include zero and also admit a positive effect, up to +0.145 against BC and +0.050 against duplication. The evidence is in `docs/studies/evidence/pendulum-positive-control/`: `positive-control.json`, the run log and the phase plot.

- **Selection:** all 10 seeds qualify, with 1,250–4,000 relabeled collection states each.
  - The demonstrator recovers from every off-support state.
  - Every label is within the bounds, and every labeled transition adds no manifold drift. The selected states' own drift is at most 2.4e-7.
  - The states are mostly in pumping. Only seeds 3302, 3304 and 3310 contribute overshoot states.
- **Primary, held-out action error:** the relabel arm's mean is 0.021 rad/s², against 0.743 for BC and 0.740 for duplication.
  - Relabel − BC: −0.722, 95% interval [−1.412, −0.153], decreases.
  - Relabel − Duplication: −0.719, [−1.404, −0.134], decreases.
- **Closed loop, upright occupancy on the final starts:** the means are 0.271 for BC, 0.323 for duplication and 0.322 for relabel.
  - Relabel − BC: +0.051, [−0.022, +0.145], no detectable difference.
  - Relabel − Duplication: −0.001, [−0.053, +0.050], no detectable difference.
- **Secondary, final hold:** 0.077 for BC, 0.205 for duplication and 0.197 for relabel. The duplication control moves as far as the relabel arm.
- **Artifact binding (added after review):** `artifact-manifest.json` in the evidence directory holds the sha256 of every artifact this report depends on: subsets, splits, selected-state files, policies, final rollouts and evaluation records.
  - It was produced by `tools/control_verify.py` (run 20261102T015616Z, from `2c5d5bc`), which recomputes from those bytes, without writing to the run:
    - the selection, from the BC policies;
    - the final rollouts, action errors and scores, from the policies;
    - the intervals and the decision.
  - Every recomputed value equals the report exactly. The log is `verify-20261102T015616Z.log`.
- **Conclusion:** this experiment did not establish that missing-state supervision at BC-visited off-support states improves closed-loop behavior on this task, under the current learner and task distribution. It does not show the absence of an effect: the intervals admit a positive occupancy effect.
- **Stop rule:** the predeclared stop rule requires an established positive control, so this result does not justify the multi-step CCIL follow-up.
- **Not tested here (exploratory observation, not a finding):**
  - The selection excluded the LQR window (|e| < 0.1). In the phase plot, the relabel arm's rollouts reach the approach to upright and pass through it without capture.
  - The duplication control gains as much as the relabel arm over BC. That suggests the 50/50 mixed-batch training procedure, not the relabels, carries the change.
  - Both would need their own predeclared study.

## Question

Does supervising a learner at the states it actually reaches off the demonstrations improve it on the Pendulum, under the current learner and task distribution? The test is a one-shot, DAgger-style experiment:
- collect states from the frozen BC policy's own rollouts;
- label them with the corrected demonstrator;
- retrain;
- compare against BC and an exposure-matched duplicated-demonstration control.

**This is not evidence for CCIL.** It tests whether the missing-state-supervision mechanism exists on this task.
- **If it fails:** the Pendulum is not suitable for studying that mechanism under the current learner and task distribution.
- **If it succeeds:** a separately predeclared multi-step backtracking generator becomes justified. Gate 3A showed that one-step CCIL cannot leave demonstration support here (CCIL `docs/experiments/pendulum-correction.md`, "Result").

**Gating for that generator.** It would be gated on its realized valid displacement and label trust. A linear reachability bound, such as about 0.38 after ten steps, is only a loose scale.

## Fixed setting

- **Task:** the continuous corrected Pendulum, with |u| ≤ 3, dt = 0.02 and horizon 500. It runs through `ilsb.tasks.pendulum.Task` and the exact `pendulum_map`.
- **Data:** the canonical continuous dataset (sha256 `59c332bc…cf43c`), at r = 0.1 only. That condition has the sparsest coverage, and it was chosen for that reason, not for any outcome.
- **Subset and split:** the study's rules. The subset is the first 10 episodes of `default_rng(SeedSequence([s, 9])).permutation(100)`. `make_split(ids, 0.2, s)` then gives 8 training and 2 validation episodes.
- **Seeds:** s ∈ {3301, …, 3310} (n = 10), fresh and non-study. Each seed fixes the subset, the split, the network initialization and the shuffle, as in the study.
- **Learner:** the study's learner block, unchanged (`studies/pendulum-j-r-k/study.yaml`):
  - `vector_bc_v2` with a 256-256 MLP, Adam at 1e-3 and batch 512;
  - checkpoint selection on the validation episodes' demonstration loss, with the plateau schedule and patience;
  - a ceiling of 260,000 steps;
  - `deterministic=True` on CUDA.
  - Each arm's 10 seeds train as one group.

## Start sets (all from `Task`'s start distribution; pairwise disjoint)

Each set holds 100 starts, drawn uniformly with θ within ±1 rad of hanging and θ̇ within ±0.5 rad/s, as `Task.starts` draws them, under its own seed:

| Set | eval_seed | Use |
|---|---|---|
| Collection | 2202 | the BC rollouts whose states are relabeled |
| Diagnostic | 2203 | the BC rollouts whose states form the held-out recovery set |
| Final | 2204 | the closed-loop comparison (fresh: not the study's 2201) |

The run refuses to proceed if any two sets share a start.

## State selection (per seed, from that seed's frozen BC policy)

- **Coverage threshold:** τ_demo(s) is the 95th percentile of d_demo over the seed's validation-episode demonstration states. It is the rule Gate 3A used, unchanged.
- **Metric:** d_demo is the distance to the nearest training-demonstration state. The physical metric is d = √(wrap(Δe)² + (Δω/√g)²), periodic in e.
- **Selected states:** every state of the BC rollouts, all 501 frames of each, that meets all of these:
  - it is off support: d_demo ≥ τ_demo(s);
  - it is outside the demonstrator's LQR window: |e| ≥ 0.1;
  - **the demonstrator recovers:** π_D, rolled out for 500 steps through the exact map from that state, ends with |e| ≤ 0.1 and |ω| ≤ 0.5 at each of its last 100 steps.

  These states are physically reachable, because the BC policy reached them through the exact map.
- **Budget:** M(s) = min(4000, available). That is about the seed's 3,992 training demonstration rows. When more are available, M is drawn without replacement by `default_rng(SeedSequence([s, 13]))`.
- **Stop rule:**
  - a seed with fewer than 100 selected collection states has no positive-control arms;
  - if fewer than 8 of the 10 seeds qualify, the study stops: BC does not produce enough off-support recovery states.
- **The diagnostic set:** the same selection on the diagnostic rollouts, capped at 4000 by `default_rng(SeedSequence([s, 17]))`.

## Labels and their checks

The label at a selected state x is π_D(x), the corrected demonstrator (`ilsb.tasks.pendulum.expert`, clipped to ±3). Each relabel set reports:
- **Action bounds:** every |π_D(x)| ≤ 3.
- **Exact transitions:** F(x, π_D(x)) through the exact map is finite, and the transition adds at most 6.7e-7 to the manifold error |sin² + cos² − 1|. The largest manifold error among the selected states is reported too.
  - *Amended in version 1.1, before any official run.* The exact map carries sin and cos separately. In the off-protocol smoke test, one BC policy spun at up to |ω| = 9.45, and over 500 steps its visited states drifted up to 1.8e-6. That is above the bound calibrated on the dataset (6.7e-7), while the labeled transitions added no drift. An absolute bound would have reported the BC rollouts' own drift as a label failure.
- **Recovery:** it holds by selection; the fraction of off-support states it removed is reported.
- **Coverage:**
  - counts by region: pumping (H < 0) and overshoot (H ≥ 0), where H = ½ω² − g(1 − cos e);
  - the d_demo distribution;
  - the phase plot.

## Arms (per seed; identical initialization, learner, validation rule, real data and final starts)

1. **BC:** the study's `bc` arm, which is also the collection policy.
2. **Duplication control:** the demonstrations, plus M(s) demonstration rows of the training episodes as an extra block.
   - The extra rows are drawn with replacement by `default_rng(SeedSequence([s, 19]))`, so the block has the same row count as the relabel arm's.
   - It trains under the same mixing as the relabel arm: half of each 512-row minibatch from the demonstrations and half from the extra block (synthetic_fraction 0.5), with every extra row drawn before stopping is allowed.
3. **Relabel:** the demonstrations, plus the M(s) relabeled collection states (x, π_D(x)) as the extra block, under that same mixing.

## Outcomes

- **Primary: held-out recovery-state action error.** The mean over the seed's diagnostic set of |clip(π(x)) − π_D(x)|, in rad/s².
- **Closed loop:** upright occupancy on the final starts, from the study's canonical scorer (`Task.score`). Final hold is reported as secondary.
- **Phase plot:** for each arm, the final rollouts on the oracle's axis contract (e ∈ [−π, π) periodic, ω ∈ [−8, 8]). It shows the demonstrations, the relabeled states, and per-start occupancy, so it shows where behavior changes. It is descriptive.

## Decision rule (fixed)

For each contrast, Relabel − BC and Relabel − Duplication, take the paired per-seed differences over the qualifying seeds. Each gets a 95% interval: the union of the BCa and t intervals, `ilsb.analysis.union_interval`.

The positive control is **established** only if all four hold:
- the action error is lower than BC's (interval below 0);
- the action error is lower than the Duplication control's (interval below 0);
- occupancy is higher than BC's (interval above 0);
- occupancy is higher than the Duplication control's (interval above 0).

Because every one of the four must pass, no multiplicity adjustment applies. Any other outcome, including mixed or ambiguous ones, is reported as **"not established"**. The seed count is not increased.

**Stop rule for follow-up work:** a multi-step CCIL generator is considered only if the positive control is established.

## Running and artifacts

- **Tool:** `tools/positive_control.py`, with `--resume`. It runs on siddh in tmux, with:
  - logs in `runs/`;
  - progress at least every 5 minutes;
  - each stage's artifact written atomically (BC policies, selected states, the two arms' policies, rollouts);
  - a 30-minute stall threshold;
  - a watcher.
- **Committed evidence:** `docs/studies/evidence/pendulum-positive-control/`:
  - `positive-control.json`: per-seed selections, label checks, per-arm outcomes, the four intervals, the decision and the code state;
  - the run log;
  - the phase plot.
- **Release asset:** the policies and the per-seed state sets. Their sha256 are in `artifact-manifest.json`, which was added after review.
