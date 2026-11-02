# Pendulum capture-region positive control: predeclaration, version 1 (#12)

**Status: version 1, pushed before any policy in this document is trained.** This study is **exploratory**. Its region was chosen after inspecting the rollouts of the pumping-region positive control (`docs/studies/pendulum-positive-control.md`, "Result"), so it cannot be described as confirmatory. The outcome is reported as it falls. An ambiguous outcome is "not established". Seeds are not added, the region is not changed, and no longer recovery segment is implemented.

## Result (version 1: run 20261102T003101Z from `c64b672`; exploratory)

**Decision: passes.** Both required contrasts hold for Relabel − Duplication. The evidence is in `docs/studies/evidence/pendulum-capture-control/`: `capture-control.json`, the run log, and the phase plot.

- **Selection:** all 10 seeds qualify, with 3,934–4,000 collection states each.
  - The capture check removed no LQR-branch state: π_D captures from every BC-visited state with |e| < 0.1.
  - Every label is within the bounds, and every labeled transition adds no manifold drift.
- **Required 1, held-out capture action error:** relabel 0.015 rad/s², BC 0.603, duplication 0.627.
  - Relabel − Duplication: −0.612, 95% interval [−1.050, −0.174], decreases.
- **Required 2, final hold:** relabel 1.00 on every seed; BC 0.40; duplication 0.40.
  - Relabel − Duplication: +0.602, [+0.234, +0.970], increases.
- **Secondary:**
  - Relabel − BC: action error −0.588 [−1.037, −0.150]; final hold +0.600 [+0.288, +0.913].
  - Upright occupancy: relabel 0.628, BC 0.417, duplication 0.396. Relabel − Duplication +0.232 [+0.098, +0.366]; Relabel − BC +0.210 [+0.091, +0.330].
  - Duplication does not move final hold relative to BC here (0.398 against 0.400).
- **Descriptive, from the phase plot (seed 3311):** inside the LQR branch the relabel arm's action field matches π_D. Just outside it (|e| from 0.1 to 0.5, |ω| up to about 2), the relabel arm disagrees with π_D more than BC does. The capture labels sharpen the branch interior and distort the field across its boundary. Rollouts still enter the branch and hold.
- **Naming (recorded after the run; the definition is unchanged and nothing is rerun):** the region is the **demonstrator LQR-branch capture region**: |e| < 0.1, intersected with the states from which π_D captures. It is not the oracle's verified LQR terminal set Ω_c = {xᵀPx ≤ c} (CCIL `docs/experiments/pendulum-oracle.md`).
  - Ω_c is an ellipse in (e, ω) from the DARE on the exact map, verified invariant under its own LQR gain.
  - This region is a band in e alone, defined by the demonstrator's switching rule and its fixed gains, and verified by rolling π_D out from each state.
  - The two are related but are not the same set, and neither contains the other by construction. Where this document says "capture region", it means the demonstrator LQR-branch capture region.
- **Conclusion:** for states visited by BC inside the demonstrator's LQR branch, corrected-demonstrator labels improve both held-out action accuracy and final capture, beyond exposure-matched duplication.
- **Scope:** exploratory, because the region was chosen after inspecting the pumping-region control.
  - It establishes that recovery supervision can matter on Pendulum.
  - It does not establish that CCIL generates those states or labels.
  - Per the decision rule, CCIL #171 may implement the data-only recovery-segment diagnostic. That diagnostic must cover this region's held-out capture states from demonstrations alone before any CCIL learner arm runs.

## Question

Does demonstrator supervision inside the demonstrator's local LQR capture region improve capture and final hold beyond an exposure-matched duplication control?

**This is not evidence for CCIL.** It is the last Pendulum positive control.
- **If it fails:** Pendulum stops as a positive CCIL showcase.
- **If it passes:** CCIL #171 may implement the data-only recovery-segment diagnostic. CCIL then receives demonstrations only, selects M without policy outcomes, and must show that its selected segments cover the capture region before any CCIL learner arm runs.

## Fixed setting (unchanged from the pumping-region control except where stated)

- **Task, data, subset, split and learner:** as in `docs/studies/pendulum-positive-control.md`, "Fixed setting":
  - the continuous corrected Pendulum, the canonical continuous dataset (sha256 `59c332bc…cf43c`) at r = 0.1;
  - the subset and the 8/2 split rule;
  - the study's learner block, unchanged, with each arm's seeds training as one group, and a fit that stops at the step ceiling instead of patience is a failure.
- **Seeds (new):** s ∈ {3311, …, 3320} (n = 10). They are fresh and exploratory, frozen here, and never increased.

## Start sets (new; pairwise disjoint, and disjoint from 2201–2204)

Each set holds 100 starts from `Task(eval_seed=…).starts()`.

| Set | eval_seed | Use |
|---|---|---|
| Collection | 2205 | the BC rollouts whose capture states are relabeled |
| Diagnostic | 2206 | the BC rollouts whose capture states form the held-out set |
| Final | 2207 | the closed-loop comparison |

Validation uses the seed's validation demonstration episodes, which share no rollout with any of these sets. The run refuses to proceed if any two of 2201–2207 share a start.

## The capture region and state selection (per seed, from that seed's frozen BC policy)

- **Capture region:** a state x is in it when both hold:
  - **it is in the demonstrator's LQR branch:** |e| < 0.1, where e = wrap(θ − π). This is the branch `ilsb.tasks.pendulum.expert` uses, with its gains fixed;
  - **the demonstrator captures from it:** π_D, rolled out for 500 steps through the exact map from x, ends with |e| ≤ 0.1 and |ω| ≤ 0.5 at each of its last 100 steps. This is the pumping-region control's recovery check, unchanged.

  Nothing about the region depends on demonstration support, on the arms, or on any outcome.
- **Visited states:** every frame of every BC rollout, 501 per rollout and 100 rollouts.
- **Per-episode thinning, deterministic:** in each rollout, take the frames in the capture region in time order. If there are more than 40, keep the 40 at indices round(linspace(0, n − 1, 40)). There is no random draw, so the budget is at most 4000 states, about the seed's training-demonstration row count.
- **Stop rule:**
  - a seed with fewer than 100 selected collection states has no arms;
  - if fewer than 8 of the 10 seeds qualify, the study stops: BC does not reach the capture region often enough.
- **The held-out set:** the same rule on the diagnostic rollouts.

## Labels and their checks

The label at a selected state x is π_D(x), clipped to ±3. Each set reports:
- **Action bounds:** every |π_D(x)| ≤ 3.
- **Exact transitions:** F(x, π_D(x)) is finite and adds at most 6.7e-7 manifold error (version 1.1's rule).
- **Capture:** it holds by selection. The fraction of LQR-branch states the capture check removed is reported.
- **Coverage:** the counts per seed and the e and ω quantiles of the selected states, plus the phase plot.

## Arms (per seed; identical initialization, learner, validation rule, real data and final starts)

1. **BC:** the demonstrations only. It is also the collection policy.
2. **Duplication control:** the demonstrations plus M(s) demonstration rows drawn with replacement by `default_rng(SeedSequence([s, 19]))`, where M(s) is the relabel arm's row count. The extra block trains as the synthetic block: half of each 512-row minibatch, with every extra row drawn before stopping is allowed.
3. **Relabel:** the demonstrations plus the M(s) capture states with their labels (x, π_D(x)), under the same mixing.

The duplication and relabel arms have the same row count and the same synthetic exposure.

## Outcomes

- **Required 1: held-out capture action error.** The mean over the seed's held-out capture states of |clip(π(x)) − π_D(x)|, in rad/s².
- **Required 2: final hold.** On the 100 final starts, the fraction of episodes within 8° of upright at every one of the last 100 steps, the last two seconds (`Task.score`, `final_hold`).
- **Secondary:**
  - upright occupancy;
  - all contrasts against BC;
  - a phase plot of each arm's action field around the LQR boundary (π − π_D on a grid over e ∈ [−0.5, 0.5] and ω ∈ [−3, 3]), with the final rollouts near upright. It is descriptive.

## Decision rule (fixed)

The decision rests on the paired per-seed differences, Relabel − Duplication, over the qualifying seeds. Each gets a 95% interval, the union of the BCa and t intervals (`ilsb.analysis.union_interval`).

The positive control **passes** only if both hold:
- the action-error interval lies below 0;
- the final-hold interval lies above 0.

Any other outcome, including a mixed or ambiguous one, is **"not established"**. Relabel − BC and the secondary outcomes are reported with the same intervals but do not enter the decision. No multiplicity adjustment applies, because both conditions must hold.

## Running and artifacts

- **Tool:** `tools/capture_control.py`, with `--resume`. It reuses `tools/positive_control.py`'s learner, training, supervision and geometry functions unchanged, and runs on siddh in tmux, with:
  - logs in `runs/`, headed by the command, commit, dirty state, host, load and versions;
  - progress at least every 5 minutes;
  - each stage's artifact written atomically;
  - a 30-minute stall threshold;
  - a watcher for completion, failure and stall.
- **Committed evidence:** `docs/studies/evidence/pendulum-capture-control/`:
  - `capture-control.json`: per-seed selections and label checks, per-arm outcomes, the intervals, the decision and the code state;
  - the run log;
  - the phase plot (`tools/capture_control_figure.py`, with `--check` for regeneration).

## Not rerun

The DP oracle or β grid, the σ scan, the pumping-region positive control, the 39-seed response surface, and the historical reward or demonstrator studies.
