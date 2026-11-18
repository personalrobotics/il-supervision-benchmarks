# Recovery-segment feasibility under exact dynamics (protocol)

This is the protocol of [#70](https://github.com/personalrobotics/il-supervision-benchmarks/issues/70), Stage 2 of
[CCIL #171](https://github.com/personalrobotics/CCIL/issues/171). It was frozen before any start below was solved. Its
machine-readable form is [`tools/recovery_feasibility/stage-2.json`](../../tools/recovery_feasibility/stage-2.json).

**Question.** Do valid recovery segments of M > 1 recorded action intervals exist from starts where no one-step
segment can recover? And does a solver declared in advance find them?

**Why.** One-step CCIL can only express corrections that a single action interval can make. #66 found that observed
PushT failures lay far from demonstrated support, and its local one-step solve could not recover them. That result
motivates the question but does not answer it. This stage asks it under exact or reference dynamics, before any
learned model or learner is involved.

**Not a rescue.**
- #21 asked whether segments help *BC's observed failures* on Pendulum. Those failures lie on the demonstrations' own
  trajectories, so that study's geometry was inadequate.
- This stage asks a different question: feasibility from perturbed starts. It never revisits #21's states or its
  learner, and no learner is trained here.

The segment, its residuals and its views are those of CCIL
[`docs/recovery-segments.md`](https://github.com/personalrobotics/CCIL/blob/12b2fbf/docs/recovery-segments.md).
M counts recorded action intervals.

## Problem

- **Forward problem.** A segment starts at a fixed, observed start x_0. It is a single-shooting rollout under the
  system's reference map, so the per-step dynamics defect is zero.
- **Target.** The target is the anchor's demonstrated continuation y_M = x_(t+M), in the same episode.
- **Valid segment.** A segment is valid if every one of these holds:
  - every action is within the action bounds;
  - every state is within the state envelope;
  - the endpoint residual d(x_M, y_M) is at most ε = 0.1 τ_demo, in the system's metric;
  - on PushT, the handoff residual is also at most ε ([below](#information-state-and-handoff));
  - the replay under the reference map reproduces the stored states exactly;
  - where it applies, the replay under refined integration also meets the endpoint and handoff tolerances;
  - on PushT, the segment also passes the independent continuation audit ([below](#pusht-continuation)).

**Targets constrain the controllable present and future, never observation history.** A segment cannot change
the frames before its start. The perturbation radius therefore uses only the current state. The record keeps every
chain state, so a learner's two-frame observation [x_(k−1), x_k] is available for each k ≥ 1. x_0's history frame is
the anchor's frame t − 1, moved by the same perturbation.

### Information state and handoff

Amendment 3 (CCIL #171, 2026-11-18). Recovery must restore the policy's information state, not one observation. A
policy that reads a window of W observations sees η_k = Ψ_W(y_(k−W+1) … y_k). After the M free actions, the segment
applies L = W − 1 fixed demonstrated handoff actions a_(t+M) … a_(t+M+L−1). These are not optimization variables.
The endpoint residual r_end = d(x_M, y_M) and the handoff residual r_handoff = d_η(η̂_(M+L), η_(t+M+L)) are separate
constraints: each must be within its tolerance, and a small one cannot compensate for a large one.

| | W | Ψ_W | L | endpoint metric, ε_end | handoff metric, ε_handoff |
|---|---|---|---|---|---|
| DI | 1 | the state | 0 | the system metric, 0.1 τ | none (r_handoff = r_end) |
| Pendulum | 1 | the state | 0 | the system metric, 0.1 τ | none (r_handoff = r_end) |
| PushT | 2 | the last two frames' 9-point configurations | 1 | the 9-point current-frame metric, 0.1 τ = 2.870 px | max(d(x_M, y_M), d(x_(M+1), y_(M+1))), 0.1 τ = 2.870 px |

- **DI and Pendulum are Markov.** W = 1 and L = 0, so their problems are unchanged.
- **PushT.** A segment must match two consecutive observations: its endpoint, and the observation after one
  demonstrated action. That flushes the stale pre-recovery frame without the simulator's hidden velocity or contact
  state. The handoff metric takes the window's largest per-frame distance, and its tolerance is ε, so no new
  threshold is introduced.
- **The optimizer.** PushT's CEM ranks samples by r_handoff, which bounds both residuals. DI's and Pendulum's
  objectives are unchanged.
- **M = 1 is ruled out** only if no M = 1 action passes both the endpoint and the handoff. On PushT the sweep steps
  each grid action, then the handoff action a_(t+1), and certifies infeasibility if no grid action has
  r_handoff ≤ 2ε. The endpoint-only minimum is recorded beside it. At M = 1 the construction is ordinary one-step
  CCIL. The handoff decides whether that construction restored the information state.
- **The continuation audit stays independent.** The 8-action continuation drift is not part of the optimizer, and it
  cannot repair a failed handoff.

**Starts.**
- **Anchors.** Each system has 50 anchors. Each is a demonstrated training state with t + max(ladder) inside its
  episode. On PushT, t ≥ 1, t + 16 + 8 must be inside the episode, and the anchor's row must be replay-verified
  (#52 B0). Anchors are drawn without replacement by `SeedSequence([171, 2, system])`.
- **Perturbations.** Each anchor is perturbed to radius r · τ_demo in the system's metric:
  - radii r ∈ {0.5, 1, 2, 4};
  - two seeded directions per radius;
  - so **400 starts per system**.
- **One availability rule for every system.** A draw is invalid if the perturbed start leaves the state envelope,
  or on PushT, if it penetrates or if bisection cannot reach the radius. An invalid draw is redrawn from the same
  seeded stream, at most 16 times. If all 17 draws are invalid, the start is **unavailable**: it is recorded and
  counted, never solved, and it enters no denominator. In particular, an unavailable start is never counted as
  M = 1-infeasible.
- **The smoke.** It uses its own 5 anchors per system, drawn with key 99 from the anchors the study does not use.

### PushT continuation

PushT's terminal reach compares only the current configuration, which can hide the wrong velocity or contact state.
Two checks reuse #52's replay and continuation machinery (`stage_a.replay`, and `oracle_probe.distance` within
Stage A's 1.0 px continuation tolerance):
- **Eligibility of an anchor–M pair.** The anchor's replay-reconstructed full state is restored fresh, and its
  recorded actions a_t … a_(t+M+7) are stepped in one live space. The pair is eligible only if every row
  t + 1 … t + M + 8 stays within 1.0 px of the recording. An ineligible pair is recorded and counted. It is never
  solved and enters no denominator. If (start, M = 1) is ineligible, the start is not in I.
- **Continuation audit of a segment** (amendment 2). In the same live space as the segment, the recorded actions
  a_(t+M) … a_(t+M+7) are stepped from x_M. With d_0 ≤ ε the segment's terminal miss and d_k the distance of the
  k-th resulting state to recorded row t + M + k, the segment passes only if max_k (d_k − d_0) ≤ 1.0 px. The 1.0 px
  tolerance measures additional drift: whether a velocity or contact mismatch makes the recovered trajectory diverge
  after rejoining. It does not demand a closer terminal match than ε.

The M = 1 certificate (no grid action within 2ε) concerns the terminal alone, so it remains a certificate that no
M = 1 segment is valid. DI and Pendulum have no eligibility gate. Their target is the full state, and the record
reports each pair's demonstrated-action replay miss d(F^M(x_t, a_t…), y_M).

## Systems

| | Double integrator (DI) | Pendulum | PushT |
|---|---|---|---|
| Role | solver calibration | treatment | treatment |
| Map | x′ = Ax + Bu, A = [[1, 0.1], [0, 1]], B = [0, 0.1] (CCIL quickstart) | `ilsb.tasks.pendulum` continuous evaluator map, dt 0.02 | gym_pusht 0.1 s action interval, physics dt 0.01 (LeRobot 0.6.1) |
| Demonstrations | quickstart `demos.py` (seed 0), 6 episodes × 40 steps; split by CCIL `make_split(·, 0.2, 42)` | `pendulum_cont_corrected.npz` (`59c332bc…`), r = 0.1 subset and split of design seed 3321 (#21's rules) | the r = 0.25 subset's 41 episodes (#66's support) |
| Metric | per-coordinate training-std-scaled Euclidean | √(wrap(Δe)² + (Δω/√g)²) | largest distance among the current frame's 9 points (agent and 8 T keypoints), px (`oracle_probe.distance`) |
| τ_demo | q95 of validation → nearest training state distance | the same rule (pendulum.md) | 28.703 px, refrozen on the current frame by #52 B0's rule ([`tau.json`](evidence/recovery-feasibility/tau.json)); B0's 29.764 px was on 18 points |
| Actions | \|u\| ≤ max\|u\| over the training demonstrations | \|u\| ≤ 3 (the task's) | agent target in [0, 512]² |
| State envelope | the training demonstrations' bounding box, widened by 50% of its range on each side | \|ω\| ≤ 1.5 · max\|ω\| over the training demonstrations | agent and T center in [0, 512]², with no agent–T penetration (the simulator's) |
| Perturbation | uniform direction in metric coordinates | uniform direction in (e, ω/√g) | agent (x, y) and T (x, y, θ) along a uniform direction, scaled by bisection to the radius. Velocities are kept from the anchor's restored state |
| Ladder M | {1, 2, 4, 8, 16} | {1, 2, 4, 8, 16, 32, 64} | {1, 2, 4, 8, 16} |
| Solver | LM shooting **and** CEM | LM shooting (a smooth, closed-form map) | CEM (query-only and nonsmooth; #66's local solve had zero gradient out of contact) |
| M = 1 infeasibility | **exact**: the bounded least-squares optimum (`scipy.optimize.lsq_linear`) exceeds ε | **Lipschitz-certified sweep** ([below](#pendulum-certificate)) | **sweep-certified, empirical**: no action on the 2 px grid (257² actions) ends within 2ε |
| Reachability oracle, M > 1 | **exact**: the same bounded least squares in U | none: a valid solver result is a constructive certificate | none |
| Integration refinement | not applicable (the map is exact) | replay with `solve_ivp` RK45 at rtol 1e-10, atol 1e-12 must end within ε | replay at physics dt 0.0025 (4× substeps) must end within ε |

The ladders come from each system's time scale. DI needs at least 2 intervals to correct both coordinates with one
input. Pendulum's 64 intervals are 1.28 s, about 4/√g. PushT's 16 intervals are the Diffusion Policy horizon.

### Pendulum certificate

The certificate covers u on a grid of 6,001 points over [−3, 3], so h = 0.001. It uses two bounds:
- **L_u**, the Lipschitz constant of the exact one-step flow in u. From the sensitivity equation and Grönwall,
  L_u ≤ dt · e^(L·dt), with L = √(1 + g²), in the metric (whose ω scaling only shrinks it).
- **δ**, the largest metric distance between the evaluator map and the refined flow over the grid.

A start is **M = 1-infeasible** if min over the grid of d_refined(F(x_0, u), y_1) − L_u · h / 2 − 2δ > ε. The factor
2 on δ covers the evaluator's adaptive step between grid points. The record states that this is a certificate under
those bounds, not a symbolic proof.

## Solvers (frozen)

- **LM shooting.**
  - The actions are parameterized as u = ū · tanh(v), with a finite-difference Jacobian (step 1e-6).
  - It runs at most 50 iterations, with damping starting at 1e-3, multiplied by 10 on a rejected step and divided by
    10 on an accepted one.
  - It makes 8 starts: zero actions, the demonstrated actions a_t … a_(t+M−1), and 6 seeded random ones.
- **CEM.**
  - It samples a population of 128 and keeps 13 elites, for 30 iterations.
  - The initial mean is the demonstrated actions. The initial standard deviation is 0.5 ū; for PushT it is 64 px.
  - Samples are clipped to bounds, and it makes 2 seeded restarts.
  - Each restart returns the better of the best sample over all iterations and the final mean, clipped to bounds; a
    tie keeps the sample (amendment 1).
- **Both.**
  - Every (start, M) is attempted, and the first valid segment is kept.
  - All restarts run, so that ambiguity is measured.
  - Seeds come from `SeedSequence([171, 2, system, start, M, restart])`.

## Measures (all reported)

- Unavailable starts and redraws by system × radius, and PushT's ineligible anchor–M pairs.
- The valid yield by system × radius × M, and the shortest valid M per start.
- Distance reach (radius) and temporal reach (M).
- Terminal-residual quantiles, the per-step defect (zero by construction, recorded), and constraint violations by
  type.
- Ambiguity: the number of distinct valid solutions across restarts (RMS action difference > 0.05 ū), and ties in
  terminal miss within 1e-9.
- For DI: each solver's recall against the oracle per M, and every returned segment the oracle or a validity check
  rejects.
- Integration-refinement deltas, where applicable.
- For every eligible (start, M) a solver leaves unsolved, the first failure of its best restart, in this order:
  endpoint, handoff, constraint (bounds, envelope, refinement), then continuation. On Pendulum and PushT, which have
  no oracle, an endpoint failure cannot be separated per pair into infeasibility and optimizer failure. Optimizer
  failure is measured on DI, by each solver's recall against the oracle.

## Decision

**DI, solver calibration.**
- Every returned segment must be valid.
- Each solver must recover ≥ 95% of the analytically reachable (start, M) cases with M > 1. Reachable means an oracle
  optimum ≤ 0.9 ε; the margin keeps tolerance-boundary cases out of the count.
- LM below 95%: Stage 2 stops with "solver inadequate".
- CEM alone below 95%: PushT's verdict is "solver inadequate", and Pendulum still decides.

Only available starts and, on PushT, eligible pairs enter any denominator, including DI's recall population.

**Pendulum and PushT, treatment.** Let I be the available starts certified M = 1-infeasible, and V ⊆ I those with
at least one valid segment at an eligible M > 1. A system passes if both hold:
- |V| ≥ 20;
- |V_r| / |I_r| ≥ 0.10 at at least one radius r ∈ {0.5, 1, 2, 4}.

Every counted segment must replay, satisfy every state and action constraint, meet the endpoint and handoff
tolerances, stay valid under refinement, and pass the continuation audit. On PushT, "certified M = 1-infeasible"
means no M = 1 action passes the endpoint and the handoff.

**Stage 2 passes** only if DI passes **and** at least one of Pendulum and PushT passes.
- DI alone never unlocks Stage 3.
- Stage 3 repeats only the systems that passed, with learned dynamics.
- If no treatment system passes, the refusal is the result, and #171 stops before Stage 3.
- No ε, radius, ladder, solver setting or threshold is changed after any study start is solved.

**PushT's certificate is empirical.** If PushT alone carries the pass, the result says so. Whether that suffices, or
a fallback system with a provable certificate is needed, is decided by the user before Stage 3.

## Smoke, then the run

**The smoke.** It runs the full pipeline on the smoke anchors (5 per system), with:
- a byte-identical rerun;
- a fresh-process audit that replays every valid segment and recomputes every certificate;
- injected faults that the audit must refuse for the right reason:
  - an action out of bounds;
  - a forged terminal state;
  - a flipped certificate;
  - a segment whose refined replay misses;
  - a PushT segment whose handoff residual is beyond ε;
  - a PushT segment whose continuation drifts more than 1.0 px;
- a measured cost projection. It must be within the ceiling before the run. It times the complete path for every
  system: availability, eligibility, the M = 1 certificate, every M of the ladder, every restart, and the replay,
  refinement and continuation checks. It does not time a truncated ladder or a single restart.

**Compute.** It is unpaid, and every system runs on siddh's CPU, under tmux. Each start's results are written
atomically and the run is resumable, with progress logged every 5 minutes.

| | work | measured unit cost | core-hours |
|---|---|---|---|
| PushT CEM | 400 · 2 restarts · 30 iterations · 128 · (2 + 3 + 5 + 9 + 17) ≈ 110.6M action-interval steps, with the handoff action, plus 15.4M fresh restores | 176 µs per step, 75 µs per restore (siddh) | ≈ 5.7 |
| PushT M = 1 sweep | 400 · 257² ≈ 26.4M evaluations (restore, the grid action, the handoff action) | ≈ 430 µs | ≈ 3.1 |
| Pendulum LM | 400 · 8 restarts · ≤ 50 iterations over the 7 rungs, with finite-difference columns batched | 150–600 µs per batched evaluator call (local) | ≈ 1 |
| Pendulum certificate | 400 · 6,001 refined one-step solves | 90 µs (local) | ≈ 0.1 |
| DI | ≈ 95M vectorized CEM map steps and LM shooting, all batched | negligible | < 0.1 |

That is about 11 core-hours. On 32 workers it is roughly 20 minutes of wall time. **The ceiling is 3 hours of wall
time for all three systems together.** If the smoke's projection exceeds it, the run does not start.

## Amendments before the study

All three were made after a smoke and before any study start was solved. Nothing else changed: not ε, the
radii, the ladder, the 95% DI gate, the terminal tolerance, action or state validity, refined replay, or the
treatment population gate.
1. **CEM returns its final mean when it is better.** The first smoke recalled 94.3% on DI, below the gate. CEM
   returned only its best sample, although the final mean is its estimate of the solution, and on the convex DI
   problem it may be better. This is the one solver correction. If CEM still misses 95%, it is recorded as
   inadequate, and it is not tuned further.
2. **PushT continuation measures drift.** The absolute 1.0 px rule conflicted with ε = 2.870 px: a segment that met
   its terminal contract failed continuation before taking a step. The rule is now max_k (d_k − d_0) ≤ 1.0 px. The
   anchor eligibility rule is unchanged: the anchor is the replayed state itself, so its d_0 is the replay's.
3. **Continuation-consistent recovery** ([above](#information-state-and-handoff)), from CCIL #171's Stage 2
   amendment. On the reference simulator, segments that matched PushT's endpoint diverged under the demonstrated
   continuation, so matching one observation does not restore a two-frame policy's input. PushT now optimizes and
   checks a one-action handoff, and M = 1 is ruled out against the endpoint and the handoff together. DI and
   Pendulum are unchanged. No tolerance was chosen from smoke or policy outcomes: the handoff uses ε.

## Code and smoke

The code is in [`tools/recovery_feasibility/`](../../tools/recovery_feasibility/):
- `run.py` solves one system's starts;
- `audit.py` is the fresh-process audit;
- `smoke.py` runs the smoke and combines the projections.

DI and Pendulum run under the ilsb environment with `PYTHONPATH=src`; PushT runs under the LeRobot environment. Each
file's docstring gives its command. The study set refuses to start without `--study-go`.

Implementation choices the protocol leaves open:
- **System ids.** The seed keys use DI 0, Pendulum 1 and PushT 2.
- **Restart streams.** LM's random restarts use restarts 2–7, and CEM's use 0–1. The smoke inserts 99 after the
  system id in every seed key.
- **One LM iteration** is one damped trial step. The Jacobian is recomputed after each accepted step.
- **PushT perturbation.** The perturbation is drawn in #52's scaled coordinates (x, y, L θ). A "returned" segment is
  one whose terminal miss is within ε; DI's "every returned segment valid" applies to those.
- **PushT envelope.** The start must not penetrate. Contact during a segment is the simulator's to resolve.

**Smoke result**, from `3cfb9b7` (the amended protocol) on siddh, 40 starts per system on the key-99 anchors
([`evidence/recovery-feasibility/smoke/`](evidence/recovery-feasibility/smoke/)):
- **Reproduction.** Each system's second run was byte-identical on every start file and on its record, and the
  fresh-process audit found nothing.
- **Faults.** Each injected fault was refused with its own reason first: action bounds, a forged terminal state, a
  flipped certificate, a refined-replay miss (Pendulum, PushT), and a continuation drift (PushT).
- **Projection.** The complete path takes 0.07 s per start for DI, 3.4 s for Pendulum and 87 s for PushT. That
  projects to 0.32 h of wall time for the study on 32 workers, within the 3 h ceiling (`projection.json`).

These smoke figures are not study outcomes. Three of them bear on the run:
- On the smoke's DI starts, CEM still recalled 100/106 oracle-reachable cases (94.3%) with amendment 1, below the
  95% gate. If the study repeats this, CEM is recorded as inadequate and PushT's verdict is "solver inadequate".
- On PushT, 37 segments ended within ε. Under the drift rule 3 are valid, all at r = 0.5 on starts whose M = 1 is
  not certified infeasible; 24 drift more than 1.0 px, and 10 drift and also miss under refined replay.
- Pendulum's 25 valid starts pass its treatment gate on the smoke anchors.
