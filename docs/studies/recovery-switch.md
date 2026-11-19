# A value-free switch for post-training recovery on Pendulum (protocol)

This is the protocol of [#79](https://github.com/personalrobotics/il-supervision-benchmarks/issues/79). It is
frozen before any study or control episode is evaluated; the split and start times were drawn before any wrapper
code ran.

**Question.** Can a value-free switching rule turn #76's learned receding-horizon recovery into a post-training
wrapper around the frozen BC: inactive on trusted behavior, taking authority when the state has departed and a
feasible plan exists, handing back when close and in agreement, and abstaining otherwise?
[#76](recovery-receding.md#result) recovered 182 of 400 starts with the recovery controller always on.

**Value-free.** The switch uses no reward, cost-to-go, expert query, learned value or new supervision. Task outcome
is reported descriptively and defines nothing.

## Fixed inputs (reused from #76, unchanged)

- The frozen BC (#73's seed-3321 policy), #73's model 0 (float64), the Pendulum evaluator map and refined flow.
- #76's receding-horizon controller: Stage 2's LM with amendment 1, 8 restarts on the first solve and 2 warm restarts
  (previous plan shifted, remaining demonstrated actions) on each replan, the executed action from the restart with
  the smallest predicted terminal miss, shrinking horizon to a fixed target, the ladder M ∈ {1, 2, 4, 8, 16, 32, 64},
  ε = 0.1 τ_demo, |u| ≤ 3, the |ω| envelope, and #76's audit (bounds, envelope, bitwise replay, endpoint, refinement).
- Stage 2's 400 Pendulum starts, each with its known source episode and anchor time t, read by provenance from the
  releases, and #76's per-start classification from `evidence-recovery-receding-v1`.

Each episode keeps its known source continuation c_p = (state of the source demonstration at time p). There is no
retrieval among demonstrations.

## Demonstration data

The Pendulum dataset has 100 episodes of 500 steps. Ten are #70's (8 train, 2 validation). The other 90 are
**held out**:
- **Calibration (45):** the 20 episodes inspected by the pre-protocol reference check below (ids 0–19), plus 25 of
  the remaining 70 drawn by `SeedSequence([171, 5, 0])`.
- **Controls (45):** the rest, untouched by anything before the evaluation.

Each held-out episode gives 4 distinct start times in [0, 435] from `SeedSequence([171, 5, 1, episode])`, so every
start has at least 64 continuation steps. The split and every start time were drawn from the seeds and episode ids
alone, before any wrapper code ran, and are frozen in [`tools/recovery_switch/split.json`](../../tools/recovery_switch/split.json).
Calibration fixes the action-agreement scale and threshold; controls are evaluated with them.

## The wrapper

An episode starts at x_0 with phase p_0 = t (the start's anchor time, or the control's start time), in mode `base`,
and runs to episode step 500, or until it abstains. At each step k, with state x_k:

- **Phase tracking (online, monotone).** In `base`, p_k is the index in [p_(k−1), p_(k−1) + 2] minimizing
  d(x_k, c_p); ties choose the smallest forward advance. In `recover`, the phase is timed by the segment:
  p = p_entry + j at the segment's step j. The tracked distance is D_k = d(x_k, c_(p_k)).
- **Entry (base → recover).** If D_k > δ_out, the wrapper looks for a segment: for M in the ladder, in order, with
  p_k + M ≤ 499, it runs #76's first solve (8 restarts, keyed `[171, 5, 2, episode, k, M, restart]`) through model 0
  toward c_(p_k + M). The segment is **feasible** if its best restart's predicted endpoint is within ε and its
  predicted states stay in the envelope. The shortest feasible M is executed as one #76 receding-horizon execution,
  reusing that first solve. If no M is feasible, the wrapper **abstains** (`no_feasible_plan`, or
  `continuation_end` if no M fits before step 499).
- **During a segment.** Each executed step j records the state x_j before the action, the recovery action u_j,
  the base's action π(x_j), close_j = (D_j ≤ δ_in) and agree_j = (|clip(π(x_j)) − u_j| / σ_u ≤ α). Agreement
  compares the bounded actions that would actually be executed (π clipped to ±3, as the task applies it), in physical
  units normalized by σ_u.
  A `solver` or `envelope` failure abstains with that category.
- **At a segment's end.** A segment is **declared** if it ran all M steps and its observed endpoint is within ε of
  c_(p_entry + M); otherwise the wrapper abstains (`endpoint_miss`).
  - **Handoff (recover → base)** if the declared segment's endpoint is close and the last H executed recovery steps
    of this recovery episode (which may span chained segments) were all close and agreeing.
  - **Otherwise the recovery continues:** a new segment is planned from the endpoint at its phase, by the entry
    rule's search, without requiring D > δ_out.
- **Budgets.** One recovery episode (entry to handoff) may use at most 127 exact steps (#76's per-start budget);
  exceeding it abstains (`timeout`). An episode may enter recovery at most 3 times; a fourth departure abstains
  (`reentry_limit`).
- **Abstention ends the episode's record.** The wrapper claims nothing after it, and never executes an unaudited
  fallback action.

**Hysteresis.** Entry needs D > δ_out = 2ε. Exit needs D ≤ δ_in = ε, plus H steps of agreement and a declared
segment. Every mode transition is recorded, as is any re-entry after a handoff.

## Tolerances

**The demonstrations, not the base policy's rollouts, define the trusted continuation.** The frozen BC is imperfect,
and its departures from the continuation are what the wrapper exists to catch, so they are not part of the nominal
distribution.
- **δ_in = ε and δ_out = 2ε**, both from #76's recovery contract and fixed before any outcome: a simple hysteresis
  band in the existing physical tolerance. A recovery that meets its endpoint contract is close enough to be
  considered for handoff; the five-step action agreement is the additional protection.
- **σ_u** is the population standard deviation of the demonstrated actions over the calibration episodes. **α** is
  the 95th percentile, over every calibration transition, of |clip(π(x)) − clip(u_demo)| / σ_u: the base's own
  disagreement with the demonstrator on demonstrated states.
- **H = 5** steps (0.1 s).
- **No separate model-error threshold.** "Trusted enough to try" means model 0 finds a feasible constrained plan.
  Execution uses observed states, and any endpoint miss abstains. A threshold calibrated on demonstration transitions
  would likely reject the recoveries #76 showed can work: their executed one-step errors exceed the validation
  error.

The calibration record (σ_u, α and their inputs) is written and checked by hash before any population runs.

## Populations

1. **Held-out demonstration starts (controls):** the 180 control starts, each on its demonstrated state. A switch
   later in such an episode is not automatically a false activation: if the base has actually departed, switching is
   correct. These episodes measure when the frozen BC departs, whether recovery catches it, whether it hands back,
   and how the trajectory compares with the BC alone.
2. **Recover population:** #76's 182 starts with a learned receding-horizon recovery.
3. **Unknown population:** #76's other 218 starts.

The comparator is #76's always-on learned arm, read from its record and not rerun.

## Gates

All of the following must hold, in every population:
1. **Base first.** Every episode begins under the base policy.
2. **Exact base reproduction.** Before the first frozen departure event (the first step with D > δ_out), the
   wrapper's actions are exactly, bitwise, the frozen BC's actions from the same start, and that event is where the
   wrapper first leaves `base`.
3. **Valid entries.** Every entry occurs at a step with D > δ_out and executes a feasible recovery plan.
4. **Valid handoffs.** Every handoff has D ≤ δ_in and the five-step close-and-agree window, on recomputation.
5. **Retention.** At least 80% of the recover population (≥ 146 of 182) are **retained**: the wrapper enters
   recovery at step 0, and that recovery episode ends in a handoff with every segment declared.
6. **Audited claims.** Every declared segment passes #76's audit.
7. **Complete accounting.** Every episode ends at step 500 in `base`, or with a recorded abstention category; no
   action is unaccounted for.

A negative result ends #79. It does not lead to threshold tuning, a target selector, online model updates or policy
training.

## Measures (by population and radius)

- Retention; abstention by category; first failure category.
- On the held-out demonstration starts: the step of the BC's first departure, whether recovery entered, whether it
  handed back, and the upright steps against the BC alone.
- Handoff time (steps from entry to handoff), recovery steps used, switch count and re-entries after a handoff.
- **Descriptive only:** upright steps to step 500 (#73's 8° criterion) of the switched episode against the frozen BC
  alone, for episodes that reach step 500.

## Smoke, then the run

The smoke uses Stage 2's key-99 smoke starts and 8 calibration-episode starts as controls, with tolerances from a
calibration that excludes those episodes. It checks a byte-identical rerun, a fresh-process audit, and faults the
audit must refuse for the right reason: a forged mode transition, a forged handoff (an agreement flag), a forged
declared segment, and a forged tolerance. It also projects the cost against a 3 h wall ceiling on siddh with 32
workers. The run waits for the smoke and its own go.

**Pre-protocol reference check** (no switching outcome was consulted). On 20 held-out episodes, replaying the
demonstrated actions reproduces the continuation to within 1e-13 ε, so a replay-only tolerance would be zero. The
frozen BC from a demonstrated state stayed within ε of its time-aligned continuation for 64 steps in 18 of 20 cases.
Those 20 episodes (ids 0–19) are assigned to calibration and never serve as controls.

## Revision before the study

The protocol frozen at `b45a46e` calibrated δ_out as the 99th percentile of the BC's maximum tracked distance over
full calibration rollouts. A development smoke of the wrapper, on calibration episodes and Stage 2's key-99 smoke
starts, showed that rule made the BC's own failures nominal: the BC leaves its continuation on most calibration runs
(median maximum 263 ε), so δ_out was about 265 ε and the switch could never enter. No study or control episode had
run. The revision above replaces that rule by δ_in = ε, δ_out = 2ε, and replaces the "controls stay with the BC"
gate by gates 1–4.
