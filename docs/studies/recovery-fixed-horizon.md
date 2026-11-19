# Fixed-horizon post-training recovery MPC on Pendulum (protocol)

This is the protocol of [#82](https://github.com/personalrobotics/il-supervision-benchmarks/issues/82). It and the
fresh start inventory are frozen before any fresh start is solved.

**Question.** Can a fixed, long receding-horizon controller recover most of the capability #76 found, without asking
the learned model to choose a recovery horizon? [#76](recovery-receding.md#result) recovered 182 of 400 starts when
seven horizons were executed independently and any success counted. [#79](recovery-switch.md#result) retained 78 of
those 182 when the wrapper chose the shortest horizon model 0 predicted feasible.

**H = 64 is outcome-informed.** It was chosen after #76 and #79 completed, so the gate is decided on fresh starts;
the #76/#79 starts are development evidence only (implementation checks and descriptive replay).

## Fixed inputs (unchanged)

- #76: the frozen BC, #73's model 0 in float64, Stage 2's LM with amendment 1, the bounds and envelope, ε, the
  ladder, the segment audit (bounds, envelope, bitwise replay, endpoint, refinement).
- #79: phase tracking, the entry rule (D > δ_out = 2ε and a feasible plan), the handoff rule (D ≤ δ_in = ε and five
  consecutive close-and-agree steps), hysteresis, the budgets (127 recovery steps per recovery episode, 3 entries),
  abstention, and its calibration (σ_u = 1.452, α = 0.0572, from
  [`evidence/recovery-switch/study/calibration.json`](evidence/recovery-switch/study/calibration.json)).

## Fresh starts

**Source.** The only demonstrations used neither to fit the dynamics model nor by #76 or #79 are #70's two
validation episodes, 80 and 81. They set τ in Stage 2 and gave #73's descriptive one-step validation error; no
recovery, switch or base outcome was ever computed on them.
- **Anchors.** The pool is every (episode, t) with t ≤ 435 in episodes 80 and 81 (872 anchors). 50 anchors are drawn
  by `SeedSequence([171, 6, 0])`, as #70 drew its 50.
- **Starts.** Each anchor gives #76's 8 starts (radii 0.5, 1, 2, 4 τ, two directions each), by Stage 2's perturbation
  and availability rule, keyed `[171, 6, 1, start]`: 400 fresh starts. Stage 2's M = 1 certificate is computed and
  recorded for each; the gate counts every start, as #76 did.
- **Controls.** 20 held-out demonstrated starts: 10 times per episode, drawn by `SeedSequence([171, 6, 2, episode])`
  from t ≤ 435 excluding the anchor times.

The inventory (anchors, start states, availability redraws, certificates and controls) is computed from the seeds and
the demonstrations alone, written to `tools/recovery_fixed_horizon/inventory.json`, and committed before any
controller runs.

## Arms (all on the 400 fresh starts)

1. **Any-horizon reference:** #76's learned arm, seven independent executions (one per M); a start is recovered if
   any execution is declared. It defines the capability available when the horizon is supplied externally.
2. **Shortest-feasible comparator:** #79's frozen wrapper.
3. **Fixed-H learned (primary):** the wrapper below, planning through model 0.
4. **Fixed-H exact (control ceiling):** the same wrapper planning through the evaluator map.

Arms 2–4 also run the 20 controls.

## The fixed-H wrapper

#79's wrapper with the horizon handling replaced; nothing else changes.
- **Entry.** At a base step with D > δ_out, solve H = min(64, 499 − p) from x toward c_(p+H) with #76's 8 restarts
  (keyed `[171, 6, 3, arm, unit, k, restart]`). Enter only if the best restart's predicted endpoint is within ε and
  its predicted states stay in the envelope; otherwise abstain (`no_feasible_plan`). No shorter horizon is tried.
- **Each recovery step** at phase p: execute the plan's first action, observe the real state, advance the phase by
  one, and solve again with H = min(64, 499 − p) toward c_(p+H) from 2 warm restarts: the previous plan shifted by
  one with the demonstrated action a_(p+H−1) appended, and the demonstrated actions a_p … a_(p+H−1). The executed action is the
  first action of the restart with the smallest predicted terminal miss; ties go to the earlier restart.
- **A valid plan** at every recovery step is finite with predicted states in the envelope. A non-finite plan
  abstains (`solver`); predicted states outside the envelope abstain (`plan_envelope`); an executed state outside
  the envelope abstains (`envelope`). The predicted endpoint is not required to stay within ε after entry.
- **Handoff** after any executed recovery step at which D ≤ ε and the last five recovery steps were close and
  agreeing (#79's rule, checked every step rather than at segment ends).
- **Budgets and the end.** More than 127 recovery steps in one recovery episode abstains (`timeout`); a fourth entry
  abstains (`reentry_limit`); if no target remains (p = 499), abstain (`continuation_end`).

**A declared recovery** is a recovery episode that ends in a handoff. Its audit is #76's segment audit applied to the
executed actions from the entry state to the handoff state, with target c_(p_entry + steps): bounds, envelope,
bitwise replay, endpoint within ε, and refined integration within ε.

## Gates (decided on the fresh starts only)

All must hold:
1. **Exact-H control:** at least 20 starts recovered, and at least 10% at one radius (#76's treatment gate). A start
   is recovered if the wrapper enters at step 0 and that recovery episode ends in an audited handoff.
2. **Fixed-H learned:** the same treatment gate.
3. **Retention:** fixed-H learned recovers at least 80% of the starts the any-horizon reference recovers.
4. **Exact base reproduction:** in every episode of every wrapper arm, every action before the first departure is
   bitwise the frozen BC's.
5. **Audits:** every entry, handoff and declared recovery passes on recomputation, and every abstention has its
   recorded category.
6. **Complete accounting:** every episode ends at step 500 in `base` or with an abstention; no unaudited action.

A negative result ends #82: no horizon sweep, tolerance relaxation, online model update or policy training.

## Measures

- Per start, paired: any-horizon (and its shortest declared M), shortest-feasible, fixed-H learned, fixed-H exact.
  Reported, never used to tune H, ε, the solver or the switch.
- By arm and radius: recoveries, retention, abstention by category, handoff time, re-entries, recovery steps.
- Controls: the BC's first departure, entries, handoffs.
- Descriptive only: upright steps to step 500 against the BC alone.

## Development replay, smoke, then the run

- **Development replay** (descriptive, not gated): the fixed-H learned wrapper on #76's 400 starts, after the
  inventory is frozen, as an implementation check.
- **Smoke:** Stage 2's key-99 smoke starts and #79's 8 smoke controls (calibration episodes 0 and 1), so the smoke
  touches no fresh episode; byte-identical rerun, fresh-process audit, injected faults (a forged base action, a forged
  agreement flag, a forged plan validity, a forged handoff), and a cost projection against a 3 h wall ceiling on
  siddh with 32 workers.
- The run waits for the smoke and its own go.
