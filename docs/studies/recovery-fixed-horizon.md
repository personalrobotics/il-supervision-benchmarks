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

**Eligibility.** "Fresh" is relative to the quantity tested: an episode is eligible if it was not used to fit model 0
and no recovery controller ever ran from it. BC-only use (#79's calibration of σ_u and α, the pre-protocol reference
check) does not reveal whether fixed-H recovery works and is allowed. From the committed records:
- excluded: the 8 episodes that fit model 0 (also every #70/#73/#76/#79 start's source), #79's 45 control episodes,
  and calibration episodes 0 and 1 (#79's smoke controls);
- eligible: the other 43 #79 calibration episodes and #70's validation episodes 80 and 81, 45 in all.

**Selection** ([`tools/recovery_fixed_horizon/select.py`](../../tools/recovery_fixed_horizon/select.py), output
[`selection.json`](../../tools/recovery_fixed_horizon/selection.json)). The eligible list's sha256 seeds a
permutation (`SeedSequence([171, 6, 0, int(sha256[:16], 16)])`); the first 20 episodes supply anchors and the next
5 are the control episodes, disjoint from them. The 50 anchors are spread as evenly as possible: 3 in each of the
first 10 episodes and 2 in each of the next 10, at distinct times drawn by `SeedSequence([171, 6, 1, episode])`.
Each control episode gives 4 times by `SeedSequence([171, 6, 2, episode])`. Every anchor and control time is at most
308 = 499 − (64 + 127), so the demonstration continues for H + the recovery budget after it. The selection reads no
demonstration state and was frozen before any perturbed start was generated.

**Starts.** Each anchor gives #76's 8 starts (radii 0.5, 1, 2, 4 τ, two directions each), by Stage 2's perturbation
and availability rule, keyed `[171, 6, 3, start]`: 400 fresh starts. Stage 2's M = 1 certificate is computed and
recorded for each; the gate counts every start, as #76 did. The 20 controls start on their demonstrated states. The
start inventory is computed from the selection, the seeds and the demonstrations alone, written to
`tools/recovery_fixed_horizon/inventory.json`, and committed before any controller runs.

## Arms (all on the 400 fresh starts)

1. **Any-horizon reference:** #76's learned arm, seven independent executions (one per M); a start is recovered if
   any execution is declared. It defines the capability available when the horizon is supplied externally.
2. **Shortest-feasible comparator:** #79's frozen wrapper.
3. **Fixed-H learned (primary):** the wrapper below, planning through model 0.
4. **Fixed-H exact (control ceiling):** the same wrapper planning through the evaluator map.

Arms 2–4 also run the 20 controls. Arm 2 keeps #79's own horizon search, unchanged.

## The fixed-H wrapper

#79's wrapper with the horizon handling replaced; nothing else changes.
- **Entry.** At a base step with D > δ_out, solve H = 64 from x toward c_(p+64) with #76's 8 restarts
  (keyed `[171, 6, 4, arm, unit, k, restart]`). Enter only if the best restart's predicted endpoint is within ε and
  its predicted states stay in the envelope; otherwise abstain (`no_feasible_plan`). No shorter horizon is tried.
- **H is fixed at 64 throughout.** It is never shortened. Each recovery step at phase p: execute the plan's first
  action, observe the real state, advance the phase by one, and solve again with H = 64 toward c_(p+64) from 2 warm
  restarts: the previous plan shifted by one with the demonstrated action a_(p+63) appended, and the demonstrated
  actions a_p … a_(p+63). The executed action is the
  first action of the restart with the smallest predicted terminal miss; ties go to the earlier restart.
- **A valid plan** at every recovery step is finite with predicted states in the envelope. A non-finite plan
  abstains (`solver`); predicted states outside the envelope abstain (`plan_envelope`); an executed state outside
  the envelope abstains (`envelope`). The predicted endpoint is not required to stay within ε after entry.
- **Handoff** after any executed recovery step at which D ≤ ε and the last five recovery steps were close and
  agreeing (#79's rule, checked every step rather than at segment ends).
- **Budgets and the end.** More than 127 recovery steps in one recovery episode abstains (`timeout`); a fourth entry
  abstains (`reentry_limit`). An entry or replan with p + 64 > 499 abstains (`insufficient_continuation`); with
  every anchor at t ≤ 308 this cannot happen within a recovery entered at step 0.
- **Entry establishes authority once.** The predicted endpoint is required within ε only at entry; replanning then
  supplies feedback. Requiring a fresh terminal certificate at every step would recreate #79's feasibility
  classifier.

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

- **Development replay** (descriptive, not gated): the fixed-H learned wrapper on #76's 400 starts. It runs only
  after the fresh inventory and the implementation commit are frozen, and its outcomes authorize no parameter or code
  change.
- **Smoke:** Stage 2's key-99 smoke starts and #79's 8 smoke controls (calibration episodes 0 and 1), so the smoke
  touches no fresh episode; byte-identical rerun, fresh-process audit, injected faults (a forged base action, a
  forged agreement flag, a forged plan validity, a forged handoff), and a cost projection against a 3 h wall ceiling
  on siddh with 32 workers.
- The run waits for the smoke and its own go.

## Code, smoke and development replay

The code is [`tools/recovery_fixed_horizon/`](../../tools/recovery_fixed_horizon/) (`fixedh.json` restates the frozen
choices and pins #79's calibration by hash). It builds on #79's code: the shortest-feasible arm is #79's wrapper
unchanged, the any-horizon arm is #76's segment execution per M, and the fixed-H wrapper is #79's switch with the
horizon handling above. Implementation choices the protocol leaves open:
- **Keys** are listed in `fixedh.json`; the fresh starts and controls are populations 2 and 3, distinct from every
  #76 and #79 stream.
- **The any-horizon reference** counts a start if some M's execution is declared and passes #76's audit.
- **An episode whose clock ends during a recovery** records `episode_end_in_recovery`; with every fresh anchor at
  t ≤ 308 it cannot happen within a recovery entered at step 0.
- **The audit** replays every arm from its stored plans (each fixed-H step's chosen plan is stored), recomputing
  states, phase, distances, base actions, plan validity, closeness, agreement, handoffs, #76's audit of each
  declared recovery, and the base reproduction.

**Smoke result**, from `2caba58` on siddh with CCIL `12b2fbf`, Stage 2's 40 key-99 starts and #79's 8 smoke controls,
all four arms ([`evidence/recovery-fixed-horizon/smoke.json`](evidence/recovery-fixed-horizon/smoke.json)):
- **Reproduction.** The second run was byte-identical on all 49 files. The fresh-process audit found nothing.
- **Faults.** Each injected fault was refused with its own reason first: a forged base action, a forged agreement
  flag, a forged plan validity, and a forged handoff.
- **Projection.** 0.92 h of wall time for the study on 32 workers, within the 3 h ceiling.

**Development replay** (descriptive; it authorizes no change), from `2caba58`: the fixed-H learned wrapper on #76's
400 starts ([`evidence/recovery-fixed-horizon/development/`](evidence/recovery-fixed-horizon/development/)). Its
audit found nothing. It recovered 108 starts, 93 of them among #76's 182 learned recoveries. These starts include
anchors later than t = 308, and 84 episodes ended in `insufficient_continuation`, so they are not comparable with the
fresh population.

These smoke and development figures are not study outcomes.

## Result

The study ran from `bdc6dd1` on siddh on 2026-11-19, with CCIL `12b2fbf`, on the 400 fresh starts and 20 fresh
controls, reading Stage 2, #73 and #76 from their releases. The fresh-process audit found nothing. The record and
audit are in [`evidence/recovery-fixed-horizon/study/`](evidence/recovery-fixed-horizon/study/); the complete run
directory is the asset of `evidence-recovery-fixed-horizon-v1`
([`archive.json`](evidence/recovery-fixed-horizon/archive.json)).

**#82 does not pass.** Both treatment gates and every mechanical check hold; retention fails:

| Gate | Required | Result |
|---|---|---|
| 1. Exact-H control | ≥ 20, ≥ 10% at one radius | 201 (76% at 0.5 τ) |
| 2. Fixed-H learned | ≥ 20, ≥ 10% at one radius | 117 (57% at 0.5 τ) |
| 3. Retention of the any-horizon reference's 174 | ≥ 140 | **97** (56%) |
| 4. Exact base reproduction before the first departure | all | 420 of 420 per wrapper arm |
| 5. Audits: entries, handoffs, declared recoveries | all | learned 455, 295, 295; exact 555, 485, 485 |
| 6. Complete accounting | all | 420 of 420 |

Recoveries by arm and radius 0.5, 1, 2 and 4 τ (100 starts each):

| Arm | 0.5 | 1 | 2 | 4 | All |
|---|---|---|---|---|---|
| Any-horizon reference | 68 | 64 | 31 | 11 | 174 |
| Shortest-feasible (#79) | 30 | 18 | 13 | 5 | 66 |
| Fixed-H learned | 57 | 32 | 18 | 10 | 117 |
| Fixed-H exact | 76 | 65 | 37 | 23 | 201 |

Retention by radius: 49 of 68, 27 of 64, 15 of 31, 6 of 11. Fixed-H learned also recovered 20 starts the reference
did not. Fixed-H exact recovered 146 of the reference's 174.

**Where the retention is lost.** Of the 77 reference recoveries fixed-H learned missed, 25 had no predicted H = 64
recovery at entry (`no_feasible_plan`) and 52 entered and reached the 127-step budget (`timeout`). In 47 of those 52
the executed state never came within ε of the continuation; the median closest approach was 4.0 ε. The misses
concentrate where the reference needed its longest horizon: 51 of the 77 were recovered by the reference only at
M = 64, against 27 of the 97 retained.

**Paired with #79's wrapper on the same starts:** fixed-H learned recovered 117 and shortest-feasible 66; 49 both,
68 fixed-H only, 17 shortest-feasible only. On the 20 controls, both fixed-H arms entered and handed back on all 20;
the shortest-feasible wrapper ended all 20 in `endpoint_miss`. The 64 learned and 109 exact
`insufficient_continuation` outcomes are later re-entries near the episode's end, not first recoveries.

**Descriptive.** The 30 learned and 67 exact episodes (starts and controls) that reached step 500 were upright a
median 128 and 111 steps more than the BC alone; too few complete for more than description.

**The conclusion.** A single fixed long horizon does not recover enough of the any-horizon capability. With the
horizon supplied externally and any success counted, the learned model recovers 174 of 400 fresh starts; a fixed
H = 64 receding-horizon controller through the same model, with #79's switch, retains 97 of them (56%), short of the
80% bar. The controller's mechanics held on every check, and through exact dynamics the same wrapper recovered 201,
so the shortfall lies in planning through model 0 at a fixed long horizon: most lost starts entered and then failed
to close on the continuation within the budget. Fixed-H learned did recover more than #79's shortest-feasible
wrapper (117 against 66). By the protocol, #82 ends here, with no horizon sweep, tolerance relaxation, online model
update or policy training, and no adaptive-horizon follow-up is launched from this result.
