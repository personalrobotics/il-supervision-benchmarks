# Commit-and-extend recovery on Pendulum (protocol)

This is the protocol of [#85](https://github.com/personalrobotics/il-supervision-benchmarks/issues/85). It and the
fresh start inventory are frozen before any fresh start is solved.

**Question.** Can post-training recovery be posed as the first reliable return to a phase-consistent trusted
continuation, without choosing one global horizon or using task value? [#76](recovery-receding.md#result) recovered
starts when every horizon was executed independently and any success counted. [#79](recovery-switch.md#result)
committed to the shortest horizon model 0 predicted feasible and ended the recovery on a miss.
[#82](recovery-fixed-horizon.md#result) replanned toward a target always 64 steps ahead and retained 97 of the
reference's 174. The controller here keeps a target fixed while feedback supports it, and extends to a later target
only after observing that the current one was not reached.

**The design is informed by #82**, so the gates are decided on fresh starts; #82's starts are development evidence
and are not used.

## Fixed inputs (unchanged)

- #76: the frozen BC, #73's model 0 in float64, Stage 2's LM with amendment 1 and its restarts, the bounds and
  envelope, ε, the ladder {1, 2, 4, 8, 16, 32, 64}, the segment audit (bounds, envelope, bitwise replay, endpoint,
  refinement).
- #79: phase tracking, the entry rule (D > δ_out = 2ε), the handoff rule (D ≤ δ_in = ε and five consecutive
  close-and-agree recovery steps), the entry limit (3), and its calibration (σ_u = 1.452, α = 0.0572, from
  [`evidence/recovery-switch/study/calibration.json`](evidence/recovery-switch/study/calibration.json)).
- #82: the fixed-H wrapper, unchanged, as the comparator; the definition of a recovered start and of a declared
  recovery's audit.

Nothing is trained; no new model, solver, tolerance, uncertainty estimate or task value enters the controller.

## Fresh starts

**Eligibility.** As in #82: an episode is eligible if it was not used to fit model 0 and no recovery controller ever
ran from it. #82's 20 anchor episodes and 5 control episodes are now excluded too. From the committed records,
20 episodes remain: 4, 5, 7, 13, 15, 17, 18, 19, 21, 27, 28, 30, 36, 41, 46, 62, 65, 80, 81, 83.

**Selection** ([`tools/recovery_commit_extend/select.py`](../../tools/recovery_commit_extend/select.py), output
[`selection.json`](../../tools/recovery_commit_extend/selection.json)). #82's rule with population key 7: the
eligible list's sha256 seeds a permutation (`SeedSequence([171, 7, 0, int(sha256[:16], 16)])`); the first 15
episodes supply anchors and the last 5 are the control episodes, disjoint from them. The 50 anchors are spread as
evenly as possible: 4 in each of the first 5 episodes and 3 in each of the next 10, at distinct times drawn by
`SeedSequence([171, 7, 1, episode])`. Each control episode gives 4 times by `SeedSequence([171, 7, 2, episode])`.
Every anchor and control time is at most 308, as in #82, so the fixed-H comparator keeps H = 64 through its full
recovery budget. The selection reads no demonstration state.

**Starts.** Each anchor gives #76's 8 starts (radii 0.5, 1, 2, 4 τ, two directions each) by Stage 2's perturbation
and availability rule, keyed `[171, 7, 3, start]`: 400 fresh starts, with Stage 2's M = 1 certificate recorded. The 20
controls start on their demonstrated states. The inventory
([`inventory.json`](../../tools/recovery_commit_extend/inventory.json)) is computed from the selection, the seeds and
the demonstrations alone and committed before any controller runs. If it cannot supply 400 available starts and 20
controls, #85 stops before running; no earlier confirmatory population is reused.

## Arms (all on the 400 fresh starts)

1. **Any-horizon learned reference:** #76's learned arm, seven independent executions, one per M.
2. **Fixed-H learned comparator:** #82's wrapper, unchanged.
3. **Commit-and-extend exact:** the controller below, planning through the evaluator map.
4. **Commit-and-extend learned:** the same controller, planning through model 0.

Arms 2–4 also run the 20 controls. #79's shortest-feasible wrapper is not rerun.

## The commit-and-extend controller

#79's wrapper with the horizon handling replaced; entry, phase tracking, handoff and the entry limit are unchanged.

**Invariant.** Every executed recovery action belongs to a plan that currently predicts a valid arrival at its
committed target. Observed failure can only move the target later or cause abstention.

- **The authority test.** A plan toward target c_j passes if its actions are finite, its predicted states stay in the
  envelope, and its predicted endpoint is within ε of c_j. Every commitment and every replan is held to it.
- **Targets.** A recovery entered at phase p₀ has the ladder targets c_(p₀+M), M ∈ {1, 2, 4, 8, 16, 32, 64}, all
  relative to p₀. At phase p a target is unexpired if p₀ + M > p.
- **Commitment.** At a base step with D > δ_out, try the targets in increasing M: solve toward c_(p₀+M) with horizon M
  from #76's 8 restarts (keyed `[171, 7, 4, arm, unit, k, M, restart]`) and commit to the first that passes. If none
  passes, abstain (`no_feasible_plan`).
- **Countdown.** A commitment executes its plan's first action. At each later step committed to target j at phase
  p < j, replan toward the same c_j with horizon j − p from 2 warm restarts: the previous plan shifted by one, and the
  demonstrated actions a_p … a_(j−1). The best restart has the smallest predicted terminal miss; ties go to the
  earlier. If it passes, execute its first action, observe the real state and advance the phase by one. The target
  never moves while it is active.
- **Handoff** after any executed recovery step at which #79's rule holds: D ≤ ε and the last five recovery steps
  close and agreeing.
- **Target failure is local.** The current target fails, and its plan is not executed, if the replan is non-finite,
  leaves the predicted envelope, or predicts an endpoint miss above ε; it also fails when the phase reaches j without
  a handoff. The controller then tries the remaining unexpired targets in increasing M from the current state, each
  with horizon p₀ + M − p and 8 restarts (keyed as above), and commits to the first that passes. Each failure and
  its reason are recorded.
- **Abstention**, only when: no unexpired target passes (`targets_exhausted`); an observed state leaves the envelope
  (`envelope`); the demonstrated continuation ends, here an entry with p₀ + 64 > 499 (`insufficient_continuation`);
  the recovery exceeds #79's 127-step budget (`timeout`); a fourth entry (`reentry_limit`); or the episode ends during
  a recovery (`episode_end_in_recovery`). There is no unaudited fallback action. Because every target lies within
  64 steps of p₀, the budget and the episode end cannot bind within one recovery; they are kept as audited
  categories.

**A recovered start** enters at step 0 and that recovery ends in an audited handoff. **A declared recovery's audit**
is #82's: #76's segment audit of the executed actions from the entry state to the handoff state, with target
c_(p₀ + steps).

## Gates (on the fresh starts only)

**Stage A, exact prerequisite** (arm 3 on the starts and controls). All must hold:
1. at least 20 starts recovered, and at least 10% at one radius;
2. every action before the first departure is bitwise the frozen BC's;
3. every entry, commitment, replan, target failure, handoff and declared recovery passes on recomputation, and
   every abstention has its recorded category;
4. every episode ends at step 500 in `base` or with an abstention.

If Stage A fails, its result is published and #85 stops; the learned arms do not run.

**Stage B, learned result** (arms 1, 2 and 4 on the starts; 2 and 4 on the controls). All must hold:
1. commit-and-extend learned meets the treatment gate of A1;
2. it recovers at least 80% of the starts the any-horizon reference recovers;
3. it recovers strictly more starts than the fixed-H learned comparator;
4. base reproduction, audits and accounting as A2–A4, for arms 2 and 4.

A negative result at either stage ends #85: no ladder, tolerance, solver, model, target-rule or gate change.

## Measures

- Paired outcomes per start among the four arms; overlaps.
- By arm and radius: recoveries, the committed M at entry and at handoff, target failures per recovery by reason,
  abstention categories, steps to handoff, re-entries.
- Controls: the BC's first departure, entries, handoffs.
- **Recovery advantage** (descriptive, H3; it affects no selection or gate). #73's task measure, with no effort term:
  for each start and wrapper arm, Δ = upright steps under the wrapper (recovery, then the frozen BC) − upright steps
  under the frozen BC alone, from the same start through step 500, counted on the state before each action (#73's
  `score` convention). Secondary: #73's final two-second hold, for both branches. For evaluation only, an abstained
  episode is continued by the frozen BC from the abstention state; that continuation never counts as a recovery.
  Reported: the fraction with Δ ≥ 0 among recovered starts against failed or abstained ones.

## Smoke, then the runs

- **Smoke:** Stage 2's key-99 smoke starts and #79's 8 smoke controls (calibration episodes 0 and 1), all four arms,
  so it touches no fresh episode. It must exhibit each path at least once: commitment at M > 1, countdown, target
  failure by a failed authority test and by expiry, extension, handoff and abstention. It needs a byte-identical
  rerun, a fresh-process audit, and injected faults (a forged base action, a forged committed target, a forged
  countdown horizon, a forged authority test, a forged target failure, a forged handoff), each refused for its own
  reason; and a cost projection against a 3 h wall ceiling on siddh with 32 workers.
- **No development replay** on #82's starts.
- **Stage A run**, after the smoke, on its own go. **Stage B run**, only if Stage A passes, on its own go.
