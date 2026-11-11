# PushT keypoints v2: stopped before its freeze

**Status: not run.** The CCIL arm on PushT keypoints stopped at a solver-only
feasibility probe, before any CCIL policy trained. No v2 protocol was frozen.
Controlling issue: personalrobotics/CCIL#30.

## What v2 established

- **A competent learner.** IBC on the two-frame history state x̄_t =
  [x_{t−1}, x_t] passed its [feasibility gate](pusht-v2-ibc-gate.md): 0.8213
  coverage, 0.859 of the hold-to-demonstrator gap. Single-frame ACT ([pilot](pusht-m1-pilot.md))
  and velocity-aware ACT ([gate](pusht-v2-gate.md)) did not.
- **An exact history shift in CCIL's model.** CCIL#196 predicts the known
  shift exactly instead of learning it. On the held-out episodes its shift error
  is 0, against 2.58 px (p50) without the structure, and its newest-slot error
  is 1.44 px against 1.64 ([development evidence](evidence/pusht-v2/development/history-fit.json)).

## Why the CCIL arm stopped

`noisy_action_fe` must find a previous frame from which the perturbed action
reaches the recorded next state. The structured model makes the history
invariant exact, but the solve has no physically valid root on this state:

- **Yield.** The smoke run's labels stage converged 1,044 of 100,952 proposals,
  and 887 were exported ([`smoke-preimage`](evidence/pusht-v2/development/smoke-preimage/)).
  The failed solves end 36–50 px from their targets (median), and 1,000
  iterations rescue 35 of 2,000. ∂(newest next slot)/∂(previous frame) is
  near singular, with median condition number 9.3 × 10³.
- **The probe** ([spec](../../tools/pusht_v2/probe_spec.json), frozen before it
  ran; [result](evidence/pusht-v2/development/preimage-probe/probe.json)). It ran
  on 10,000 proposals, under the same equation, initialization, invariant and
  10⁻⁴ residual tolerance. A row is usable iff it converges, keeps the history
  invariant, keeps the T rigid in both slots (within 4.14 × 10⁻⁴ px, 10 × the
  demonstrations' largest deviation) and stays inside the action bounds.

| solver | converged | usable | rigidity violations among converged | T-shape error of converged rows (p50) | failed rows' residual (p50) |
|---|---|---|---|---|---|
| Newton, 50 iterations | 1.00% | 0% | 100 of 100 | 121 px | 65.7 px |
| Newton, 200 | 1.32% | 0% | 132 of 132 | 148 px | 65.6 px |
| damped minimum-norm, 50 | 0.05% | 0% | 5 of 5 | 41 px | 0.029 px |
| damped minimum-norm, 200 | 0.96% | 0% | 96 of 96 | 88 px | 0.019 px |

The history invariant held on every converged row. Two results ended the arm:
- **No valid root.** The rows that converge do so by deforming the previous
  frame's T by about 100 px. The smoke run's 887 labels were all impossible T
  shapes.
- **Damping does not fix it.** The damped solve turns divergence into
  approximate answers whose residuals plateau near 0.02 px, two orders above the
  tolerance.

The redundant keypoints (16 coordinates for 3 rigid degrees of freedom)
explain Newton's instability. They do not supply what is missing: an exact,
physically valid preimage of the generic learned keypoint model. The tolerance
was not weakened. A pose representation would change the learner's input and
need a new gate, so it was not substituted.

## What did not run

The three-arm comparison (bc, duplication, ccil; five seeds, about 32 GPU
hours) did not run. Its runner ([`study.py`](../../tools/pusht_v2/study.py)) and
draft protocol ([`protocol.json`](../../tools/pusht_v2/protocol.json)) passed a
smoke run end to end: arms, training, selection, evaluation, audit and report.
They are kept as they stand. The smoke run's history-invariant assertion and
audit passed, but its labels fail the rigidity check above.
