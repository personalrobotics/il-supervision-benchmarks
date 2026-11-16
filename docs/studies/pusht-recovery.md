# PushT recovery windows for Diffusion Policy (protocol)

This is the protocol of [#52](https://github.com/personalrobotics/il-supervision-benchmarks/issues/52), Stage A. It is
frozen before any recovery window is built from the study's proposals. The machine-readable form is
[`tools/pusht_recovery/stage-a.json`](../../tools/pusht_recovery/stage-a.json).

**Question.** Does a one-step CCIL correction help a competent temporal learner when it is written as a physically
consistent recovery window, a history and an action sequence, rather than as an isolated state-action row?

**Status of the evidence.** This is an exploratory mechanism study. Its cell (r = 0.25, α = 0.25) and its oracle
proposals come from studies whose outcomes were seen ([pusht-oracle.md](pusht-oracle.md)), so no result here is
confirmatory evidence for CCIL.

## Stages

- **Stage A** (this protocol) builds the windows and audits them. No policy is trained. If the yield below is not
  met, Stage A refuses, the refusal is published, and the study stops. The window and replay rules are not relaxed
  after the result.
- **Stage B** runs only if Stage A passes, under its own protocol PR. It trains BC, source duplication and oracle
  CCIL with LeRobot's Diffusion Policy ([pusht-diffusion.md](pusht-diffusion.md)) on two development seeds.
- **Stage C**, learned-dynamics windows, runs only if Stage B passes.

D_test (seeds 5000–5099) is not read by any stage before the last.

## Inputs

- **Data:** `lerobot/pusht_keypoints` at `ace8c161`, as the PushT studies' source rows (content `333a9b17…`). Stage A
  reads the r = 0.25 development subset only: 41 training episodes from subset seed 0, 5,040 source rows.
- **Proposals:** #43's oracle Stage A run (content `b31c2201…`) over the store entry `c30c0f98…`'s 19,996 proposals
  (candidates `6f8c9c97…`). Its 12,743 accepted proposals are the population. Each carries a proposal action ã at a
  source frame t and the oracle's first-order decoded previous slot.
- **Simulator:** gym_pusht 0.1.8 with pymunk 6.11.1. Its states repeat bit for bit across processes; only the
  coverage reward can differ in its last digit, and Stage A does not read the reward.

## The recovery window

A window anchored at source frame t has Diffusion Policy's shape: two observations and 16 action slots, of which
slots 1–8 are executed.

- **Observations** [o′_{t−1}, o_t]. o_t is the source frame's own observation. o′_{t−1} is the agent position and T
  keypoints of a predecessor state P.
- **Actions.** Slot 0 is a preceding action b. Slot 1 is the correction ã. Slot k ≥ 2 is the demonstrated action of
  frame t + k − 1, or the episode's last action beyond its end, as LeRobot pads.
- **One simulated chain.** P is loaded into a fresh simulator, and then, in that same live space, b, ã and the
  demonstrated actions are stepped in turn. o′_{t−1} is the chain's first state, o_t is the recorded frame the chain
  reaches within 0.05 px, and every supervised action is one the chain applied. The oracle's decoded previous slot is
  not used as an observation; it is only where the solve starts.
- **Solve.** The unknowns are P's positions and velocities and b, 12 values in pixels. The residuals are the chain's
  positions after b against frame t and after ã against frame t + 1, 10 values. Gauss–Newton takes minimum-norm
  least-squares steps from P = (the decoded previous slot, the replayed source velocities at t − 1) and b = the
  demonstrated action of frame t − 1. The residual is rank-deficient by design: when nothing touches the T its
  velocity dies within one control step, so its position after b and after ã is the same equation. Converged means
  both distances are within 0.05 px, the oracle's tolerance.
- **Loss mask.** Slots 0 and 1 are supervised. Slot k ≥ 2 is supervised while the chain's positions at frames
  t + 1 … t + k − 1, where the demonstrated actions are applied, are each within 1.0 px of the recorded frames and the
  frame lies in the episode. The supervised slots are a prefix. Masked slots keep the demonstrated values, because the
  denoising network still reads them.
- **Provenance.** Each window records its episode, anchor frame and proposal, and each action slot records its source
  frame, or that it is the synthetic history action or the correction.

### Acceptance, in this order

| check | rule |
|---|---|
| predecessor | frame t − 1 is a source row |
| anchor | frame t is a LeRobot anchor (t ≤ T − 1 − 7) |
| solve | the iterate stays finite, the Jacobian is finite and non-zero, and Gauss–Newton converges within 20 iterations, with up to 8 halvings per step |
| finite | every chain state is finite |
| action bounds | every supervised action (b, ã and the supervised demonstrated actions) is inside [0, 512]² |
| b change | ‖b − a_{t−1}‖ ≤ the 99.9th percentile of the subset's consecutive action changes |
| speed | P's agent speed, T speed and ℓ\|ω\| each ≤ its maximum over the subset's replay-verified states |
| envelope | P and every chain state where a supervised action is applied lie within #43's penetration (22.98 px) and extent ([7.49, 523.06] px) envelope |
| continuation | the executed chunk, slots 1–8, is supervised |

One window is kept per source anchor: of its accepted proposals, the one picked by
`default_rng(SeedSequence([52, source row]))`.

## Yield and decision

Stage B is planned at r = 0.25 and α = 0.25, where α is the fraction of each epoch's training windows that are
synthetic. The subset has n_real = 4,794 real Diffusion Policy windows (`ilsb.temporal`), so Stage B needs
n_synthetic = round(α / (1 − α) · n_real) = 1,598 windows.

**Stage A passes when it keeps at least 1,598 windows, each from a distinct source anchor, from at least 30 of the
41 episodes. Otherwise it refuses and the study stops.**

Real windows supervise all 16 slots, padded ones included (LeRobot's default for Diffusion Policy); a recovery window
supervises only its mask's prefix. The record states both masses, so Stage B can match the duplication arm's
supervised mass to the recovery windows' exactly.

## Audit and record

- **Record:** `docs/studies/evidence/pusht-recovery/stage-a.json`. It holds the counts by reason, the selected windows
  per episode, the history, rejoin and continuation residuals, the supervised-slot distribution, the exact supervised
  loss mass against the real windows', novelty (o′_{t−1} against the source frame t − 1, against the oracle's decoded
  slot, and against the nearest real history), the limits, the input and output digests, and the commit.
- **Windows:** `windows.npz`, kept by content digest on siddh. Stage B consumes it.
- **Audit:** [`tools/pusht_recovery/audit.py`](../../tools/pusht_recovery/audit.py) runs in a fresh process. It
  recomputes the limits and the window counts from the inputs; replays every accepted proposal's chain and requires it
  bit for bit; recomputes its mask and every check; recomputes the selection; builds each selected window's slots
  independently (source frames from `ilsb.temporal`'s offsets, actions from the chain and the source rows); refuses
  any observation equal to a held-out (D_V) row; and recomputes the window digests and the decision. It does not
  re-solve rejected proposals; the byte-identical second run covers them.
- **Reproduction:** two clean runs of Stage A must write a byte-identical record.

## Smoke

[`tools/pusht_recovery/smoke.py`](../../tools/pusht_recovery/smoke.py) runs the construction on the first two training
episodes outside the subset, with proposals made the way #43's were. It says nothing about Stage A's yield. It
requires a byte-identical rerun, a clean fresh-process audit, and a refusal, for the right reason, of each injected
fault: an action out of bounds, a predecessor inside the T, a demonstrated slot from another episode, a held-out
observation, a wrong loss mask, a wrong source frame, a wrong selection, a wrong limit, a wrong window count and an
altered source. Its record is [`evidence/pusht-recovery/smoke.json`](evidence/pusht-recovery/smoke.json).

## Reproduce

```
L=$ILSB_LEROBOT_PYTHON   # gym_pusht 0.1.8, pymunk 6.11.1
$L -I tools/pusht_recovery/smoke.py --source SOURCE.npz --work WORK
$L -I tools/pusht_recovery/stage_a.py --source SOURCE.npz --candidates CANDIDATES.npz --proposals PROPOSALS.npz --run RUN
$L -I tools/pusht_recovery/audit.py --source SOURCE.npz --candidates CANDIDATES.npz --proposals PROPOSALS.npz --run RUN
```

SOURCE.npz is built by [`tools/pusht_v2/source.py`](../../tools/pusht_v2/source.py); CANDIDATES.npz and PROPOSALS.npz
are #43's, identified by the digests above.
