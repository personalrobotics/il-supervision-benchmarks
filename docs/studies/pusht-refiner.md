# PushT gated residual refiner: Stage A refusal

This is the Stage A record of [#55](https://github.com/personalrobotics/il-supervision-benchmarks/issues/55). The
question was whether CCIL labels help more when they train a gated residual energy over a frozen r = 0.25 IBC base,
E = E₀ + g(x) ΔE, than when they are mixed into one shared IBC policy. Stage A had to define the gate g from training
artifacts alone before anything was trained.

## Result

**Stage A refuses. No refiner is trained, and the study stops.** The existing r = 0.25 pool does not define a
correction-specific domain in this state representation. Two gates were examined:
- an absolute gate around the CCIL labels covers the whole demonstration distribution;
- a relative gate that favours labels over demonstrations separates them only in sample. On episodes it was not built
  from, it scores demonstration states nearly as high as labels.

Nothing was trained or evaluated, no simulator ran, and no held-out or evaluation data was read. The record is
[`evidence/pusht-refiner/stage-a.json`](evidence/pusht-refiner/stage-a.json).

## Inputs

- **Pool:** the surface study's r = 0.25 CCIL pool (store entry `c30c0f98…`), labels `6024b739…`, candidates
  `6f8c9c97…`, selection `ed5beb46…`. It holds 19,982 accepted labels of 19,996 proposals, from the 41-episode subset
  (5,040 source rows, content `333a9b17…`).
- **Bases** (named, not loaded): the surface study's r = 0.25 BC units, seed 1 (`1c44a71a…`) and seed 2 (`8e098a53…`).

A state is its 18 points: the agent and the T's 8 keypoints, in the previous and the current frame. The distance between
two states is the largest distance between corresponding points, in px.

## Findings

**Evidence.** Every label carries weight 1.0. None carries a certificate or a finite error; the selection admits by
acceptance alone (as #40's inventory found). The gate's quality factor q is therefore constant.

**Geometry.** A label differs from its source row only in the previous frame; its current frame is the source's own (0 px
in every row).

| distance, px | 10% | 50% | 90% | 99% |
|---|---|---|---|---|
| label to its source row | 12.2 | 25.4 | 59.3 | 133.7 |
| label to the nearest subset demonstration row | 10.4 | 20.6 | 48.3 | 96.0 |
| subset demonstration row to the nearest label | 6.7 | 12.4 | 25.1 | 47.8 |

Every demonstration row lies within 71.4 px of a label. A gate around the labels wide enough to reach the duplication
targets, which are the labels' source rows, therefore covers the whole demonstration distribution.

**Relative gate.** g = d_real² / (d_real² + d_CCIL² + 10⁻⁶ px²), with d the distance to the nearest subset demonstration
row and to the nearest label. It is 0 on a demonstration row and about 1 on a label in sample. With each subset episode
held out against the other 40:

| g, held-out episode | 10% | 50% | 90% |
|---|---|---|---|
| demonstration rows | 0.33 | 0.49 | 0.52 |
| labels | 0.46 | 0.54 | 0.71 |

The rank AUC of labels over demonstration rows is 0.79. AUC is unchanged by any monotone reshaping of g, so a sharper gate
would not separate them better. At states a rollout reaches, which are not training rows, the refiner would act at about
half strength on demonstration-like states.

**Controls.** Both matched controls collapse to the base exactly:
- **Duplication:** every one of its 4,999 targets is a subset demonstration row, where the relative gate is exactly 0.
- **Identity**, a refiner that distils the base on the label states: a zero-initialized residual has zero gradient, so it
  stays zero.

## Decision and limits

The stop rule was set in #55's review before any refiner trained: stop if the relative gate cannot distinguish the two
populations. It was set after these separation numbers were computed, and no numeric threshold was pre-registered.

The finding is about this state representation and this pool. It is not a finding about residual refiners in general,
and it does not test whether the labels carry useful local corrections.

## Reproduce

On siddh, from a clean clone, with the LeRobot environment (NumPy 2.2.6; float64 on the CPU, about two minutes):

    $L -I tools/pusht_refiner/stage_a.py --source SOURCE.npz --labels LABELS.npz \
        --candidates CANDIDATES.npz --selection SELECTION.npz

The inputs are checked by the hashes above, and the script refuses a dirty tree. Two clean clones of `75d372d` wrote a
byte-identical record (sha256 `aac3dcfa…`).
