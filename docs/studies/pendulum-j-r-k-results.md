# J(r, K | α = 0.5) on Pendulum: results

The protocol is `docs/studies/pendulum-j-r-k.md`. This document reports one run of it and interprets it. The evidence is in `docs/studies/evidence/pendulum-j-r-k/study/`.

**Summary.**
- None of the six primary contrasts shows a detectable difference. CCIL with learned dynamics does not differ detectably from BC at any real fraction, and exact dynamics does not differ detectably from learned dynamics.
- The derived contrast exact − BC at r = 100% increases by +0.020 [+0.002, +0.045]. It is not a primary claim.
- The label diagnostics show what was tested. At the supervised-validation-selected σ, every synthetic state lies within 0.013 standard deviations of a training demonstration state, and every label action within 0.145 N·m of its source.
- So the study measures near-duplicate supervision with small action noise. It does not test genuinely corrective labels, and the null results say nothing about them.

## Execution

| | |
|---|---|
| Root | `runs/20261031T141026Z-j-r-k` on siddh, started 14:10:26 UTC on 2026-10-31. Wall time 101 min 52 s. |
| Governing commit | `5647117`, the merge commit of the protocol PR (#14), clean checkout (`run.log`, `root.json`). The earlier run `runs/20261031T134157Z-j-r-k` was aborted before any outcome was read and is excluded (protocol, "Run history"). |
| Code | ccil `cc711b5`, `uv.lock` `63ccc463…c136`, torch 2.14.1+cu130, numpy 2.5.4, Python 3.12.15, CUDA on an RTX PRO 6000 Blackwell. |
| Data | `pendulum_cont_corrected.npz`, SHA-256 `59c332bc…f43c`, checked by preflight. |
| Units | 471 planned and 471 trained. None was reused from another run. |
| Stopping | Every unit stopped by the plateau rule. None reached the 260,000-step ceiling; the latest stop was at step 160,836. |
| Feasibility | No candidate failed at the fixed σ under either dynamics, in any unit. |
| Exposure | The realized synthetic row fraction was 0.5 in every CCIL unit. |
| Audits | `audit` and `evaluation_audit` both pass on 471 of 471 units, with no problems listed. |
| Stage hashes | `manifest.json` lists the SHA-256 of each stage file; the copies in `stages/` match it. |

## Primary contrasts

The metric is upright occupancy: the fraction of steps within 8° of upright, averaged per seed over 100 fixed final-test starts. Each contrast is paired by seed over 39 seeds at K = 20. The interval is the union of BCa and paired t intervals at Bonferroni confidence 1 − 0.05/6 (`stages/report.json`, `contrasts`).

| Contrast | r | Mean | 99.17% interval | Outcome |
|---|---|---|---|---|
| learned − BC | 10% | −0.007 | [−0.039, +0.030] | no detectable difference |
| learned − BC | 25% | −0.013 | [−0.038, +0.011] | no detectable difference |
| learned − BC | 100% | +0.014 | [−0.001, +0.040] | no detectable difference |
| exact − learned | 10% | −0.011 | [−0.043, +0.018] | no detectable difference |
| exact − learned | 25% | +0.005 | [−0.016, +0.030] | no detectable difference |
| exact − learned | 100% | +0.006 | [−0.003, +0.024] | no detectable difference |

A "no detectable difference" result is not a claim that the effect is absent. Each interval bounds the effect only at this σ. Every primary interval lies within [−0.043, +0.040] occupancy.

## Derived contrasts

These are reported but are not a third primary family (`stages/report.json`, `derived`). The report's label "increases" is the protocol's "improves": the lower endpoint is above 0.

| Contrast | r | Mean | 99.17% interval | Outcome |
|---|---|---|---|---|
| exact − BC | 10% | −0.018 | [−0.055, +0.020] | no detectable difference |
| exact − BC | 25% | −0.008 | [−0.034, +0.018] | no detectable difference |
| exact − BC | 100% | +0.020 | [+0.002, +0.045] | increases |

## Response surface (descriptive)

These are the mean upright occupancies over each cell's own seeds: 39 at K = 0 and K = 20, and 10 at K = 1 and K = 4. No test is made on them. Per-seed values are in `stages/report.json`, under `cells`.

| r | BC, K = 0 | learned, K = 1 | learned, K = 4 | learned, K = 20 | exact, K = 1 | exact, K = 4 | exact, K = 20 |
|---|---|---|---|---|---|---|---|
| 10% | 0.346 | 0.295 | 0.325 | 0.339 | 0.314 | 0.304 | 0.329 |
| 25% | 0.493 | 0.470 | 0.478 | 0.480 | 0.481 | 0.498 | 0.485 |
| 100% | 0.588 | 0.597 | 0.609 | 0.603 | 0.607 | 0.609 | 0.609 |

Real data dominates the surface: going from r = 10% to r = 100% moves every arm by about 0.25. Across arms and K, the cells differ as follows:
- At r = 10% and 25%, by at most 0.051. The standard errors of the 10-seed cells are 0.03–0.045 there.
- At r = 100%, every CCIL cell lies 0.009–0.021 above BC's 0.588, and the standard errors are 0.001–0.012.

The best seed of every arm at r = 100% scores 0.613, so the effect of K there is bounded by a shared ceiling.

## Label diagnostics

The label diagnostics were frozen before any outcome was read: protocol "Label diagnostics", PR #15 at `6cbc8db`. They were computed from that commit in a separate checkout, against the study's own store, which they left unchanged (`label-diagnostics.json`, `label-diagnostics.log`). How to read the table:
- Displacements are from each synthetic row's source demonstration. State quantities are standardized by the training demonstrations' per-coordinate standard deviation.
- Medians are taken over rows, then over seeds; maxima over rows and seeds.
- The first attempt refused before reading anything, because its checkout had no dataset file (`label-diagnostics-refused.log`).

| r | K | σ | Action displacement, median (N·m) | Learned state displacement, median | Exact rows with zero state displacement | Nearest demonstration, max | Exact−learned state gap, median |
|---|---|---|---|---|---|---|---|
| 10% | 1 | 1e-5 | 2.0e-5 | 2.1e-4 | 1.00 | 1.1e-2 | 2.1e-4 |
| 10% | 4 | 1e-5 | 2.0e-5 | 2.1e-4 | 1.00 | 1.1e-2 | 2.1e-4 |
| 10% | 20 | 1e-3 | 2.0e-3 | 1.7e-4 | 0.90 | 1.3e-2 | 1.7e-4 |
| 25% | 1 | 1e-3 | 2.0e-3 | 2.2e-4 | 0.90 | 8.4e-3 | 2.1e-4 |
| 25% | 4 | 1e-5 | 2.0e-5 | 2.1e-4 | 1.00 | 8.4e-3 | 2.1e-4 |
| 25% | 20 | 1e-5 | 2.0e-5 | 1.3e-4 | 1.00 | 1.1e-2 | 1.3e-4 |
| 100% | 1 | 1e-3 | 2.0e-3 | 1.5e-4 | 0.90 | 5.6e-3 | 1.4e-4 |
| 100% | 4 | 1e-2 | 2.0e-2 | 3.4e-4 | 0.13 | 5.5e-3 | 1.8e-4 |
| 100% | 20 | 1e-4 | 2.0e-4 | 1.4e-4 | 1.00 | 1.0e-2 | 1.4e-4 |

**Accepted labels.** Both arms accepted every proposal, and their proposals were identical row for row.

**Exact arm: action noise, not correction.** At σ ≤ 1e-4, every exact-arm synthetic state equals its source state. At σ = 1e-3, 90% of them do.
- **Why:** the Newton preimage starts from the source state and stops once the residual is within its 1e-4 tolerance (`ccil/core/generators.py`, `newton_preimage`). A perturbation this small leaves the source state within tolerance, so the solver returns it unchanged.
- **Effect:** at these σ the exact arm trains on the demonstration states, labeled with the demonstrated action plus noise of about 2σ N·m (median).
- **Exception:** only at r = 100%, K = 4 (σ = 1e-2) do most exact rows move, by a median of 2.2e-4 standard deviations.

**Learned arm: model error, not correction.** The learned arm's states move by about 1e-4 standard deviations. That is the same size as the gap between the two arms' states, which isolates dynamics-model error. So the learned arm's displacement is mostly the solver compensating for model error, not correcting toward the demonstration.

**Every synthetic example stays near the data.**
- Every synthetic state, in every unit, is within 0.013 standard deviations of a training demonstration state.
- Every label action is within 0.145 N·m (4.8% of the bound) of its source.

**Correction to a frozen metric.** The frozen "distinct novel examples" counts are an artifact and should not be read as novelty.
- **Mechanism:** many demonstration actions sit exactly at the ±3 N·m bound, where a/bound = ±1 falls on a δ-grid edge. A label nudged to 2.99999 then lands in the neighboring cell.
- **Evidence:** in one probe unit (exact arm, r = 10%, K = 1, seed 1301), all 822 rows whose cell differed from their source's cell differed only in the action coordinate (`novelty-probe.py`, `novelty-probe.log`).
- **Status:** the frozen output is committed unchanged, and no replacement metric is computed. The two maxima above already bound novelty.

## Interpretation

What the study shows:
- Under the supervised-validation-selected σ, mixing synthetic CCIL labels at α = 0.5 into BC's minibatches has no detectable effect on upright occupancy, at any real fraction and with either dynamics model. The intervals bound any effect at this σ to a few hundredths of occupancy.
- Real data is the dominant factor.
- The one detectable derived difference, exact − BC at r = 100% (+0.020), is small and secondary. It is consistent with action-noise augmentation of a policy near its ceiling, and is not evidence of correction.

What it does not show:
- **Corrective labels.** The diagnostics confirm the protocol's stated limit. Held-out imitation loss chose σ between 1e-5 and 1e-2. At those σ the synthetic supervision is the demonstration data with small action noise, so the study does not test whether corrective labels help or hurt.
- **Dynamics fidelity.** The exact − learned contrasts compare two dynamics models on near-identical labels. They do not measure the value of dynamics fidelity for correction.
- **The next question.** Choosing σ needs a criterion that rewards novelty as well as fidelity (CCIL #167).

## Reproducing and checking

- **Stage files:** check them against `manifest.json` with `shasum -a 256 docs/studies/evidence/pendulum-j-r-k/study/stages/*.json`.
- **Label diagnostics:** from a clean checkout of `6cbc8db` on siddh, with the dataset in `data/`, run:

  ```bash
  uv run python tools/diagnostics.py studies/pendulum-j-r-k/study.yaml <study root>/store --root <new dir> --out <file>
  ```

  It refuses if the store changes.
- **Study:** from a clean checkout of `5647117` on siddh, run the protocol's "Execution" command.
