# J(r, K | α = 0.5) on Pendulum: protocol

Status: **draft, pilots pending.** Merging this protocol freezes it. Until then nothing here has been run on study seeds or on the final-test starts.

## Question

How do the amount of real data r and the synthetic dose K change the performance of a policy trained by behavior cloning (BC), by CCIL with learned dynamics, and by CCIL with the task's exact dynamics, when each minibatch is half real and half synthetic (α = 0.5)?

The exact-dynamics arm isolates dynamics-model error: it uses the same generator, the same selected σ and the same learner as learned-dynamics CCIL, and differs only in the transition model. It is not a task-optimal oracle (#145 remains a separate analysis).

## Design

| | |
|---|---|
| Task | Continuous Pendulum swing-up through the evaluator's discrete 0.02 s transition map (`ilsb.tasks.pendulum`), applied torque clipped at ±3, horizon 500. Commanded, applied and saturation diagnostics are kept. |
| Data | The canonical corrected continuous dataset: 100 episodes × 500 steps, SHA-256 `59c332bc…f43c` (CCIL release `data/pendulum-corrected-demonstrator-v1`). Every transition replays through the task map to within 9e-16; every action is within ±3. |
| Real fractions r | 10%, 25%, 100% of the episodes: nested prefixes of a seed-keyed permutation, the same for every arm. |
| Validation | 20% of each subset's episodes, seed-keyed, held out from training; used for checkpoint selection, patience and σ selection. |
| Synthetic doses K | 0 (BC), 1, 4, 20 = K_max: nested views of one K_max candidate set per (r, seed, dynamics, σ). |
| Arms | `bc`; `ccil_learned` (dynamics fitted on the subset's training episodes); `ccil_exact` (the task's transition map). |
| Generator | Newton preimage (`noisy_action_preimage`) for both CCIL arms, 30 iterations, tolerance 1e-4, proposals truncated to ±3 (the first K feasible draws of each transition's stream, with the raw attempt index recorded); validators: finite, solver success, support radius 1.0, action bound ±3. |
| Dynamics | The corrected recipe: MLP 512×512, soft-sampling Lipschitz (constraint 2.0, weight 1e-3, ε 3e-4, 5 samples, model-normalized coordinates), learning rate 1e-3, weight decay 1e-5, 100 epochs, batch 512. |
| Learner | `vector_bc_v2` for every arm: MLP 256×256, initial learning rate 1e-3, batch 512, standard observation scaling, min-max action scaling, α = 0.5 for CCIL arms. |
| Training procedure | Convergence-driven and the same for every unit; the number of updates is not held fixed. Training runs in steps, not epochs. The full validation set is evaluated every few steps, and the best checkpoint is kept. The learning rate drops by a factor after a plateau in validation loss, never below a floor. A unit stops only at the floor, after a further patience period without improvement. The step ceiling is a safety limit: a unit that reaches it is an incomplete fit, and the run fails. Pilot 1 chooses the interval, plateau patience, factor, floor, stopping patience and ceiling. |
| Evaluation | 100 fixed final-test starts (`eval_seed` 2201), used only by the study's evaluate stage. |
| Host | siddh, CUDA, is the canonical execution host. Store entries are scoped to it, and no numerical artifact is reused from another environment. |

## σ selection

σ is a hyperparameter of CCIL and is selected per (r, K, seed) from validation data only (`docs/harness.md`, "Selecting σ"):
- **Grid,** in normalized action units (a fraction of the bound 3): 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.
- **Feasibility before training:** a σ is not trained when more than 5% of its learned-dynamics candidates fail genuinely (non-finite, failed solve, or outside the support). The action bound cannot reject a bounded proposal; if it does, the run stops.
- **Selection:** the least mean squared action error, in raw torque units, on the unit's held-out demonstration episodes; ties go to the smaller σ.
- **The exact arm** uses the selected σ of the same (r, K, seed). If more than 5% of its own candidates at that σ fail, the run stops.

`tools/sigma_geometry.py` reports what each σ produces under exact dynamics, as a committed diagnostic. It does not select.

## Analysis

- **Metric:** upright occupancy (fraction of steps within 8° of upright), per seed the mean over the 100 final-test episodes. Normalized height, final hold, effort and saturation are secondary.
- **Interval:** the union of BCa and paired t over seeds, paired by seed.
- **Primary family (9):** learned − BC, exact − BC and exact − learned at K = K_max, at each r. Bonferroni: c = 1 − 0.05/9 ≈ 0.9944.
- **Secondary:** the same contrasts at K = 1 and 4 (uncorrected, reported as description), the selected σ distribution per (r, K), and the candidates' acceptance.
- **Outcomes:** "improves" when the lower endpoint is above 0, "worse" when the upper endpoint is below 0, otherwise "no detectable difference", reported with its interval and no claim of absence.

## Pilots

Pilots use non-study seeds (9501–9520) and non-study evaluation starts (`eval_seed` 9601). They choose only the convergence settings, the step ceiling and the seed count, by the rules in `tools/pilot.py`. They report no arm or contrast mean.

An earlier pilot trained 200 epochs at a constant learning rate. At the end, validation loss was still falling: the median best step was 90–95% of the schedule, and the loss swung about 2× between evaluations. That schedule did not converge, so it was replaced by the procedure above.

1. **Convergence** (`studies/pendulum-j-r-k/pilot-stopping-{1..4}.yaml`, seeds 9501–9505). Each rung trains every unit under one set of convergence settings, cheapest first:

   | Rung | Interval (steps) | Plateau patience | Factor | Floor | Stopping patience |
   |---|---|---|---|---|---|
   | 1 | 50 | 5 | 0.5 | 1e-5 | 5 |
   | 2 | 50 | 10 | 0.5 | 1e-5 | 10 |
   | 3 | 100 | 10 | 0.2 | 1e-6 | 20 |
   | 4 | 100 | 20 | 0.2 | 1e-6 | 40 |

   - **Extension:** after its stop, each unit trains on to twice its stop step, once, to measure what more training would have gained. The returned checkpoint is the one selected at the stop.
   - **Choice:** the first rung whose rule fires on every unit, and whose extension improves no unit's selected validation loss by more than 2%.
   - **Ceiling:** three times that rung's latest stop step, rounded up to 10,000.
2. **Precision** (seeds 9511–9520, with pilot 1's settings and ceiling): the standard deviation of each primary contrast's paired differences. The study uses the smallest n with z_c · sd / √n ≤ 0.05 for every primary contrast, clipped to [20, 50]. Study seeds are 1301 onward.

## Execution

From a clean checkout of the merged protocol on siddh:

```bash
mkdir -p data && curl -fL -o data/pendulum_cont_corrected.npz https://github.com/personalrobotics/CCIL/releases/download/data/pendulum-corrected-demonstrator-v1/pendulum_cont_corrected-sha256-59c332bc829db9c9da40d9c1bd9ebbf808965aca6a167be48dc0f2ade52cf43c.npz
sha256sum data/pendulum_cont_corrected.npz   # 59c332bc829db9c9da40d9c1bd9ebbf808965aca6a167be48dc0f2ade52cf43c (preflight refuses other bytes)
uv run python -m ilsb run studies/pendulum-j-r-k/study.yaml --root runs/<timestamp>-j-r-k
```

The run proceeds through preflight, train, audit, evaluate, evaluation audit and report, and stops at the first failed check. Any undeclared decision stops the run, and the protocol is amended in a new pull request.

## Deferred

Value iteration, HJB and Pontryagin checks; the (ρ, λ) regime map; MIP and flow learners, multiple learners and the α sweep; external benchmarks.
