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
| Generator | Newton preimage (`noisy_action_preimage`) for both CCIL arms, 30 iterations, tolerance 1e-4; validators: finite, solver success, support radius 1.0, action bound ±3. |
| Dynamics | The corrected recipe: MLP 512×512, soft-sampling Lipschitz (constraint 2.0, weight 1e-3, ε 3e-4, 5 samples, model-normalized coordinates), learning rate 1e-3, weight decay 1e-5, 100 epochs, batch 512. |
| Learner | `vector_bc_v2` for every arm: MLP 256×256, learning rate 1e-3, batch 512, standard observation scaling, min-max action scaling, α = 0.5 for CCIL arms, validation checkpoint selection every 50 steps, patience P and step ceiling from pilot 1. |
| Evaluation | 100 fixed final-test starts (`eval_seed` 2201), used only by the study's evaluate stage. |
| Host | siddh, CUDA, is the canonical execution host. Store entries are scoped to it, and no numerical artifact is reused from another environment. |

## σ selection

σ is a hyperparameter of CCIL and is selected per (r, K, seed) from validation data only (`docs/harness.md`, "Selecting σ"):
- **Grid:** 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.
- **Feasibility before training:** a σ whose learned-dynamics candidates have acceptance below 0.95 is not trained.
- **Selection:** the least mean squared action error, in raw torque units, on the unit's held-out demonstration episodes; ties go to the smaller σ.
- **The exact arm** uses the selected σ of the same (r, K, seed). If its own acceptance at that σ is below 0.95, the run stops.

`tools/sigma_geometry.py` reports what each σ produces under exact dynamics, as a committed diagnostic. It does not select.

## Analysis

- **Metric:** upright occupancy (fraction of steps within 8° of upright), per seed the mean over the 100 final-test episodes. Normalized height, final hold, effort and saturation are secondary.
- **Interval:** the union of BCa and paired t over seeds, paired by seed.
- **Primary family (9):** learned − BC, exact − BC and exact − learned at K = K_max, at each r. Bonferroni: c = 1 − 0.05/9 ≈ 0.9944.
- **Secondary:** the same contrasts at K = 1 and 4 (uncorrected, reported as description), the selected σ distribution per (r, K), and the candidates' acceptance.
- **Outcomes:** "improves" when the lower endpoint is above 0, "worse" when the upper endpoint is below 0, otherwise "no detectable difference", reported with its interval and no claim of absence.

## Pilots

Pilots use non-study seeds (9501–9520) and non-study evaluation starts (`eval_seed` 9601). They choose only patience, the step ceiling and the seed count, by the rules in `tools/pilot.py`. They report no arm or contrast mean.

1. **Stopping** (`studies/pendulum-j-r-k/pilot-stopping.yaml`, seeds 9501–9505): every unit trains a 200-epoch schedule with the stopping rule off and records its validation history.
   - **Patience:** the smallest P in {5, 10, 20, 40} evaluations that fires on every unit with a 95th-percentile regret of at most 1%.
   - **Ceiling:** twice the latest stop step, rounded up to 1000.
2. **Precision** (seeds 9511–9520, with pilot 1's P and ceiling): the standard deviation of each primary contrast's paired differences. The study uses the smallest n with z_c · sd / √n ≤ 0.05 for every primary contrast, clipped to [20, 50]. Study seeds are 1301 onward.

## Execution

From a clean checkout of the merged protocol on siddh:

```bash
gh release download data/pendulum-corrected-demonstrator-v1 -R personalrobotics/CCIL \
    -p 'pendulum_cont_corrected-*.npz' -O data/pendulum_cont_corrected.npz
uv run python -m ilsb run studies/pendulum-j-r-k/study.yaml --root runs/<timestamp>-j-r-k
```

The run proceeds through preflight, train, audit, evaluate, evaluation audit and report, and stops at the first failed check. Any undeclared decision stops the run, and the protocol is amended in a new pull request.

## Deferred

Value iteration, HJB and Pontryagin checks; the (ρ, λ) regime map; MIP and flow learners, multiple learners and the α sweep; external benchmarks.
