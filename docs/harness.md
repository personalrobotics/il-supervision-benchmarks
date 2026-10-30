# The study harness

This is the first vertical slice of #11. It runs one study at a time, through one state machine, from a frozen specification to a reproducible report. The first study (#12) is Pendulum J(r, K | α = 0.5). Other benchmarks join by adding adapters; they do not copy the runner.

## Workflow

A study gets two pull requests and one automatic run in between.

1. **Protocol PR, before any outcome.** It contains:
   - the specification (task, data hash, arms, coordinates, seeds, settings, analysis);
   - any new adapter code;
   - a smoke run of the whole sequence.

   Merging it freezes the protocol.
2. **One command, from a clean checkout of the merged protocol:**

   ```bash
   uv run python -m ilsb run studies/<study>.yaml --root RUNS/<study> [--store STORE]
   ```

   The stages run in order: `preflight → train → audit → evaluate → evaluation_audit → report`. A passing stage advances automatically. A failed check stops the run, and the failing stage's report is kept for diagnosis.
3. **Results PR.** It contains the run manifest, the stage reports and the interpretation. Every number in the report regenerates from the recorded artifacts.

Nobody inspects results during the run. If an undeclared scientific decision comes up, the run stops and the protocol is amended in a new pull request.

## State and resume

The run root holds three kinds of file:
- `root.json`: the protocol identity, meaning the specification and its hash, the harness commit, the pinned ccil commit, the lock hash, and the scope (below).
- `stages/<stage>.json`: each stage's report.
- `manifest.json`: the SHA-256 of every report that passed.

A stage runs only when every predecessor's report passed and still hashes to the manifest's record under the same protocol. Running a stage out of order is refused. A root created under another protocol is refused.

**Units are store entries.** Each entry is named by the hash of its complete input identity: subsets, splits, dynamics fits, K_max candidate sets, nested-K views, policies and rollouts. An entry:
- is written whole and renamed into place;
- carries a manifest of its files' hashes.

**Every identity carries the scope:** the implementation and the execution environment.
- The implementation: the SHA-256 of the harness package source (so an uncommitted change is named too), the pinned ccil commit, and the lock hash. A run refuses a ccil that is not installed from a pinned commit.
- The environment: Python, operating system and machine, NumPy, PyTorch, and the device (with the CUDA version and GPU on CUDA).

So work done by other code, another lock or another environment is never reused. The study's numbers belong to its declared execution host; bitwise agreement across hosts is not claimed.

**Reuse** requires an equal identity, scope included, and verified files. So rerunning a root resumes without recomputing, and a store shared by several studies reuses equal work. A mismatched or corrupted entry is refused, never silently replaced.

## The stages

| Stage | What it checks or produces |
|---|---|
| preflight | The dataset matches its declared hash; the plan of units. |
| train | Every policy unit, each group of seeds trained together through the learner adapter, in three steps (below): BC and the learned arm at every feasible σ; the σ selection; the exact arm at the selected σ. Every unit must stop by the validation rule (patience at the plateau schedule's floor); a unit that reaches the step ceiling is an incomplete fit, and the stage fails. So does a coordinate with no feasible σ. |
| audit | Every trained unit is a verified entry. Its training identity, which hashes the supervision manifest with the dataset, split and candidate hashes, recomputes from the unit's own inputs. Its weights hash to the record, and its reported row (stopping, exposure, validation error) recomputes. Feasibility, the σ selection and the analysis units recompute to the reported ones. |
| evaluate | One rollout per analysis unit from the task's fixed final-test starts, with its scores. Unselected σ candidates are never evaluated. |
| evaluation_audit | Every rollout reopened. It checks hashes, shapes, starts and the action bound, replays every transition through the task map, and recomputes every score exactly. |
| report | The predeclared paired contrasts (union BCa/t interval) over seeds. |

## Selecting σ

The action-noise scale σ is a hyperparameter of CCIL: it sets the distribution and locality of the synthetic supervision. The study does not fix it. It selects it the way a user of the method would, from validation data only:

1. **Proposals and feasibility, before training.** σ is declared in normalized action units (a fraction of the action bound). Proposals are drawn from a Gaussian truncated to the actuator range (`action_bounds` of the CCIL generator): the first K_max feasible proposals of each transition's stream, so at a saturated action the perturbations point inward, and a nested K view is the first K of them. For each (r, seed) and each σ of the grid, the learned arm's K_max candidate set is generated. A σ whose genuine-failure fraction (non-finite, a failed solve, or a state outside the support) exceeds `generation.max_failure` is infeasible and never trained. The action-bound validator stays: after bounded proposals, any rejection by it is an implementation error and stops the run.
2. **Candidates.** The learned arm trains at every feasible σ and every K.
3. **Selection, per (r, K, seed).** Each candidate's selected checkpoint predicts the actions of the unit's held-out demonstration episodes. The σ with the least mean squared error, in raw action units, is selected; ties go to the smaller σ. Raw units keep the errors comparable across σ, whereas the trainer's own validation loss is in each model's scaled units.
4. **The exact arm** trains at the selected σ of the same (r, K, seed). So learned − exact compares two dynamics models under one selected σ, and the contrast stays a measure of dynamics-model error.

The final-test starts are never used for selection. The train report records every candidate's validation error, every candidate set's failure fraction and feasibility, and the selected σ; the audit recomputes all three. `tools/sigma_geometry.py SPEC` reports what each σ of the grid produces under the exact dynamics (failures, rejections, proposals per transition, displacement against the data's spacing) as a diagnostic. It selects nothing: selecting σ from exact dynamics would make the learned method oracle-assisted.

## Adapters

A benchmark contributes adapters, not a runner.

**Task** (`ilsb/tasks/`). It provides the transition map or evaluator, the fixed evaluation starts, the scorer, the nested real-data subsets, and optionally an exact model for an exact-dynamics arm.
- **Pendulum** ports CCIL's batched evaluator verbatim. `tools/port_check.py` checks it bit for bit against CCIL at the pinned commit, along with the scorer and the evaluation starts.

**Supervision** (`ilsb/supervision/`). It provides each arm's training and validation supervision, built from a method's public artifacts.
- **CCIL:**
  - a dynamics fit on the subset's training episodes;
  - one K_max candidate set per (subset, dynamics);
  - nested-K views (`ccil.core.nested_view`);
  - `ccil.supervision.build_supervision` with the episode split.

**Learner** (`ilsb/learners/`). It trains through the backend's own validation selection, stopping and records.
- **`vector_bc_v2`** is the first learner:
  - a plateau learning-rate schedule and a stop at its floor, on the validation history, with the step budget only a ceiling;
  - a fixed synthetic sampling fraction;
  - a realized-exposure record.

## Evidence

- `tools/port_check.py --ccil CCIL_CHECKOUT` gives the Pendulum map, scorer and starts against CCIL, written to `docs/fixtures/evidence/pendulum-port-check.json`.
- `tools/fixture.py --work NEW_DIR` is the compact end-to-end fixture, written to `docs/fixtures/evidence/vertical-slice-fixture.json`:
  - every arm, two fractions, two doses and three seeds, through every stage on CPU, with a three-point σ grid;
  - a resume that reruns nothing;
  - a second study that reuses every unit and reports identically;
  - refusals of a wrong dataset hash, a corrupted rollout, a report after a failed audit, a mismatched store entry, and an out-of-order stage;
  - a copy of the harness whose source differs by one comment, which retrains every unit instead of reusing the store's;
  - a step ceiling below any patience stop, which fails the train stage on every unit, after which the audit refuses;
  - no genuine generation failure tolerated, which skips every σ with any failure untrained and fails the train stage on the coordinates left without a feasible σ.

  The evidence holds no paths, times or commits, so a rerun on the same host reproduces it byte for byte.
