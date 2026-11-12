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
| train | Every policy unit, each group of seeds trained together through the learner adapter, in three steps (below): BC and the learned arm at every feasible σ of the grid, or at each coordinate's fixed σ; the σ selection; the exact arm at the selected σ. Each dose takes its own seeds: every seed at K = 0 and K_max, and the `surface_seeds` prefix at the doses between, when it is declared. Every unit must stop by the validation rule (patience at the plateau schedule's floor); a unit that reaches the step ceiling is an incomplete fit, and the stage fails. So does a coordinate with no feasible σ. |
| audit | Every trained unit is a verified entry. Its training identity, which hashes the supervision manifest with the dataset, split and candidate hashes, recomputes from the unit's own inputs. Its weights hash to the record, and its reported row (stopping, exposure, validation error) recomputes. Feasibility, the σ selection and the analysis units recompute to the reported ones. |
| evaluate | One rollout per analysis unit from the task's fixed final-test starts, with its scores. Unselected σ candidates are never evaluated. |
| evaluation_audit | Every rollout reopened. It checks hashes, shapes, starts and the action bound, replays every transition through the task map, and recomputes every score exactly. |
| report | Each (arm, r, K) cell's mean over its dose's seeds (descriptive), and the predeclared primary and derived paired contrasts at K_max (union BCa/t interval) over every seed. |

## Selecting σ

The action-noise scale σ is a hyperparameter of CCIL: it sets the distribution and locality of the synthetic supervision. A specification declares either `generation.sigma_grid`, selected per (r, K, seed) as below, or `generation.sigma`, one fixed σ per (r, K) chosen beforehand (for example by a pilot from validation data). With a fixed σ, step 2 trains only that σ, so the selection in step 3 is that σ, or none if it is infeasible, which fails the train stage. With a grid, σ is selected the way a user of the method would, from validation data only:

1. **Proposals and feasibility, before training.** σ is declared in normalized action units (a fraction of the action bound). Proposals are drawn from a Gaussian truncated to the actuator range (`action_bounds` of the CCIL generator): the first K_max feasible proposals of each transition's stream, so at a saturated action the perturbations point inward, and a nested K view is the first K of them. For each (r, seed) and each σ of the grid, the learned arm's K_max candidate set is generated. A σ whose genuine-failure fraction (non-finite, a failed solve, or a state outside the support) exceeds `generation.max_failure` is infeasible and never trained. The action-bound validator stays: after bounded proposals, any rejection by it is an implementation error and stops the run.
2. **Candidates.** The learned arm trains at every feasible σ and every K.
3. **Selection, per (r, K, seed).** Each candidate's selected checkpoint predicts the actions of the unit's held-out demonstration episodes. The σ with the least mean squared error, in raw action units, is selected; ties go to the smaller σ. Raw units keep the errors comparable across σ, whereas the trainer's own validation loss is in each model's scaled units.
4. **The exact arm** trains at the selected σ of the same (r, K, seed). So learned − exact compares two dynamics models under one selected σ, and the contrast stays a measure of dynamics-model error.

The final-test starts are never used for selection. The train report records every candidate's validation error, every candidate set's failure fraction and feasibility, and the selected σ; the audit recomputes all three. Selecting σ from exact dynamics would make the learned method oracle-assisted.

## Adapters

A benchmark contributes adapters, not a runner.

**Task** (`ilsb/tasks/`). It provides the transition map or evaluator, the fixed evaluation starts, the scorer, the nested real-data subsets, and optionally an exact model for an exact-dynamics arm.
- **Pendulum** ports CCIL's batched evaluator verbatim. `tools/port_check.py` checks it bit for bit against CCIL at the tag `pendulum-evidence-archive-v1`, along with the scorer and the evaluation starts.

**Supervision** (`ilsb/supervision/`). It provides each arm's training and validation supervision, built from a method's public artifacts.
- **CCIL:**
  - a dynamics fit on the subset's training episodes;
  - one K_max candidate set per (subset, dynamics);
  - nested-K views (`ccil.core.nested_view`);
  - `ccil.supervision.build_supervision` with the episode split.

**Learner** (`ilsb/learners/`). It trains through the backend's own validation selection, stopping and records.
- **`ilsb/learners/vector/`** holds the vector-policy trainers v1 and v2, NoiseBC and their batched engine, moved from CCIL (CCIL#168) with only their imports changed. Trainer names, versions and training identities are CCIL's, so a policy either CCIL trainer wrote loads and evaluates here, and a resumable checkpoint of trainer v1 resumes here. Trainer v2 writes inference checkpoints only; it has no resume.
- **`vector_bc_v2`** is the first learner:
  - a plateau learning-rate schedule and a stop at its floor, on the validation history, with the step budget only a ceiling;
  - a fixed synthetic sampling fraction;
  - a realized-exposure record.

## One policy: `ilsb policy train`

`ilsb policy train CONFIG` replaces `ccil policy train`. It trains one vector policy on CCIL's supervision artifacts. The configuration is versioned (`ilsb.policy-train` 1):

```yaml
schema: ilsb.policy-train
version: 1
labels:                  # the options of ccil labels generate
  recipe: analytic_double_integrator
  set: [data.path=rollouts.npz, compute.device=cpu, generation.method=noisy_action,
        generation.labels_per_transition=4, generation.action_noise_std=0.05]
supervision:
  source: demonstrations_plus_selection   # or demonstrations
policy:
  trainer: vector_bc_v2                   # or vector_bc_v1
  seed: 7
  device: cpu
  deterministic: true
  batch_size: 32
  epochs: 4
  noise: null                             # NoiseBC: {scale: 0.1}
run:
  output_dir: policies
```

- **Labels.** `ccil labels generate --result` runs CCIL's label stages, each reused when complete, and the artifacts are read from its machine-readable result (`ccil.labels-result` 1.x).
- **Supervision** names what the policy learns from: `demonstrations`, or `demonstrations_plus_selection` (the demonstrations and CCIL's selected labels). NoiseBC is `demonstrations` with `policy.noise` set. The former `bc`, `ccil` and `noise_bc` methods map to these.
- **The policy's seed, device and determinism are its own,** so labels generated once train any number of policy seeds, or on another device.
- **Every field is checked** for its exact type and range before CCIL runs or a learner is built. A malformed configuration exits 2 and writes nothing. The module docstring of [`src/ilsb/policy.py`](../src/ilsb/policy.py) lists every field and its default.
- **Output.** The policy is written to `<run.output_dir>/policy-<identity>/`, whole, and reused when complete. Its `manifest.json` holds the normalized configuration that wrote it. For the same seed and settings, its `policy.npz` holds the arrays `ccil policy train` wrote before CCIL removed it, and `evaluation.json` holds the same offline metrics.

## Evidence

- `tools/port_check.py --ccil CCIL_ARCHIVE` gives the Pendulum map, scorer and starts against CCIL at `pendulum-evidence-archive-v1`, written to `docs/fixtures/evidence/pendulum-port-check.json`.
- `tools/learner_check.py --work NEW_DIR --reference PYTHON` checks the learners moved from CCIL against CCIL's own trainers, written to `docs/fixtures/evidence/learner-migration.json`. CCIL removed the trainers after `REFERENCE_CCIL` (CCIL#168), so they run on the same host in a second environment, `PYTHON`, that holds CCIL at that commit and the same torch; the script's docstring gives the install command. ilsb runs with its pinned CCIL:
  - `ilsb policy train` and the reference's `ccil policy train`, with the same seed, for BC, NoiseBC and CCIL with trainers v1 and v2, generate the same label arrays, train checkpoints with the same arrays, and give the same offline metrics. The checkpoints' metadata and the supervision manifests differ only in the label files' hashes and the identities derived from them, because each CCIL records its own source in its labels' provenance;
  - two more policy seeds on the same labels reuse every label stage;
  - fifteen malformed configurations each exit 2 with their message, before CCIL runs or anything is written;
  - on the same labels, the study learner's group of three seeds trains to the same parameters and records;
  - a v1 resumable checkpoint CCIL wrote loads with the same state and predictions, and resumes to the same next checkpoint byte for byte; CCIL's v1 and v2 inference checkpoints reproduce their offline metrics; and resuming an inference checkpoint or another run's is refused.
- `tools/fixture.py --work NEW_DIR` is the compact end-to-end fixture, written to `docs/fixtures/evidence/vertical-slice-fixture.json`. One real study builds the valid artifacts once: every arm through every stage on CPU, at one real fraction, one dose, one fixed σ and two seeds, scored on normalized height so the paired analysis computes its BCa and t intervals. The other checks reuse those artifacts or copies of them, and train only where a failure needs new numerical work:
  - a preflight alone on 2 fractions × 3 doses × 3 seeds, whose 34 planned units, seed roles, K_max candidate sets and nested subsets match the design, with nothing trained;
  - the exposure of a learned unit, reconciled from its saved record against the BC unit's;
  - a resume that reruns nothing, and a second study that reuses every unit and reports identically;
  - refusals of a wrong dataset hash, a corrupted rollout, a report after a failed audit, a mismatched store entry, and an out-of-order stage;
  - the BC cell, reused under the same code and retrained under a copy of the installed harness whose source differs by one comment;
  - the BC cell under a step ceiling below any patience stop, which fails the train stage, after which the audit refuses;
  - a σ grid with no generation failure tolerated, which trains exactly the feasible settings: one seed has a feasible and an infeasible σ, and the other, with none feasible, leaves its coordinate unselectable and fails the train stage.

  The fixture's docstring maps each protected failure to its check. The evidence counts every policy trained, and holds no paths, times or commits, so a rerun on the same host reproduces it byte for byte.

CI (`.github/workflows/fixture.yml`) builds the wheel, installs it in a clean environment with CPU PyTorch, and runs `learner_check.py` (with a reference environment at `REFERENCE_CCIL`) and `fixture.py` from there; the evidence is uploaded as the run's artifact. A fixture runs only when its inputs changed: `src/`, `pyproject.toml`, `uv.lock`, the workflow, or its own script. Otherwise its check passes without running. A new push to a pull request cancels the obsolete run. There is no unit-test collection.
