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
- `root.json`: the protocol identity, meaning the specification and its hash, the harness commit, the pinned ccil commit, and the lock hash.
- `stages/<stage>.json`: each stage's report.
- `manifest.json`: the SHA-256 of every report that passed.

A stage runs only when every predecessor's report passed and still hashes to the manifest's record under the same protocol. Running a stage out of order is refused. A root created under another protocol is refused.

**Units are store entries.** Each entry is named by the hash of its complete input identity: subsets, splits, dynamics fits, K_max candidate sets, nested-K views, policies and rollouts. An entry:
- is written whole and renamed into place;
- carries a manifest of its files' hashes.

**Reuse** requires an equal identity and verified files. So rerunning a root resumes without recomputing, and a store shared by several studies reuses equal work. A mismatched or corrupted entry is refused, never silently replaced.

## The stages

| Stage | What it checks or produces |
|---|---|
| preflight | The dataset matches its declared hash; the plan of units. |
| train | Every policy unit. A whole (arm, r, K) cell trains as one group, through the learner adapter. |
| audit | Every unit is a verified entry. Its training identity, which hashes the supervision manifest with the dataset, split and candidate hashes, recomputes from the unit's own inputs. Its weights hash to the record. |
| evaluate | One rollout per policy from the task's fixed final-test starts, with its scores. |
| evaluation_audit | Every rollout reopened. It checks hashes, shapes, starts and the action bound, replays every transition through the task map, and recomputes every score exactly. |
| report | The predeclared paired contrasts (union BCa/t interval) over seeds. |

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
  - patience on the validation history, with the step budget only a ceiling;
  - a fixed synthetic sampling fraction;
  - a realized-exposure record.

## Evidence

- `tools/port_check.py --ccil CCIL_CHECKOUT` gives the Pendulum map, scorer and starts against CCIL, written to `docs/fixtures/evidence/pendulum-port-check.json`.
- `tools/fixture.py --work NEW_DIR` is the compact end-to-end fixture, written to `docs/fixtures/evidence/vertical-slice-fixture.json`:
  - every arm, two fractions, two doses and three seeds, through every stage on CPU;
  - a resume that reruns nothing;
  - a second study that reuses every unit and reports identically;
  - refusals of a wrong dataset hash, a corrupted rollout, a report after a failed audit, a mismatched store entry, and an out-of-order stage.

  The evidence holds no paths, times or commits, so a rerun on the same host reproduces it byte for byte.
