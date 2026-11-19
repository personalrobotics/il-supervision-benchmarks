# Evidence index

Each study commits its decision records here. A run archive too large to commit is published as the asset of an evidence-only release, tagged `evidence-<study>-v<N>`. Each such release has a committed `archive.json` naming the archive's size, sha256, originating commits and reproduction command.

Evidence releases are not software releases. An asset is never overwritten. A correction is published as a new version, `-v<N+1>`.

Immutable releases are enabled for this repository from 2026-11-03. A release is assembled as a draft, its asset is verified against the committed sha256, and then it is published once. Publishing locks the release's tag and assets.

| Release | Study | Archive | sha256 | Locked |
|---|---|---|---|---|
| [evidence-pendulum-recovery-segments-stage-a-v1](https://github.com/personalrobotics/il-supervision-benchmarks/releases/tag/evidence-pendulum-recovery-segments-stage-a-v1) | [recovery segments, Stage A](https://github.com/personalrobotics/il-supervision-benchmarks/pull/23) | `20261102T221735Z-recovery-segments.tar.gz`, 96,111,122 bytes ([archive.json](pendulum-recovery-segments/archive.json)) | `9d0f25d957fc612589f0889be421bf14929c44f0307b8a19d15d8de99bc11f69` | no |
| [evidence-recovery-feasibility-stage-2-v1](https://github.com/personalrobotics/il-supervision-benchmarks/releases/tag/evidence-recovery-feasibility-stage-2-v1) | [recovery feasibility, Stage 2](https://github.com/personalrobotics/il-supervision-benchmarks/pull/72) | `20261118T162312Z-recovery-feasibility.tar.gz`, 20,504,863 bytes ([archive.json](recovery-feasibility/archive.json)) | `0f06850794e97d688dea92d752c24268cad5e24fb01920aa3b3c21b737d8198c` | yes |
| [evidence-recovery-learned-stage-3-v1](https://github.com/personalrobotics/il-supervision-benchmarks/releases/tag/evidence-recovery-learned-stage-3-v1) | [learned-dynamics recovery, Stage 3](https://github.com/personalrobotics/il-supervision-benchmarks/pull/75) | `20261118T190446Z-recovery-learned.tar.gz`, 34,549,985 bytes ([archive.json](recovery-learned/archive.json)) | `dbdde528dbc6b0103622e63100e3e71d5fc64fe22132da88e0b6471ec9bd7069` | yes |
| [evidence-recovery-receding-v1](https://github.com/personalrobotics/il-supervision-benchmarks/releases/tag/evidence-recovery-receding-v1) | [receding-horizon recovery](https://github.com/personalrobotics/il-supervision-benchmarks/pull/78) | `20261119T003320Z-recovery-receding.tar.gz`, 5,326,265 bytes ([archive.json](recovery-receding/archive.json)) | `af33f1c755687eb5a4fe2d56f1d7ca1d2ba50a856f1c3dd68882ecb4d3203872` | yes |
| [evidence-recovery-switch-v1](https://github.com/personalrobotics/il-supervision-benchmarks/releases/tag/evidence-recovery-switch-v1) | [recovery switch](https://github.com/personalrobotics/il-supervision-benchmarks/pull/81) | `20261119T045404Z-recovery-switch.tar.gz`, 3,202,453 bytes ([archive.json](recovery-switch/archive.json)) | `1f0053f31cc8bb49829476c7e4bdc07ff26557b0b3d547507d7bf8f3a5455ef9` | yes |

**Limitation of v1.** v1 was published before immutable releases were enabled, and the setting does not apply retroactively. GitHub does not lock its tag or asset. Tamper detection rests on three things:

- the sha256 committed in the merged `archive.json`, which must match the downloaded asset;
- the tag pointing at the merge commit `f142391`;
- the recorded clean-clone verification on 2026-11-03: the asset was downloaded from its release URL, the sha256 check reported `OK`, and the checker exited 0 with all checks passing. The full checker and protocol are preserved at tag [`pendulum-evidence-archive-v1`](https://github.com/personalrobotics/il-supervision-benchmarks/tree/pendulum-evidence-archive-v1).

A download whose sha256 differs from `archive.json` is not this evidence. v1 is not republished solely to obtain the lock.
