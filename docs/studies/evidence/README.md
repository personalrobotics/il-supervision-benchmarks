# Evidence index

Each study commits its decision records here. A run archive too large to commit is published as the asset of an evidence-only release, tagged `evidence-<study>-v<N>`. Each such release has a committed `archive.json` naming the archive's size, sha256, originating commits and reproduction command.

Evidence releases are not software releases. An asset is never overwritten. A correction is published as a new version, `-v<N+1>`.

Immutable releases are enabled for this repository from 2026-11-03. A release is assembled as a draft, its asset is verified against the committed sha256, and then it is published once. Publishing locks the release's tag and assets.

| Release | Study | Archive | sha256 | Locked |
|---|---|---|---|---|
| [evidence-pendulum-recovery-segments-stage-a-v1](https://github.com/personalrobotics/il-supervision-benchmarks/releases/tag/evidence-pendulum-recovery-segments-stage-a-v1) | [recovery segments, Stage A](https://github.com/personalrobotics/il-supervision-benchmarks/pull/23) | `20261102T221735Z-recovery-segments.tar.gz`, 96,111,122 bytes ([archive.json](pendulum-recovery-segments/archive.json)) | `9d0f25d957fc612589f0889be421bf14929c44f0307b8a19d15d8de99bc11f69` | no |

**Limitation of v1.** v1 was published before immutable releases were enabled, and the setting does not apply retroactively. GitHub does not lock its tag or asset. Tamper detection rests on three things:

- the sha256 committed in the merged `archive.json`, which must match the downloaded asset;
- the tag pointing at the merge commit `f142391`;
- the recorded clean-clone verification on 2026-11-03: the asset was downloaded from its release URL, the sha256 check reported `OK`, and the checker exited 0 with all checks passing. The full checker and protocol are preserved at tag [`pendulum-evidence-archive-v1`](https://github.com/personalrobotics/il-supervision-benchmarks/tree/pendulum-evidence-archive-v1).

A download whose sha256 differs from `archive.json` is not this evidence. v1 is not republished solely to obtain the lock.
