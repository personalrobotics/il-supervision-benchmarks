# Supervision Benchmarks for Imitation Learning

`il-supervision-benchmarks` is a neutral home for reproducible studies of how constructing or transforming supervision changes imitation learning. Its first use is to measure the effect of synthetic labels, including CCIL, without conflating that effect with the policy architecture or training objective.

This is not another environment collection, dataset format, policy library, or general imitation-learning framework. It composes maintained work from the community:

- upstream benchmark suites own tasks, demonstrations, and metrics;
- LeRobot and upstream packages provide data and evaluation interfaces;
- established policy harnesses, initially Much Ado About Noising, provide matched regression, iterative, and flow-based learners;
- supervision methods such as CCIL produce versioned artifacts through their own repositories;
- this repository owns the factorial study design, frozen recipes, provenance, analysis, and reports.

The repository owns the experimental layer:

- pinned references to upstream task and dataset recipes;
- train, validation, and evaluation splits;
- matched policy-training and evaluation configurations;
- multi-method orchestration;
- predeclared analyses, raw per-seed results, and reports.

It does not own simulators, datasets, or learning methods. Environments and dataset semantics remain with their upstream projects. Methods remain in their own packages and enter a study through a documented artifact or adapter boundary. For example, CCIL owns candidate generation, certificates, and its method-specific quickstarts; this repository owns comparative experiments that consume those artifacts.

## Design principles

1. **Neutral supervision boundary.** A study may compare original demonstrations, CCIL, noise augmentation, or future supervision methods without making any one method the organizing abstraction.
2. **Upstream first.** Reuse maintained environments, datasets, evaluators, and policy implementations. Contribute generally useful integrations upstream before keeping local compatibility code.
3. **Declared science.** Freeze configurations, exclusions, statistics, and decision rules before inspecting study outcomes. Report negative and inconclusive results.
4. **Inspectable evidence.** Preserve resolved configurations, package and data versions, input hashes, per-seed records, and raw evaluation artifacts.
5. **Separated ownership.** Upstream projects define task semantics. Method repositories define algorithmic behavior. This repository defines comparisons.

## Initial scope

The first milestone uses the smallest complementary state-based suite that answers distinct questions:

- D3IL measures whether added supervision preserves legitimate behavioral diversity;
- Push-T measures low-data performance under contact discontinuity;
- a predeclared robomimic low-dimensional subset measures higher-dimensional manipulation;
- matched regression, MIP, and flow policies test whether a supervision effect depends on the learner.

MuJoCo locomotion, Meta-World, ManiSkill, Kitchen, LIBERO, image observations, and end-to-end representation learning remain useful later studies, but they do not all belong in the first experiment.

The evidence and selection rationale are recorded in the [benchmark landscape and M1 coverage decision](docs/benchmark-landscape.md).

## Related repositories

- [personalrobotics/CCIL](https://github.com/personalrobotics/CCIL): synthetic-label generation, certificates, adapters, and canonical CCIL examples.
- [LeRobot](https://github.com/huggingface/lerobot): common dataset, policy, and evaluation infrastructure.
- [Much Ado About Noising](https://github.com/simchowitzlabpublic/much-ado-about-noising): matched policy objectives and modern behavior-cloning recipes.
- [D3IL](https://github.com/ALRhub/d3il): diverse human demonstrations, task suites, and behavioral-diversity metrics.

## Status

The repository is defining its first study contract before implementation or outcome-bearing runs. Existing community benchmarks are dependencies and intellectual foundations, not work to be reproduced locally.
