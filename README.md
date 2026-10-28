# IL Benchmarks

`il-benchmarks` is a neutral home for reproducible imitation-learning experiments that compare ways of constructing training supervision. Its first use is to measure the effect of synthetic labels, including CCIL, against ordinary behavior cloning and other declared baselines.

The repository owns the experimental layer:

- pinned task and dataset recipes;
- train, validation, and evaluation splits;
- policy-training and evaluation configurations;
- multi-method orchestration;
- predeclared analyses, raw per-seed results, and reports.

It does not own simulators, datasets, or learning methods. Environments and dataset semantics remain with their upstream projects. Methods remain in their own packages and enter a study through a documented artifact or adapter boundary. For example, CCIL owns candidate generation, certificates, and its method-specific quickstarts; this repository owns comparative experiments that consume those artifacts.

## Design principles

1. **Neutral method boundary.** A benchmark may compare CCIL, no augmentation, noise augmentation, or future supervision methods without making any one method the organizing abstraction.
2. **Thin orchestration.** Reuse maintained environment and dataset APIs, including LeRobot where appropriate. Do not build a second simulator or imitation-learning framework here.
3. **Declared science.** Freeze configurations, exclusions, statistics, and decision rules before inspecting study outcomes. Report negative and inconclusive results.
4. **Inspectable evidence.** Preserve resolved configurations, package and data versions, input hashes, per-seed records, and raw evaluation artifacts.
5. **Separated ownership.** Upstream projects define task semantics. Method repositories define algorithmic behavior. This repository defines comparisons.

## Initial scope

The first milestone is a state-based, low-data imitation-learning suite covering maintained community tasks. It begins with Push-T, MuJoCo locomotion, Meta-World, robomimic low-dimensional tasks, and ManiSkill. Each integration should be useful for methods unrelated to CCIL.

Image observations and end-to-end representation learning are intentionally deferred until their method and dataset contracts stabilize.

## Related repositories

- [personalrobotics/CCIL](https://github.com/personalrobotics/CCIL): synthetic-label generation, certificates, adapters, and canonical CCIL examples.

## Status

This repository is being established from the benchmark work previously tracked in CCIL. The initial milestone and issues define the contracts before implementation begins.
