# Benchmark landscape and M1 coverage decision

Reviewed 2026-10-28. This document records the evidence used to scope the first supervision study before implementation or outcome-bearing runs.

## Question

The repository asks how constructing or transforming supervision changes imitation learning. It does not claim to introduce the first unified imitation-learning benchmark. Existing projects already standardize datasets, environments, policies, and broad comparisons. The remaining need is a thin experimental layer that crosses supervision choices with matched learners while preserving each upstream benchmark's semantics.

## Existing layers

| Project | What it already owns | Scientific axis | What this repository should reuse | M1 decision |
| --- | --- | --- | --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot) | A standardized dataset format, policy implementations, training tools, and unified simulation evaluation. Its [benchmark interface](https://github.com/huggingface/lerobot/blob/main/docs/source/adding_benchmarks.mdx) wraps third-party Gymnasium environments and reports common success and return metrics. | Common robot-learning infrastructure. | Dataset and evaluation conventions, released policies, and existing or upstreamable environment adapters. | Adopt as infrastructure where compatible. Do not recreate its evaluator or dataset library. |
| [Much Ado About Noising](https://simchowitzlabpublic.github.io/much-ado-about-noising-project/) | A controlled comparison over 28 behavior-cloning tasks and multiple modalities. Its [released code](https://github.com/simchowitzlabpublic/much-ado-about-noising) provides matched regression, MIP, flow, and related objectives with Robomimic, Kitchen, and Push-T configurations. | Policy parameterization, stochastic injection, and supervised iterative computation. | Matched policy backends and modern training recipes. | Adopt regression and MIP; retain flow as a reference. Validate the released code as a modern baseline rather than claiming exact reproduction of paper numbers. |
| [D3IL](https://alrhub.github.io/d3il-website/) | Seven MuJoCo manipulation tasks, diverse human demonstrations, explicit valid behavior modes, and metrics for behavioral diversity. The [official implementation](https://github.com/ALRhub/d3il) includes 11 deterministic and multimodal, state- and image-based methods. | Whether a policy captures and reproduces legitimate multimodal behavior. | State tasks, behavior annotations, diversity metrics, and upstream evaluation semantics. | Primary M1 suite. Resolve data licensing and modern-environment compatibility before redistributing or converting data. |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | Manipulation demonstrations, low-dimensional and visual observations, multiple BC and offline-learning algorithms, and reproducible task-specific experiments. | Policy learning from robot manipulation demonstrations. | A small low-dimensional task subset and its native success semantics. Much Ado already supplies relevant policy configurations. | Primary M1 suite, restricted to a predeclared subset selected by geometry and horizon rather than observed outcomes. |
| Push-T through [LeRobot](https://github.com/huggingface/lerobot) and [Much Ado](https://github.com/simchowitzlabpublic/much-ado-about-noising) | A compact contact-rich pushing task, public demonstrations, state and image variants, and established diffusion/flow baselines. | Closed-loop correction under contact discontinuity. | State observations, coverage metric, evaluator, and matched policy configurations. | Primary M1 suite. |
| [ManiSkill](https://www.maniskill.ai/) | GPU-parallel simulation, many manipulation task families, large demonstration collections, and tested imitation-learning baselines. | Generalization and scale across objects and tasks. | A later scale and throughput study. | Deferred from M1 because it adds breadth and systems cost without a new first-order scientific axis after D3IL and robomimic. |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Fifty manipulation tasks and standard multi-task and meta-learning protocols. | Task-distribution and transfer breadth, primarily with scripted policies in the proposed study. | A later task-transfer study, preferably through an upstream LeRobot adapter. | Deferred from M1 because it overlaps manipulation breadth and offers less demonstration diversity than D3IL. |
| [Minari](https://minari.farama.org/) locomotion and Kitchen datasets | A standard Gymnasium-compatible API for offline datasets, including MuJoCo locomotion and Franka Kitchen. | Smooth control, offline data access, and long-horizon subtask composition. | Dataset identity and episode semantics in later studies. | Deferred. Locomotion is less diagnostic of human demonstration diversity; Kitchen is already represented in Much Ado and can be added after the core matrix works. |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 130 language-conditioned tasks, human demonstrations, fixed initial states, and lifelong-learning protocols. | Multitask transfer and continual imitation learning. | A later image, language, and multitask study through LeRobot. | Deferred to the visual/end-to-end phase. |
| [MAGICAL](https://proceedings.neurips.cc/paper/2020/hash/d464b5ac99e74462f321c06ccacc4bff-Abstract.html) | Controlled distribution shifts for testing whether imitation methods generalize demonstrator intent. | Robustness to deployment shift. | A later robustness study if its task abstraction remains useful. | Deferred because M1 first needs to establish the supervision effect in robot manipulation. |

## M1 coverage decision

M1 uses three complementary task families rather than accumulating every available suite.

| Question | Selected evidence | Why it is not redundant |
| --- | --- | --- |
| Does synthetic supervision help in a compact contact-discontinuous system? | Push-T state | Dynamics and labels remain directly inspectable, while contact exposes local-model failures. |
| Does the effect persist in higher-dimensional manipulation? | A predeclared robomimic low-dimensional subset | It adds robot kinematics, orientation, precision, and longer horizons. |
| Does synthetic supervision preserve legitimate human behavioral diversity? | D3IL state tasks and diversity metrics | Success alone cannot reveal mode collapse or distorted behavior frequencies. |
| Does the conclusion depend on the policy learner? | Matched regression, MIP, and flow backends from Much Ado | This separates supervision effects from stochasticity and iterative policy computation. |

The exact D3IL and robomimic task subsets must be selected from task structure, horizon, behavior count, and compute before viewing any CCIL comparison. The selection rationale and exclusions become part of the study predeclaration.

## Factorial study boundary

The core comparison crosses supervision with policy learning:

| Supervision | Regression | MIP | Flow reference |
| --- | --- | --- | --- |
| Original demonstrations | required | required | required where the released task configuration is available; otherwise predeclare the adapter |
| CCIL synthetic supervision | required | required | same coverage rule as the original-demonstration arm |

Naive NoiseBC remains a historical secondary baseline. It must not stand in for the stochastic-injection and iterative-computation mechanisms isolated by Much Ado. Certificate filtering and alternative CCIL model families are secondary ablations unless separately predeclared.

All matched arms use the same demonstration split, policy capacity where architectures permit, tuning budget, study seeds, evaluation starts, and outcome definitions. Necessary architectural differences are recorded rather than hidden behind a claim of exact equality.

## Ownership and contribution policy

1. Upstream suites own environment dynamics, task definitions, demonstrations, termination, and metrics.
2. LeRobot or the relevant upstream project owns generally useful dataset and evaluator integrations. Contributions should go upstream first.
3. Much Ado and other policy projects own their learners. This repository may provide thin version-pinned adapters, not copies.
4. CCIL and other supervision methods own label generation, certificates, and method-specific claims.
5. This repository owns study manifests, frozen recipes, cross-product orchestration, provenance, analysis, and reports.

## Entry gates

No suite enters an outcome-bearing run until:

- code, dataset, asset, and redistribution licenses are recorded;
- dataset identity and hashes are fixed;
- termination, truncation, action, observation, and evaluation semantics are audited;
- the upstream evaluator is reproduced on fixed policies;
- the supervision adapter cannot silently turn synthetic samples into temporal trajectories;
- policy and supervision configurations are frozen before study outcomes are inspected.

## Deferred claims

M1 will not claim a universal ranking of imitation-learning algorithms, reproduce every source paper, establish an image or language benchmark, or identify one dynamics model for all regimes. It asks a narrower question: under matched contemporary learners, what changes when the training supervision changes?
