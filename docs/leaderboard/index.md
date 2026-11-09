# Leaderboard

Shortcut detection and mitigation methods, each scored by the same code on the same
benchmark with planted shortcuts of known strength. Four tracks, scored separately:
an embedding method and an image method are never ranked against each other.

**Have a method?** Add it in one folder and one pull request: [how to submit](submit.md).
CI scores it on the benchmark and your entry appears here, with your name and paper, when it is merged.

!!! note "Separate from the paper's audit"
    The leaderboard measures methods on synthetic benchmarks. It does not change what
    [`conditional_audit`](../api/audit.md), the package's calibrated test, reports.
    The benchmark's rules and metrics are on the [tracks page](tracks.md).

<!-- LEADERBOARD:START -->

## Embedding detection

Given embeddings E, an attribute A and a task label Y, score whether E carries A beyond what Y explains. Null datasets have A correlated with Y but no extra attribute signal, so a method that only finds the correlation scores 0.5. Ranked by **AUROC**, after grouping: entries whose flag fires on at most 20% of null datasets first, then entries without a flag, then entries whose flag fires more often. Benchmark v1.1. [Track rules](tracks.md#embedding-detection).

| # | Method | Authors | Paper | AUROC | linear | xor | variance | FPR | TPR |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | [Embedding frequency shortcut (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-frequency) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/2307.09829), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.678** | 0.989 | 0.507 | 0.539 | 0.000 | 0.111 |
| 2 | [Attribute probe (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-probe) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1610.01644), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.652** | 0.923 | 0.512 | 0.521 | 0.011 | 0.189 |
| 3 | [Conditional probe (ShortKit-ML audit)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-conditional-probe) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://doi.org/10.64898/2026.04.29.26352053) | **0.646** | 0.988 | 0.413 | 0.538 | 0.067 | 0.367 |
| 4 | [HBAC clustering (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-hbac) <small>baseline</small> | ShortKit-ML maintainers | [1](https://doi.org/10.1016/j.ipm.2021.102519), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.621** | 0.828 | 0.515 | 0.522 | 0.000 | 0.022 |
| 5 | [Equalized odds gap (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-equalized-odds) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1610.02413), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.549** | 0.649 | 0.478 | 0.521 | 0.044 | 0.100 |
| 6 | [GroupDRO gap (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-groupdro) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1911.08731), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.434** | 0.391 | 0.516 | 0.394 | 0.000 | 0.000 |
| 7 | [GCE high-loss samples (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-gce) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1805.07836), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.418** | 0.351 | 0.396 | 0.508 | 0.056 | 0.078 |
| 8 | [Within-task mean difference](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/mean-difference) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://doi.org/10.4324/9780203771587) | **0.644** | 0.991 | 0.448 | 0.493 | – | – |
| 9 | [Bias direction (PCA) (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-bias-direction-pca) <small>baseline</small> <small>flag FPR > 0.2</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1607.06520), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.696** | 0.990 | 0.539 | 0.560 | 1.000 | 1.000 |
| 10 | [Marginal probe (E -> A)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/marginal-probe) <small>baseline</small> <small>flag FPR > 0.2</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/1610.01644) | **0.679** | 0.998 | 0.570 | 0.469 | 0.489 | 0.622 |
| 11 | [Geometric bias analysis (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-geometric) <small>baseline</small> <small>flag FPR > 0.2</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/2505.12540), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.661** | 0.996 | 0.504 | 0.484 | 0.711 | 0.778 |
| 12 | [Per-dimension group tests (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-statistical) <small>baseline</small> <small>flag FPR > 0.2</small> | ShortKit-ML maintainers | [1](https://doi.org/10.1214/aoms/1177730491), [2](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x), [3](https://doi.org/10.64898/2026.04.29.26352053) | **0.656** | 0.877 | 0.583 | 0.509 | 1.000 | 0.989 |
| 13 | [Sufficient input subsets (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-sis) <small>baseline</small> <small>flag FPR > 0.2</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1810.03805), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.647** | 0.823 | 0.512 | 0.606 | 1.000 | 1.000 |
| 14 | [Demographic parity gap (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-demographic-parity) <small>baseline</small> <small>flag FPR > 0.2</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1412.3756), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.600** | 0.739 | 0.643 | 0.419 | 1.000 | 1.000 |

## Embedding mitigation

Train a task classifier on embeddings where the attribute predicts the label in training (rho >= 0.9) but not at test time (rho = 0.5). Scored by worst-group accuracy on the test set. Ranked by **Worst-group acc**. Benchmark v1.0. [Track rules](tracks.md#embedding-mitigation).

| # | Method | Authors | Paper | Worst-group acc | Mean acc | Gain over ERM | Residual leakage |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | [Group reweighting](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/group-reweighting) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/2110.14503) | **0.673** | 0.747 | 0.304 | – |
| 2 | [Group-balanced subsample (DFR-style)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/group-balanced-subsample) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/2110.14503) | **0.662** | 0.713 | 0.293 | – |
| 3 | [Adversarial debiasing (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-adversarial-debiasing) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1801.07593), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.482** | 0.683 | 0.113 | 0.856 |
| 4 | [INLP (iterative nullspace projection)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/inlp-projection) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/2004.07667) | **0.479** | 0.674 | 0.110 | 0.763 |
| 5 | [Contrastive debiasing (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-contrastive-debiasing) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/2203.01517), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.431** | 0.672 | 0.062 | 0.876 |
| 6 | [Last-layer retraining (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-last-layer-retraining) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/2204.02937), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.371** | 0.656 | 0.002 | – |
| 7 | [ERM (logistic regression)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/erm-logistic) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://doi.org/10.1109/72.788640) | **0.369** | 0.656 | 0.000 | – |

## Image detection

Given images, task labels and a CNN trained on them (no attribute labels), score whether the model relies on a hidden shortcut; optionally return per-image heatmaps, scored by whether their peak lands on the shortcut. Ranked by **AUROC**, after grouping: entries whose flag fires on at most 20% of null datasets first, then entries without a flag, then entries whose flag fires more often. Benchmark v1.0. [Track rules](tracks.md#image-detection).

| # | Method | Authors | Paper | AUROC | Spearman vs reliance | Pointing game | FPR | TPR |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | [Occlusion flip consistency](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/occlusion-flip-consistency) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/1311.2901) | **1.000** | 0.786 | 0.837 | 0.000 | 1.000 |
| 2 | [Occlusion hotspot](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/occlusion-hotspot) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/1311.2901) | **0.903** | 0.610 | 0.878 | – | – |
| 3 | [Input-gradient hotspot](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/input-gradient-hotspot) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/1312.6034) | **0.847** | 0.636 | 0.811 | – | – |
| 4 | [Saliency-peak consensus](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/saliency-peak-consensus) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://doi.org/10.1038/s41467-019-08987-4) | **0.819** | 0.278 | 0.878 | – | – |

## Image mitigation

Train an image classifier where a colored patch predicts the label in training but not at test time. Scored by worst-group accuracy on the test set. Ranked by **Worst-group acc**. Benchmark v1.0. [Track rules](tracks.md#image-mitigation).

| # | Method | Authors | Paper | Worst-group acc | Mean acc | Gain over ERM |
|---:|---|---|---|---:|---:|---:|
| 1 | [Explanation regularization, RRR (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-explanation-regularization) <small>baseline</small> | ShortKit-ML maintainers | [1](https://arxiv.org/abs/1703.03717), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.646** | 0.859 | 0.400 |
| 2 | [Group DRO (shared CNN)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/group-dro-cnn) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/1911.08731) | **0.545** | 0.800 | 0.300 |
| 3 | [Shortcut masking (ShortKit-ML)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/shortkit-shortcut-masking) <small>baseline</small> | ShortKit-ML maintainers | [1](https://doi.org/10.1145/3306618.3314293), [2](https://doi.org/10.64898/2026.04.29.26352053) | **0.529** | 0.772 | 0.284 |
| 4 | [Group-balanced sampling (shared CNN)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/group-balanced-sampling-cnn) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://arxiv.org/abs/2110.14503) | **0.437** | 0.805 | 0.192 |
| 5 | [ERM (shared CNN)](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/erm-cnn) <small>baseline</small> | ShortKit-ML maintainers | [paper](https://doi.org/10.1109/72.788640) | **0.245** | 0.757 | 0.000 |

<!-- LEADERBOARD:END -->

_Generated from `contrib/*/results.json` by `python -m shortcut_detect.leaderboard build`.
Machine-readable copy: [leaderboard.json](leaderboard.json)._
