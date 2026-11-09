# Tracks

All data is generated from seeds by the package itself, so there is nothing to download and
no licensed data is involved. Every metric is higher-is-better except FPR and residual leakage.

Each track has its own benchmark version (`TRACKS[key].version`), so changing one track's suite
only marks that track's results stale:

| Track | Version | Changes |
|---|---|---|
| embedding-detection | 1.1 | 30 null datasets per variant instead of 10 |
| embedding-mitigation | 1.0 | |
| image-detection | 1.0 | |
| image-mitigation | 1.0 | |

## Ranking the detection tracks

A detection entry returns a continuous `score` and, optionally, a yes/no `flag`. The score
is ranked by AUROC, which says nothing about the flag: a detector whose flag fires on every
dataset can still order its scores well. So on both detection tracks the table is ordered
in three groups, each by AUROC:

1. entries whose flag fires on at most 20% of null datasets
   (`shortcut_detect.leaderboard.MAX_FLAG_FPR`),
2. entries that return no flag,
3. entries whose flag fires on more than 20% of null datasets, marked *flag FPR > 0.2*.

A flag at a nominal α = 0.05 stays well under the cap. Leaving the flag out cannot lift an
entry above one whose flag is calibrated.

## embedding-detection

**Question.** Do embeddings `E` carry an attribute `A` beyond what the task label `Y` explains?

**Data.** `shortcut_detect.benchmark.conditional_synthetic.generate_conditional_synthetic` with n = 600, 64 dimensions,
`A` agreeing with `Y` 70% of the time, 4 task dimensions and 4 attribute dimensions, columns
shuffled per dataset. The attribute signal has strength δ ∈ {0 (null), 0.3, 0.6, 1.0} and one
of three forms:

| Variant | How `A` is carried | What it rewards |
|---|---|---|
| linear | mean shift of δ standard deviations | the easy case |
| xor | sign of the product of dimension pairs; no mean difference | non-linear detectors |
| variance | equal means, standard deviation 1 + δ in one group | detectors beyond the mean |

Null datasets (δ = 0) still have `A` correlated with `Y`, so `E` predicts `A` through the task
dimensions. A method that ignores `Y` fires on them.

**Suite.** Per variant, 10 seeds for each of the three strengths δ ∈ {0.3, 0.6, 1.0} and
30 null seeds (δ = 0), so nulls and shortcut datasets are equal in number: 3 × 60 = 180
datasets (quick: 2 seeds each). Version 1.0 had 10 nulls per variant; with that few, a
per-variant AUROC moved by about ±0.15 from seed noise alone. Even with 30, a method that
cannot see a variant scores 0.5 ± 0.075 (one standard deviation) on it, so read a per-variant
AUROC well under 0.5 as "blind to this variant", not as worse than chance. For example, the
conditional probe is linear and cannot see the XOR signal: it scores 0.41 on the public XOR
seeds (0.31 under version 1.0) and 0.55–0.59 on four sets of fresh seeds.

**Entry.** `detect(embeddings, attribute, task_labels, seed) -> float | {"score", "flag"}`.

**Metrics.** `auroc`: how well the score separates shortcut from null datasets, averaged over
the three variants (ranking metric, after the [flag groups](#ranking-the-detection-tracks));
per-variant AUROC. With a `flag`: `fpr` on nulls (should be near your nominal α), `tpr` on
shortcut datasets, and `tpr_delta_*` by strength.

## embedding-mitigation

**Question.** Can you train a task classifier that does not lean on the attribute?

**Data.** Linear variant, 64 dimensions, task effect 0.8. Training: n = 2000 with `A` agreeing
with `Y` at rho ∈ {0.9, 0.97}, attribute strength δ ∈ {1, 2}. Test: n = 4000 with rho = 0.5, so
the attribute says nothing about the label. 4 settings × 5 seeds (quick: 1 seed).

**Entry.** `fit(embeddings, attribute, task_labels, seed)` returning an object with
`predict(embeddings)`; optionally `transform(embeddings)`.

**Metrics.** `worst_group_acc`: test accuracy of the worst of the four (task, attribute) groups,
averaged over settings (ranking metric); `mean_acc` over the groups; `gain_over_erm`, the
worst-group gain over plain logistic regression on the same split; with `transform`,
`leakage_auroc`, the out-of-fold AUROC of a probe predicting `A` from `[transform(E), Y]` on the
test set (0.5 = nothing left).

## image-detection

**Question.** Does a trained model rely on a hidden shortcut, and where is it?

**Data.** 32×32 RGB images of one faint bar, horizontal or vertical (the task), on a noisy
background. The shortcut is a 4×4 red patch in one corner; the corner is fixed within a dataset
and chosen by its seed. The patch agrees with the label with probability rho ∈ {0.9, 0.95, 0.98, 1.0}
(shortcut-trained) or 0.5 (null datasets: the patch is in half the images but carries no
information). For each dataset the [shared CNN](#shared-cnn) is trained on 1500 images; the method
gets 400 held-out images, their labels and the trained model, and **no attribute labels**.
Seeds per strength: 9 nulls; 3, 2 and 2 at rho 0.9, 0.95 and 0.98; 6 at rho 1.0 (quick: 2 nulls,
one each at 0.9 and 1.0).

**Ground truth is measured, not assumed.** The training rho does not decide whether a given model
actually uses the patch: at rho 0.9 one seed may lean on it and another not. So for every dataset
the benchmark computes the model's *reliance*: the share of patched held-out images whose prediction
changes when the patch is painted gray. A shortcut-trained dataset counts as a **shortcut** case when
reliance ≥ 0.2, and as **ambiguous** otherwise. Below rho 1 the shared CNN usually learns the bar and
leans on the patch only a little, so most clear shortcut cases come from rho 1; the partial
strengths mostly test whether a score tracks *how much* the model relies on the patch. Null datasets are always nulls. Each dataset's rho,
reliance and label are in `details.per_dataset` of `results.json`.

**Entry.** `detect(images, task_labels, model, seed) -> float | {"score", "flag", "heatmaps"}`
with `images` of shape (n, 3, 32, 32) in [0, 1] and `heatmaps` of shape (n, 32, 32).

**Metrics.** `auroc` of the score, shortcut vs null datasets, ambiguous ones left out (ranking
metric, after the [flag groups](#ranking-the-detection-tracks)); `reliance_spearman`: rank correlation between the score and the measured reliance over
all datasets, ambiguous included (does the score track *how much* the model relies on the patch?);
with heatmaps, `pointing_game`: share of patched images in shortcut datasets whose heatmap peak
falls on the patch (within 2 pixels); with a flag, `fpr` on nulls and `tpr` on shortcut cases.

## image-mitigation

**Question.** Can you train an image classifier that ignores the patch?

**Data.** The same images. Training: n = 2000 with the patch agreeing with the label at
rho ∈ {0.9, 0.97}. Test: n = 2000 with rho = 0.5. 2 settings × 3 seeds (quick: 1 seed). Trained
without a usable patch (rho = 0.5) the shared CNN reached about 0.77 worst-group accuracy in our runs,
a rough ceiling for this backbone and budget.

**Entry.** `fit(images, attribute, task_labels, seed)` returning an object with
`predict(images)`.

**Metrics.** `worst_group_acc` (ranking metric), `mean_acc`, `gain_over_erm` against the shared
CNN trained with plain cross-entropy.

## Shared CNN

`shortcut_detect.leaderboard.image_models.make_cnn()`: three 3×3 conv blocks (16, 32, 64
channels, BatchNorm, ReLU, max-pool after the first two), global average pooling and a linear
head; `train_classifier()` trains it with Adam (lr 3e-3, batch 64, 8 epochs) on optionally
weighted cross-entropy. Mitigation methods may use any model, but using this one keeps the
comparison about the method.

## Reproducibility

The committed scores are the ones computed on the CI runner ([submitting](submit.md)).
Embedding-detection scores reproduce exactly (`verify` tolerance 1e-6). Embedding-mitigation
scores are checked with a tolerance of 0.01 and image-track scores with 0.02: entries that train
with PyTorch can move slightly between CPU models (by about 0.002 for the contrastive encoder),
even on the CI runners.
Scores on fresh seeds: `python -m shortcut_detect.leaderboard score contrib/<name> --seed-offset 1`.
