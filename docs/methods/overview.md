# Detection methods overview

ShortKit-ML provides several complementary methods for detecting shortcuts in embedding spaces; 19 are registered as `ShortcutDetector` keys. Each method has different strengths and is sensitive to different types of biases.

## Why several methods?

Each method reads a different property of the embedding (cluster structure, per-dimension differences, centroid distance, a fitted direction), and the secondary scores are on interpretable scales. Agreement between methods is reported as a description. It is not a decision rule: in the paper's synthetic benchmark the detectors' flags were strongly correlated, and requiring several to agree lowered sensitivity without a useful gain in specificity.

## The paper's audit

The accompanying paper runs `shortcut_detect.audit.conditional_audit`. The risk levels on these pages play no part in it. Its single primary test is the conditional probe: an L2 logistic regression predicting the attribute from the embedding concatenated with the task label, scored by out-of-fold AUROC (5-fold CV stratified on attribute and task, grouped by patient when `groups` is given). The null is a conditional permutation: the attribute is shuffled within task-label strata, between patients where patient ids exist, K times; the p-value is (1 + number of null statistics at or above the observed) / (K + 1). The five secondary detectors (statistical, geometric, bias direction PCA, HBAC, frequency) share that null, and the six p-values of a cell are Benjamini-Hochberg adjusted. The number of flagged detectors is reported as a description only. `AuditConfig` defaults: `C=1.0`, `n_clusters=8`, `top_k_fraction=0.05`, `n_permutations=200`; the clinical audits set `n_permutations_probe=500`. SIS, the fairness gaps and the training-dynamics methods are outside the score. The audit measures shortcut potential, whether the attribute is recoverable from the embedding given the task; a positive result does not show that a classifier trained on the embedding uses the attribute. Reference: [Audit API](../api/audit.md).

## Methods comparison

| Method | What It Detects | Strengths | Limitations |
|--------|----------------|-----------|-------------|
| **HBAC** | Clustering by protected attributes | Fast, interpretable cluster purities | Assumes cluster structure |
| **Probe** | Information leakage | Direct measurement of predictability | Requires train/test split |
| **Statistical** | Dimension-wise differences | Identifies specific features | May miss non-linear patterns |
| **Geometric** | Bias directions in subspace | Needs only group labels | Requires sufficient samples per group |
| **Equalized Odds** | TPR/FPR disparities across groups | Interpretable fairness metric | Requires binary labels |
| **Demographic Parity** | Positive rate gaps across groups | Interpretable fairness metric | Requires binary labels |
| **Intersectional** | Fairness across demographic intersections | Catches compounded disparities (Buolamwini & Gebru 2018) | Requires 2+ attributes in extra_labels |
| **GroupDRO** | Worst-group performance gaps | Finds hard groups | Requires group + task labels |
| **Early-Epoch Clustering** | Early-epoch shortcut clusters | No group labels required | Needs early representations |
| **SIS** | Minimal sufficient input subsets | Model-agnostic, concise rationales | O(samples × dims) predictor calls |
| **GradCAM** | Attention on shortcut features | Visual interpretability | Requires model access, GPU |
| **SpRAy** | Heatmap clustering (Clever Hans) | Detects systematic attention artifacts | Requires heatmaps, clustering stability |
| **CAV** | Concept-level shortcut sensitivity (TCAV) | Interpretable concept tests | Requires concept/random activation sets |
| **Causal Effect** | Spurious attributes via causal effect estimation | Identifies attributes with near-zero causal effect | Requires attributes; optional interventional data |
| **VAE** | Disentangled latent shortcut dimensions | Discovers latent factors; human-interpretable traversals | Requires raw images; VAE training cost |
| **Frequency** | Concentrated class-separable dimensions | Embedding-only, no model access needed | Assumes linear probe sufficiency |
| **Generative CVAE** | Spurious attribute influence via counterfactuals | Generates embedding-space counterfactuals; measures causal effect | Requires binary group labels; seed-sensitive |
| **Shortcut Masking (M01)** | Mitigation: mask/inpaint shortcut regions | Data augmentation for retraining (Teso & Kersting 2019) | Requires shortcut regions/dims from detection |
| **Background Randomization (M02)** | Mitigation: swap foregrounds with random backgrounds | Reduces background shortcuts (Kwon et al. 2023) | Requires foreground masks; needs at least 2 images |
| **Adversarial Debiasing (M04)** | Mitigation: remove demographic encoding | Adversarial training (Zhang et al. 2018) | Requires protected labels; PyTorch |
| **Explanation Regularization (M05)** | Mitigation: penalize gradients on shortcut regions | RRR (Ross et al. 2017) | Requires model, images, labels, shortcut masks; PyTorch |
| **Last Layer Retraining (M06)** | Mitigation: retrain last layer on balanced subset | DFR (Kirichenko et al. 2023) | Requires task + group labels; sklearn |
| **Contrastive Debiasing (M07)** | Mitigation: contrastive learning to separate shortcuts | Correct-n-Contrast (Zhang et al. 2022) | Requires task + group labels; PyTorch |

## Decision matrix

Choose methods based on your use case:

```mermaid
graph TD
    A[Start] --> B{Have model access?}
    B -->|Yes| C{GPU available?}
    B -->|No| D[Use: Probe, Statistical, Geometric]
    C -->|Yes| E[Use: All methods including GradCAM]
    C -->|No| F[Use: HBAC, Probe, Statistical, Geometric]
```

## Quick recommendations

### For production audits

Use the paper's audit (`conditional_audit`) for the measurement. To explore alongside it, run several methods:

```python
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric', 'equalized_odds', 'demographic_parity', 'intersectional', 'groupdro']
)
```

### For quick screening

Use fast methods for initial assessment:

```python
detector = ShortcutDetector(
    methods=['statistical', 'probe']
)
```

### For deep analysis

Use all methods plus GradCAM for visual insights:

```python
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric']
)
# Plus GradCAM separately for visual analysis
gradcam = GradCAMHeatmapGenerator(model, target_layer)
```

## Interpretation framework

### Risk assessment

Each method applies a fixed rule to decide whether it flags a shortcut (the defaults, from the code):

| Method | Flagged when |
|--------|--------------|
| **HBAC** | two or more clusters have purity above 0.8, or few dimensions matter and the held-out linear accuracy is above 0.85 |
| **Probe** | the held-out metric (`f1` by default) exceeds `threshold=0.70`; risk `high` from 0.85 |
| **Statistical** | any dimension stays significant after the multiple-testing correction (`fdr_bh`, alpha 0.05) |
| **Geometric** | a centroid pair has effect size at or above 0.8, or a subspace pair has mean cosine at or above 0.85 |
| **Equalized odds** | the TPR or FPR gap between groups is at or above 0.1 |
| **Demographic parity** | the positive-rate gap between groups is at or above 0.1 |
| **GroupDRO** | average accuracy minus worst-group accuracy exceeds 0.10 |

These rules are heuristics for exploration. The paper's audit calibrates each statistic against a permutation null instead and reads nothing off these thresholds.

### Aggregated risk

The unified `ShortcutDetector` provides an aggregated assessment via the `summary()` method.
The aggregation strategy is controlled by the `condition_name` parameter:

| Condition | Strategy | Best For |
|-----------|----------|----------|
| `indicator_count` (default) | Counts total risk indicators. 2+ = HIGH, 1 = MODERATE, 0 = LOW | Simple audits, general use |
| `majority_vote` | Counts methods with indicators as votes | Conservative consensus-based screening |
| `weighted_risk` | Weights each detector by evidence strength (probe accuracy, stat significance, HBAC confidence, geometric effect size) | Graded scoring |
| `multi_attribute` | Cross-references risk across sensitive attributes; escalates when multiple attributes flag risk | Multi-demographic analysis (race + sex + age) |
| `meta_classifier` | Trained sklearn meta-classifier on detector features, or heuristic fallback | Advanced pipelines with synthetic training data |

```python
# Default aggregation
detector = ShortcutDetector(methods=['hbac', 'probe', 'statistical'])
detector.fit(embeddings, labels)
print(detector.summary())

# Weighted risk for nuanced scoring
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric'],
    condition_name='weighted_risk',
)
detector.fit(embeddings, labels)
print(detector.summary())

# Access individual method results
results = detector.get_results()
print(results['probe']['results']['metrics']['metric_value'])   # probe score (f1 by default)
print(results['hbac']['report']['has_shortcut']['exists'])      # HBAC decision
print(results['probe']['risk_value'], results['hbac']['risk_value'])
```

## Detailed method guides

Learn more about each method:

- [HBAC Clustering](hbac.md) - Hierarchical clustering analysis
- [Probe-based Detection](probe.md) - Classifier-based information leakage
- [Statistical Testing](statistical.md) - Feature-wise hypothesis tests
- [Geometric Analysis](geometric.md) - Subspace and direction analysis
- [Equalized Odds](equalized-odds.md) - TPR/FPR fairness analysis
- [Intersectional Analysis](intersectional-analysis.md) - Fairness across demographic intersections
- [SIS (Sufficient Input Subsets)](sis.md) - Minimal embedding dimensions for prediction (Carter et al. 2019)
- [Early-Epoch Clustering](early-epoch-clustering.md) - Early training shortcut clustering
- [GroupDRO](groupdro.md) - Worst-group robustness analysis
- [GradCAM](gradcam.md) - Visual attention analysis
- [SpRAy](spray.md) - Heatmap clustering for Clever Hans detection
- [CAV](cav.md) - Concept activation vector testing
- [Causal Effect](causal-effect.md) - Causal effect regularization for spurious attribute detection
- [VAE](vae.md) - VAE-based shortcut detection via latent disentanglement
- [Frequency](frequency.md) - Embedding-space frequency shortcut detection
- [Generative CVAE](generative-cvae.md) - Conditional VAE counterfactual shortcut detection
- [Shortcut Masking (M01)](shortcut-masking.md) - Data mitigation: mask/inpaint shortcut regions
- [Background Randomization (M02)](background-randomization.md) - Data mitigation: swap foregrounds with random backgrounds
- [Adversarial Debiasing (M04)](adversarial-debiasing.md) - Model mitigation: adversarial training to remove demographic encoding
- [Explanation Regularization (M05)](explanation-regularization.md) - Model mitigation: penalize input gradients on shortcut regions
- [Last Layer Retraining (M06)](last-layer-retraining.md) - Model mitigation: retrain last layer on group-balanced subset
- [Contrastive Debiasing (M07)](contrastive-debiasing.md) - Model mitigation: contrastive learning to separate shortcuts

## Method limitations

All methods have limitations you should be aware of:

1. **Sample Size** - Most methods need sufficient samples per group (n > 30)
2. **Linearity** - Some methods assume linear relationships
3. **Independence** - Assumes embeddings are i.i.d. samples; when several images come from one patient, pass patient ids as `groups` to `conditional_audit` so folds and permutations are patient-level
4. **Single Protected Attribute** - Most fairness methods assume one protected attribute; use [Intersectional Analysis](intersectional-analysis.md) for multiple attributes.
