# Basic usage examples

This page walks through the package on synthetic data: the paper's audit first, then the `ShortcutDetector` pipeline and the four classic methods one at a time. Every snippet runs as written.

## Jupyter notebooks

The notebooks are in the repository:

```bash
cd examples/01_basic_usage/
jupyter lab
```

## Example 1: the audit

The paper's measurement. `generate_linear_shortcut` plants the group in the first few dimensions; the audit also needs a task label, so we draw one that is correlated with the group.

```python
import numpy as np
from shortcut_detect import generate_linear_shortcut
from shortcut_detect.audit import AuditConfig, conditional_audit
from shortcut_detect.audit.report import audit_report, summary_text

X, group = generate_linear_shortcut(n_samples=1000, embedding_dim=128, shortcut_dims=5, seed=42)
rng = np.random.default_rng(0)
task = np.where(rng.random(len(group)) < 0.3, 1 - group, group)   # task label correlated with the group

cfg = AuditConfig(n_permutations=200, n_permutations_probe=500, seed=42)
result = conditional_audit(X, group, task, config=cfg)

probe = result.results["probe"]
print(f"AUROC {probe.statistic:.3f}, null {probe.null_mean:.3f} (SD {probe.null_sd:.3f}), p = {probe.p_value:.3f}")
print(summary_text(result, attribute="group", task="task"))
audit_report(result, "audit.html", title="Synthetic audit", attribute="group", task="task")
```

## Example 2: the ShortcutDetector pipeline

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["hbac", "probe", "statistical", "geometric"], seed=42)
detector.fit(X, group)
print(detector.summary())

detector.generate_report("basic_report.html", format="html")
```

`fit(embeddings, labels, group_labels=None)` takes the task label as `labels` and the attribute as `group_labels`; with `group_labels` omitted the group-based methods use `labels`. To keep both: `detector.fit(X, task, group_labels=group)`.

---

## Example 3: HBAC clustering

```python
from shortcut_detect import HBACDetector

hbac = HBACDetector(max_iterations=3, min_cluster_size=0.05, random_state=42)
hbac.fit(X, group)

verdict = hbac.shortcut_report_["has_shortcut"]
print(f"Shortcut detected: {hbac.shortcut_detected_} ({verdict['confidence']} confidence, {verdict['types']})")
print(f"Linear test accuracy: {hbac.shortcut_report_['linearity_score']['test_accuracy']:.2f}")
for cluster in hbac.shortcut_report_["cluster_purities"]:
    print(f"  cluster {cluster['cluster_id']}: {cluster['size']} rows, purity {cluster['purity']:.2f}")

hbac.visualize(method="pca", save_path="hbac_pca.png")
```

---

## Example 4: probe-based detection

The probe splits the data itself (`evaluation="holdout"`), trains on one part and scores the other.

```python
from sklearn.linear_model import LogisticRegression
from shortcut_detect import SKLearnProbe

probe = SKLearnProbe(LogisticRegression(max_iter=1000), metric="accuracy",
                     evaluation="holdout", test_size=0.2, random_state=42)
probe.fit(X, group)

print(f"Held-out accuracy: {probe.metric_value_:.2%}")
print(f"Flagged: {probe.shortcut_detected_}, risk {probe.get_report()['risk_level']}")
```

### Compare several classifiers

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": SVC(kernel="linear"),
    "RBF SVM": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, clf in classifiers.items():
    probe = SKLearnProbe(clf, metric="accuracy", evaluation="holdout", random_state=42)
    probe.fit(X, group)
    print(f"{name:20s}: {probe.metric_value_:.2%}")
```

---

## Example 5: statistical testing

```python
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from shortcut_detect import GroupDiffTest

test = GroupDiffTest(test=mannwhitneyu)
test.fit(X, group)
corrected = test.apply_correction(alpha=0.05, method="fdr_bh", verbose=False)

comparison = next(iter(corrected["significant_features"]))
significant = corrected["significant_features"][comparison] or []
q = corrected["corrected_pvalues"][comparison]
print(f"{comparison}: {len(significant)} of {X.shape[1]} dimensions significant after BH "
      f"({len(significant) / X.shape[1]:.1%})")
print("first significant dimensions:", significant[:10])

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["red" if i in set(significant) else "gray" for i in range(len(q))]
ax.bar(range(len(q)), -np.log10(q + 1e-300), color=colors)
ax.axhline(-np.log10(0.05), color="blue", linestyle="--", label="q = 0.05")
ax.set_xlabel("dimension")
ax.set_ylabel("-log10(BH-adjusted p)")
ax.legend()
plt.tight_layout()
plt.savefig("significant_dimensions.png", dpi=150)
```

---

## Example 6: geometric analysis

```python
from shortcut_detect import GeometricShortcutAnalyzer

analyzer = GeometricShortcutAnalyzer(n_components=5)
analyzer.fit(X, group)
print(analyzer.summary_)

pair = analyzer.bias_pairs_[0]
print(f"effect size {pair.effect_size:.2f} between groups {pair.groups}")

# project onto the direction between the two centroids
g0, g1 = pair.groups
c0, c1 = analyzer.group_stats_[g0]["centroid"], analyzer.group_stats_[g1]["centroid"]
direction = (c1 - c0) / np.linalg.norm(c1 - c0)
projections = X @ direction

fig, ax = plt.subplots(figsize=(10, 4))
for g in (g0, g1):
    ax.hist(projections[analyzer.group_stats_[g]["mask"]], bins=50, alpha=0.5, label=f"group {g}")
ax.set_xlabel("projection onto the bias direction")
ax.set_ylabel("count")
ax.set_title(f"effect size {pair.effect_size:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig("bias_projections.png", dpi=150)
```

---

## Example 7: with and without a shortcut

```python
from shortcut_detect import ShortcutDetector, generate_linear_shortcut, generate_no_shortcut

X_biased, g_biased = generate_linear_shortcut(n_samples=500, embedding_dim=64, shortcut_dims=5, seed=42)
X_clean, g_clean = generate_no_shortcut(n_samples=500, embedding_dim=64, seed=42)

for name, (X_, g_) in {"biased": (X_biased, g_biased), "clean": (X_clean, g_clean)}.items():
    detector = ShortcutDetector(methods=["probe", "statistical"], seed=42)
    detector.fit(X_, g_)
    results = detector.get_results()
    score = results["probe"]["results"]["metrics"]["metric_value"]
    n_sig = sum(len(v or []) for v in results["statistical"]["significant_features"].values())
    print(f"{name}: probe f1 {score:.2f} ({results['probe']['risk_value']}), "
          f"{n_sig} significant dimensions ({results['statistical']['risk_value']})")
```

---

## Next steps

- [Real data examples](real-data.md): chest X-ray embeddings
- [Advanced analysis](advanced.md): model comparison, intersectional attributes, debiasing
- [API reference](../api/shortcut-detector.md): full documentation
