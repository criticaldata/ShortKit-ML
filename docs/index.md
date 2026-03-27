# ShortKit-ML

**Detect shortcuts and biases in machine learning embedding spaces using HBAC, Probe, Statistical, and Geometric methods.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passing-success.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-82%25-green.svg)]()

---

## What is Shortcut Detection?

Machine learning models often learn **shortcuts** - spurious correlations between input features and labels that don't generalize. For example, a medical imaging model might learn to predict disease based on hospital equipment watermarks rather than actual pathology.

This library helps you detect such shortcuts in your model's embedding space before they cause real-world harm.

## Key Features

<div class="grid cards" markdown>

-   :dna: **HBAC Clustering**

    ---

    Hierarchical Bias-Aware Clustering detects if embeddings cluster by protected attributes.

    [:octicons-arrow-right-24: Learn more](methods/hbac.md)

-   :microscope: **Probe-based Detection**

    ---

    Tests if sensitive group information can be recovered from embeddings using classifiers.

    [:octicons-arrow-right-24: Learn more](methods/probe.md)

-   :bar_chart: **Statistical Testing**

    ---

    Identifies embedding dimensions with significant group differences using hypothesis tests.

    [:octicons-arrow-right-24: Learn more](methods/statistical.md)

-   :compass: **Geometric Analysis**

    ---

    Finds bias directions and prototype subspaces in embedding geometry.

    [:octicons-arrow-right-24: Learn more](methods/geometric.md)

</div>

## Quick Example

```python
from shortcut_detect import ShortcutDetector
import numpy as np

# Load your embeddings and labels
embeddings = np.load("embeddings.npy")  # (n_samples, embedding_dim)
labels = np.load("labels.npy")          # (n_samples,)

# Detect shortcuts using all methods
detector = ShortcutDetector(methods=['hbac', 'probe', 'statistical', 'geometric'])
detector.fit(embeddings, labels)

# Generate comprehensive report
detector.generate_report("report.html", format="html")
print(detector.summary())
```

**Output:**
```
======================================================================
UNIFIED SHORTCUT DETECTION SUMMARY
======================================================================
HIGH RISK: Multiple methods detected shortcuts
  - HBAC detected shortcuts (high confidence)
  - Probe accuracy 94.2% (high risk)
  - 3 group comparisons show significant differences
```

## Interactive Dashboard

Launch the Gradio web interface for interactive analysis:

```bash
python app.py
```

![Dashboard Preview](assets/dashboard-preview.png)

**Features:**

- Sample data: CheXpert medical imaging (2000 samples, 512-dim)
- Custom CSV upload
- PDF/HTML report export
- Interactive visualizations

## Installation

```bash
pip install shortcut-detect
```

Or install from source:

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd Shortcut_Detect
pip install -e ".[all]"
```

[:octicons-arrow-right-24: Full installation guide](getting-started/installation.md)

## Detection Methods Overview

| Method | What it detects | Speed | GPU Required |
|--------|----------------|-------|--------------|
| **HBAC** | Clustering by protected attributes | Fast | No |
| **Probe** | Information leakage via classifiers | Medium | Optional |
| **Statistical** | Dimension-wise group differences | Fast | No |
| **Geometric** | Bias directions & subspaces | Fast | No |
| **GradCAM** | Attention overlap with shortcuts | Slow | Yes |
| **SpRAy** | Heatmap clustering (Clever Hans) | Medium | Optional |

## Who Uses This?

This library is designed for:

- **ML Researchers** studying fairness and bias
- **Healthcare AI Teams** validating medical imaging models
- **ML Engineers** auditing production models
- **Data Scientists** exploring embedding quality

## License

MIT License - see [LICENSE](https://github.com/criticaldata/ShortKit-ML/blob/main/LICENSE) for details.
