# Generative CVAE Counterfactual Detection

The Generative CVAE detector trains a Conditional Variational Autoencoder (CVAE) on embeddings conditioned on a binary spurious attribute. It generates counterfactual embeddings by encoding with the original attribute and decoding with the flipped attribute, then measures how a probe classifier's predictions change.

## What It Detects

- Whether a spurious (group) attribute causally influences model predictions in embedding space.
- Large probe prediction shifts after counterfactual attribute flipping indicate shortcut reliance.

## How It Works

1. **Train CVAE**: A conditional VAE learns to reconstruct embeddings conditioned on the binary group label (spurious attribute).
2. **Train Attribute Predictor**: A small linear network learns to predict the group label from embeddings.
3. **Generate Counterfactuals**: For each sample, encode with the original group label and decode with the flipped label. Latent guidance optimizes the counterfactual to actually flip the predicted attribute while staying close to the original embedding.
4. **Evaluate Probe Shift**: A probe classifier (internal LogisticRegression or user-provided) scores both original and counterfactual embeddings. Large prediction shifts indicate the model relies on the spurious attribute.

## Required Inputs

- `embeddings`: `np.ndarray` `(n, d)` — representation space
- `group_labels`: `np.ndarray` `(n,)` — binary spurious attribute labels (0/1)

Optional:

- `labels`: `np.ndarray` `(n,)` — task labels (required for probe training; if omitted, detection is inconclusive)
- `probe_classifier`: Pre-trained sklearn-like classifier or callable

## Unified API Example

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["generative_cvae"])
detector.fit(embeddings=emb, labels=labels, group_labels=groups)

result = detector.get_results()["generative_cvae"]
print(result["metrics"])
print(result["shortcut_detected"])
```

## Direct API Example

```python
from shortcut_detect.causal import GenerativeCVEDetector

detector = GenerativeCVEDetector(epochs=50, random_state=42)
detector.fit(embeddings, group_labels, labels)

print(detector.results_["shortcut_detected"])
print(detector.results_["metrics"])
print(detector.summary())
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| `mean_delta` | Mean probe prediction shift (original - counterfactual) |
| `frac_large_change` | Fraction of samples with |delta| > 0.1 |
| `mean_cosine_similarity` | Average cosine similarity between original and counterfactual embeddings |
| `probe_accuracy` | Accuracy of the internal/external probe |

## Detection Rule

A shortcut is detected when **both**:

- `abs(mean_delta) > mean_delta_threshold` (default: 1e-4)
- `frac_large_change > frac_large_threshold` (default: 0.01)

## Interpretation

- **High risk**: Large prediction shifts indicate the model relies heavily on the spurious attribute.
- **Low risk**: Minimal prediction shifts suggest the spurious attribute has little influence.
- **Unknown**: No probe available (labels not provided).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | CVAE training epochs |
| `hidden` | 256 | Hidden layer size |
| `zdim` | 64 | Latent dimension |
| `guidance_steps` | 50 | Latent optimization steps for counterfactual generation |
| `guidance_weight` | 5.0 | Weight for attribute-flip loss |
| `proximity_weight` | 1.0 | Weight for staying close to original embedding |

## Limitations

- Requires binary group labels (0/1).
- Detection thresholds are sensitive to dataset size and noise level.
- CVAE quality depends on sufficient training data and appropriate hyperparameters.
- Results can vary across random seeds, especially with small datasets.
