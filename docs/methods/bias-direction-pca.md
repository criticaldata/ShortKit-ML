# Embedding Bias Direction (PCA)

This method extracts a bias direction from group prototypes using PCA and measures
the projection gap across groups. Large gaps indicate systematic embedding bias.

## Usage

```python
from shortcut_detect.geometric import BiasDirectionPCADetector

detector = BiasDirectionPCADetector()
detector.fit(embeddings, group_labels)
print(detector.report_)
```

## Outputs
- `projection_gap`: max–min projection across groups
- `explained_variance`: variance explained by the bias direction
- `group_projections`: per-group projection means and support

## Reference
Bolukbasi et al. 2016: https://arxiv.org/abs/1607.06520
