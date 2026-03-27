"""Example of using SSA detector with splits and extra_labels.

This example demonstrates how to use the SSA (Spectral Shift Analysis) detector
for semi-supervised shortcut detection. SSA requires:
1. splits: Dictionary with 'train_l' (labeled) and 'train_u' (unlabeled) indices
2. extra_labels (optional): Additional supervision signals like spurious labels
"""

import numpy as np

from shortcut_detect import ShortcutDetector
from shortcut_detect.datasets import generate_linear_shortcut

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with shortcuts
print("Generating synthetic data...")
embeddings, labels = generate_linear_shortcut(n_samples=400, embedding_dim=30, shortcut_dims=5)
protected_attr = np.random.randint(0, 2, 400)

print(f"Generated {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
print(f"Labels: {np.bincount(labels)}")
print(f"Protected attributes: {np.bincount(protected_attr)}")
print()

# Define splits for semi-supervised learning
# First 200 samples are labeled, rest are unlabeled
splits = {
    "train_l": np.arange(0, 200),  # Labeled indices (DL)
    "train_u": np.arange(200, 400),  # Unlabeled indices (DU)
}

print("Splits configuration:")
print(f"  Labeled (train_l): {len(splits['train_l'])} samples")
print(f"  Unlabeled (train_u): {len(splits['train_u'])} samples")
print()

# Create spurious labels for the labeled set
# -1 indicates unknown/unlabeled, non-negative values are actual labels
spurious_full = np.ones(400, dtype=int) * -1  # All start as -1 (unknown)
spurious_full[:200] = np.random.randint(0, 3, 200)  # Assign spurious labels to labeled data

extra_labels = {
    "spurious": spurious_full,
}

print("Extra labels configuration:")
print(f"  Spurious labels provided: {np.sum(spurious_full >= 0)} samples")
print(f"  Unlabeled spurious: {np.sum(spurious_full == -1)} samples")
print()

# Example 1: SSA alone
print("=" * 70)
print("Example 1: Using SSA alone")
print("=" * 70)

detector_ssa = ShortcutDetector(methods=["ssa"], seed=42)
detector_ssa.fit(
    embeddings,
    labels,
    group_labels=protected_attr,
    splits=splits,
    extra_labels=extra_labels,
)

print(detector_ssa.summary())
print()

# Example 2: SSA with other methods
print("=" * 70)
print("Example 2: Using SSA with other methods")
print("=" * 70)

detector_combined = ShortcutDetector(methods=["ssa", "probe", "hbac"], seed=42)
detector_combined.fit(
    embeddings,
    labels,
    group_labels=protected_attr,
    splits=splits,
    extra_labels=extra_labels,
)

print(detector_combined.summary())
print()

# Example 3: What happens without splits?
print("=" * 70)
print("Example 3: Using SSA without splits (should fail gracefully)")
print("=" * 70)

detector_no_splits = ShortcutDetector(methods=["ssa"], seed=42)
detector_no_splits.fit(
    embeddings,
    labels,
    group_labels=protected_attr,
    # No splits parameter provided
)

print(detector_no_splits.summary())
print()

# Example 4: Backward compatibility - other methods work without splits
print("=" * 70)
print("Example 4: Backward compatibility (no splits, other methods)")
print("=" * 70)

detector_backward_compat = ShortcutDetector(methods=["probe", "hbac"], seed=42)
detector_backward_compat.fit(
    embeddings,
    labels,
    group_labels=protected_attr,
    # No splits - should work fine for probe and hbac
)

print(detector_backward_compat.summary())
print()

print("=" * 70)
print("Examples complete!")
print("=" * 70)
print()
print("Note: SSA detector is currently a placeholder implementation.")
print("The actual SSA algorithm will be implemented by @amarzullo24.")
print("See shortcut_detect/ssa/detector.py for the placeholder code.")
