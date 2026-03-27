# Contributing

We welcome contributions to ShortKit-ML! This guide explains how to get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Adding New Detection Methods](#adding-new-detection-methods)
- [Reporting Issues](#reporting-issues)

---

## Development Setup

### Clone the Repository

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd Shortcut_Detect
```

### Create Virtual Environment

```bash
# Using uv (recommended)
uv venv --python 3.10
source .venv/bin/activate

# Or using venv
python -m venv .venv
source .venv/bin/activate
```

### Install Development Dependencies

```bash
# Install all dependencies including dev tools
pip install -e ".[all]"

# Or using uv
uv pip install -e ".[all]"
```

### System Dependencies (Optional Features)

Some features require system libraries that cannot be installed via pip:

**PDF Report Generation** (weasyprint):

```bash
# macOS
brew install pango gdk-pixbuf libffi

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev

# Fedora/RHEL
sudo dnf install pango gdk-pixbuf2 libffi-devel
```

If you don't need PDF export, you can skip this—HTML and Markdown reports work without these dependencies.

### Install Pre-commit Hooks

```bash
pre-commit install
```

## Code Style

We use the following tools for code quality:

- **Black** - Code formatting
- **Ruff** - Linting
- **MyPy** - Type checking

### Running Formatters

```bash
# Format code
black shortcut_detect/ tests/

# Lint
ruff check shortcut_detect/ tests/

# Type check
mypy shortcut_detect/
```

### Configuration

See `pyproject.toml` for tool configurations.

### Method Implementation Conventions

To keep detection methods consistent across contributors, follow these project conventions:

- **File naming:** Prefer `{method}_detector.py` for modules whose primary purpose is a single detector class.
- **Config classes:** For detectors with multiple constructor parameters, use a frozen dataclass config (for example `MyMethodConfig`) and expose it as an optional `config=` argument.
- **Probe reuse:** Reuse shared probe implementations (`SKLearnProbe` / `TorchProbe`) instead of re-implementing standalone probe training logic inside method modules.
- **Shared validation/helpers:** Reuse common utilities (for example `validate_embeddings_labels`) instead of duplicating shape/length validation helpers across files.

If a method needs a deviation from these rules, document the reason in the module docstring and PR description.

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (requires pytest-cov)
pytest tests/ --cov=shortcut_detect --cov-report=html
```

> **Note:** Coverage requires `pytest-cov` which is included when you install with `pip install -e ".[all]"` or `pip install -e ".[dev]"`.

```bash
# Run specific test file
pytest tests/test_probes.py -v

# Run specific test
pytest tests/test_probes.py::test_sklearn_probe -v
```

### Writing Tests

Tests are in the `tests/` directory. Use pytest fixtures for common setup:

```python
# tests/test_my_feature.py
import pytest
import numpy as np
from shortcut_detect import ShortcutDetector

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_detector_fit(sample_data):
    """Test that detector fits without error."""
    X, y = sample_data
    detector = ShortcutDetector(methods=['probe'])
    detector.fit(X, y)
    assert 'probe' in detector.results_
```

## Documentation

### Building Docs Locally

You can build docs in a separate virtual environment to avoid conflicts with your main development environment:

```bash
# Create a separate docs environment (recommended)
python -m venv .venv-docs
source .venv-docs/bin/activate  # Windows: .venv-docs\Scripts\activate

# Install only docs dependencies (no need for the full package)
pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions

# Serve docs locally
mkdocs serve
# Visit http://127.0.0.1:8000

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

Alternatively, install docs dependencies in your main environment:

```bash
pip install -e ".[docs]"
mkdocs serve
```

### Writing Documentation

- Documentation is in `docs/` using Markdown
- API docs auto-generated from docstrings via mkdocstrings
- Follow Google docstring style

Example docstring:

```python
def fit(self, embeddings: np.ndarray, group_labels: np.ndarray) -> "ShortcutDetector":
    """Fit the shortcut detector on embeddings.

    Args:
        embeddings: Array of shape (n_samples, n_features) containing
            the embedding vectors to analyze.
        group_labels: Array of shape (n_samples,) containing the
            protected attribute labels for each sample.

    Returns:
        Self, for method chaining.

    Raises:
        ValueError: If embeddings and group_labels have mismatched lengths.

    Example:
        >>> detector = ShortcutDetector()
        >>> detector.fit(embeddings, group_labels)
        >>> print(detector.summary())
    """
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Commit and Push

```bash
git add .
git commit -m "Add my new feature"
git push origin feature/my-new-feature
```

Pre-commit hooks will automatically run **Black** (formatting) and **Ruff** (linting) on your staged files.

### 4. Create a Pull Request

Open a Pull Request on GitHub. The following will happen automatically:

- **CI Pipeline**: GitHub Actions runs the full test suite on Python 3.10, 3.11, and 3.12
- **Status Checks**: PR must pass all CI checks before merge
- **Code Review**: PRs require approval from a team member

### 5. Address Feedback

- Fix any CI failures
- Respond to review comments
- Push additional commits as needed

Once approved and all checks pass, your PR can be merged.

## Adding New Detection Methods

> **Start from the template.** Copy `shortcut_detect/detector_template.py`
> into your method's package and rename the classes.  The template contains
> inline comments for every required step and produces valid Python that
> you can run immediately.

### Overview

A detection method in this library consists of two pieces:

| Piece | Base class | Purpose |
|---|---|---|
| **Standalone detector** | `DetectorBase` | Runs the algorithm, stores results in the standardized `results_` dict |
| **Builder / runner** | `BaseDetector` | Constructs the detector and integrates it with the `ShortcutDetector` orchestrator |

Both are demonstrated in the template file.

### Required `results_` schema

Every detector must populate `results_` via `_set_results()`.  The schema
is:

```python
{
    "method": str,                # unique snake_case identifier
    "shortcut_detected": bool | None,  # True / False / None
    "risk_level": "low" | "moderate" | "high" | "unknown",
    "metrics": {                  # small scalar values only
        "score": 0.42,
    },
    "notes": str,                 # human-readable explanation
    "metadata": {                 # config, dataset info
        "threshold": 0.5,
        "n_samples": 1000,
    },
    # optional:
    "report": { ... },            # detailed structured output
    "details": { ... },           # large / auxiliary data
}
```

Guidelines:

- **metrics** -- keep small and scalar (floats, ints, short strings).
  Arrays, tables, and large objects belong in `report` or `details`.
- **risk_level** -- use `"unknown"` if the method cannot assess risk.
  The legacy value `"medium"` is automatically mapped to `"moderate"`.
- **shortcut_detected** -- use `None` if detection logic is not yet
  implemented.

### Step-by-step process

#### 1. Copy the template

```bash
# Create your method package
mkdir -p shortcut_detect/my_method
cp shortcut_detect/detector_template.py shortcut_detect/my_method/detector.py
```

#### 2. Implement the standalone detector

In `shortcut_detect/my_method/detector.py`, rename `MyMethodDetector` and
fill in the `fit()` method.  The key contract inside `fit()` is:

```python
def fit(self, embeddings, labels, group_labels):
    # 1. Validate inputs
    # 2. Run your detection algorithm
    # 3. Determine shortcut_detected (bool | None)
    # 4. Assess risk_level (RiskLevel enum or string)
    # 5. Call self._set_results(...)
    # 6. Set self._is_fitted = True
    # 7. Return self
```

#### 3. Implement the builder (optional but recommended)

The builder lets the `ShortcutDetector` orchestrator run your method
alongside others.  Implement `build()` (returns an unfitted detector) and
`run()` (fits it and returns a summary dict).

The dict returned by `run()` must have this shape:

```python
{
    "detector": detector,          # the fitted DetectorBase instance
    "results": detector.results_,  # the standardized results_ dict
    "summary_title": "My Method",  # short title for reports
    "summary_lines": [...],        # list of human-readable lines
    "risk_indicators": [...],      # list of risk strings (can be empty)
    "success": True,               # False if the method failed
}
```

#### 4. Register with the package

```python
# shortcut_detect/my_method/__init__.py
from .detector import MyMethodDetector

__all__ = ["MyMethodDetector"]
```

```python
# shortcut_detect/__init__.py  (add to existing exports)
from .my_method import MyMethodDetector
```

#### 5. Integrate with ShortcutDetector

Register your builder in `shortcut_detect/unified.py` so users can
request it by name:

```python
# In _run_method():
if method == "my_method":
    builder = MyMethodBuilder(seed=self.seed, kwargs=self.my_method_params)
    return builder.run(embeddings, labels, group_labels, ...)
```

#### 6. Add tests

```python
# tests/test_my_method.py
import numpy as np
from shortcut_detect.my_method import MyMethodDetector

def test_my_method_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    y = rng.integers(0, 2, 100)
    g = rng.integers(0, 3, 100)

    detector = MyMethodDetector(threshold=0.5)
    detector.fit(X, y, g)

    assert detector._is_fitted
    assert detector.results_["method"] == "my_method"
    assert "metrics" in detector.results_
    assert detector.results_["risk_level"] in ("low", "moderate", "high", "unknown")
    assert isinstance(detector.shortcut_detected_, (bool, type(None)))
```

#### 7. Add documentation

- Create `docs/methods/my-method.md` -- conceptual overview and usage
  examples.
- Create `docs/api/my-method.md` -- API reference (auto-generated from
  docstrings via mkdocstrings).
- Update `mkdocs.yml` navigation to include both pages.

## Reporting Issues

### Bug Reports

Include:

- Python version
- Library version
- Minimal reproduction code
- Full error traceback
- Expected vs actual behavior

### Feature Requests

Include:

- Use case description
- Proposed API design
- Any relevant papers/references

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open a [GitHub Issue](https://github.com/criticaldata/ShortKit-ML/issues)
- Check existing issues and PRs first

Thank you for contributing!
