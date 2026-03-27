# Installation

This guide covers different installation options for ShortKit-ML.

## Requirements

- Python 3.9, 3.10, or 3.11
- pip or uv package manager
- PyTorch (torch) and torchvision (included in core install; see [PyTorch with CUDA](#pytorch-with-cuda) for GPU)

## Quick Install

### From PyPI (Recommended)

```bash
pip install shortcut-detect
```

### From Source

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd Shortcut_Detect
pip install -e .
```

## Installation Options

The library has several optional dependency groups for different use cases:

### Basic Installation

Core functionality with all detection methods:

```bash
pip install shortcut-detect
```

### With Dashboard

For the interactive Gradio web interface:

```bash
pip install "shortcut-detect[dashboard]"
```

### With Jupyter Support

For running examples in Jupyter notebooks:

```bash
pip install "shortcut-detect[jupyter]"
```

### With Reporting

For PDF/HTML report generation:

```bash
pip install "shortcut-detect[reporting]"
```

### With Hugging Face

For embedding generation from HuggingFace models:

```bash
pip install "shortcut-detect[hf]"
```

### With VAE

VAE shortcut detection requires `torch` and `torchvision`; both are core dependencies and included in the default install. To explicitly install VAE support:

```bash
pip install "shortcut-detect[vae]"
```

### Full Installation

Install everything (recommended for development):

```bash
pip install "shortcut-detect[all]"
```

## Using uv (Faster)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all dependencies
uv pip install -e ".[all]"
```

## PyTorch with CUDA

For GPU acceleration (optional):

```bash
# CUDA 12.1
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

## Jupyter Kernel Setup

To use the library in Jupyter notebooks:

```bash
# Install Jupyter support
pip install "shortcut-detect[jupyter]"

# Register the kernel
python -m ipykernel install --user --name=shortcut_detect --display-name="Python (Shortcut_Detect)"

# Launch Jupyter
jupyter lab
```

## Verify Installation

```python
import shortcut_detect
print(shortcut_detect.__version__)  # Should print: 0.1.0

# Test imports
from shortcut_detect import ShortcutDetector, HBACDetector, SKLearnProbe
print("All imports successful!")
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -e ".[all]" --force-reinstall
```

### CUDA Issues

If PyTorch doesn't detect your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Number of GPUs
```

### WeasyPrint Issues (PDF Generation)

WeasyPrint requires system dependencies:

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0
    ```

=== "macOS"

    ```bash
    brew install pango gdk-pixbuf libffi
    ```

=== "Windows"

    Install GTK3 runtime from [gtk.org](https://www.gtk.org/docs/installations/windows/)

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first shortcut detection
- [Interactive Dashboard](dashboard.md) - Launch the web interface
- [Detection Methods](../methods/overview.md) - Learn about each method
