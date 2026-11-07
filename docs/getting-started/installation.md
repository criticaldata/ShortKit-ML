# Installation

ShortKit-ML is the Python package `shortcut_detect`. Install it from source: the PyPI release 0.1.0 predates the `shortcut_detect.audit` module that the ShortKit-ML paper uses.

## Requirements

- Python 3.10, 3.11 or 3.12 (`requires-python = ">=3.10,<3.13"` in `pyproject.toml`)
- pip or [uv](https://github.com/astral-sh/uv)
- PyTorch 2.2 and torchvision 0.17 are core dependencies and come with the install; see [PyTorch with CUDA](#pytorch-with-cuda) for a GPU build

## Install from source

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd ShortKit-ML
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

Without uv:

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd ShortKit-ML
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install (`-e`) means a later `git pull` updates the package in place.

## Optional extras

The base install covers the audit (`shortcut_detect.audit`), the 19 registered detectors, the Gradio dashboard and the HTML, PDF and Markdown reports. Extras add:

| Extra | Adds |
|-------|------|
| `hf` | `transformers`, to embed raw text with Hugging Face models |
| `mcp` | the MCP server (`shortkit-ml-mcp`) |
| `jupyter` | JupyterLab, ipykernel and widgets for the example notebooks |
| `reporting` | `umap-learn`, for UMAP projections in reports |
| `docs` | MkDocs and the Material theme, to build this site |
| `dev` | pytest, black, ruff, mypy, pre-commit |
| `all` | everything above |

```bash
uv pip install -e ".[all]"       # or: pip install -e ".[all]"
uv pip install -e ".[hf,mcp]"
```

PyPI: `pip install shortkit-ml` installs release 0.1.0, which has the detectors and the dashboard but not the audit module. Use the source install until the next release.

## PyTorch with CUDA

The package pins `torch>=2.2.0,<2.3.0` and `torchvision>=0.17.0,<0.18.0`. For a GPU build, install the matching wheels from the PyTorch index:

```bash
# CUDA 12.1
uv pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
uv pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Replace `uv pip` with `pip` if you do not use uv.

## PDF export

PDF reports from `ShortcutDetector.generate_report(..., format="pdf")` go through WeasyPrint, which needs system libraries:

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0
    ```

=== "macOS"

    ```bash
    brew install pango gdk-pixbuf libffi
    ```

=== "Windows"

    Install the GTK3 runtime from [gtk.org](https://www.gtk.org/docs/installations/windows/)

The audit report, `shortcut_detect.audit.report.audit_report(result, "audit.pdf")`, writes its PDF with matplotlib and does not need these.

## Jupyter kernel

```bash
uv pip install -e ".[jupyter]"
python -m ipykernel install --user --name=shortcut_detect --display-name="Python (Shortcut_Detect)"
jupyter lab
```

## Check the install

```python
from shortcut_detect.audit import AuditConfig, conditional_audit
from shortcut_detect.audit.report import audit_report, summary_text
from shortcut_detect import ShortcutDetector, HBACDetector, SKLearnProbe
```

If the `shortcut_detect.audit` import fails, the installed copy is the PyPI 0.1.0 release rather than the source checkout.

## Troubleshooting

### Import errors

Reinstall with all extras:

```bash
uv pip install -e ".[all]" --force-reinstall
```

### CUDA not found

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## Next steps

- [Quick start](quickstart.md): run the audit, then the other methods
- [Interactive dashboard](dashboard.md): launch the web interface
- [Detection methods](../methods/overview.md): what each method does
