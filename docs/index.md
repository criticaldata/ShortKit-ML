---
hide:
  - navigation
  - toc
---

<p align="center" class="home-logo">
  <video autoplay muted loop playsinline controls preload="metadata" style="width:100%;border-radius:12px;border:1px solid #d9deea;box-shadow:0 8px 28px rgba(31,39,51,.10)" poster="assets/walkthrough/dashboard-audit-poster.jpg" src="assets/walkthrough/dashboard-audit.mp4"></video>
</p>

*Ninety seconds: install, bring your embeddings (any encoder, any modality), the dashboard from `python app.py` to the downloaded PDF, and the MCP call for agents.*

[![PyPI version](https://img.shields.io/pypi/v/shortkit-ml.svg)](https://pypi.org/project/shortkit-ml/)
[![CI](https://github.com/criticaldata/ShortKit-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/criticaldata/ShortKit-ML/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-criticaldata.github.io-blue.svg)](https://criticaldata.github.io/ShortKit-ML/)
[![Paper](https://img.shields.io/badge/paper-medRxiv-red.svg)](https://doi.org/10.64898/2026.04.29.26352053)
{ .home-badges }

ShortKit-ML measures the shortcut potential of a machine learning embedding with one calibrated audit and writes the result as an HTML or PDF report.
{ .home-lead }

The audit answers one question: does the embedding carry a sensitive attribute (sex, age group, race, site, scanner) beyond what the task label explains? A conditional probe is the primary test, five secondary detectors are reported as diagnostics, and all six are calibrated by a permutation null that shuffles the attribute within task-label strata, so the probe decides at p < 0.05 and nothing else does.

## Start here

<div class="grid cards home-cards" markdown>

-   :material-play-circle-outline:{ .home-icon .dot-blue } __Walkthrough__

    ---

    The whole audit on one public data set, from a fresh install to the PDF report, in five terminal recordings and a dashboard video, all replayable on your own data.

    [:octicons-arrow-right-24: Walkthrough](getting-started/walkthrough.md)

-   :material-download-outline:{ .home-icon .dot-teal } __Installation__

    ---

    Clone and install from source with `uv` or `pip`. The audit module is newer than the PyPI release, so source is the way in for now.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-robot-outline:{ .home-icon .dot-purple } __MCP server for agents__

    ---

    Nine tools over stdio. `run_audit` runs the audit on files you name in chat, from Claude Code, Claude Desktop or any other MCP client.

    [:octicons-arrow-right-24: MCP server](mcp-server.md)

-   :material-file-document-outline:{ .home-icon .dot-green } __The paper__

    ---

    ShortKit-ML: an open toolkit for measuring shortcut potential in medical imaging embeddings. Preprint on medRxiv, with the method, the synthetic calibration and the clinical audits.

    [:octicons-arrow-right-24: Read the preprint](https://doi.org/10.64898/2026.04.29.26352053)

</div>

## What the audit needs

| Array | Shape | Meaning |
|---|---|---|
| `E` | `(n, d)` float | one embedding per row, from any encoder, any width `d` |
| `A` | `(n,)` | the sensitive attribute |
| `Y` | `(n,)` | the task label the classifier would learn |
| `groups` | `(n,)`, optional | a unit id (patient, identity) when one unit has several rows |

## Run it

```python
import numpy as np
from shortcut_detect.audit import AuditConfig, conditional_audit
from shortcut_detect.audit.report import audit_report

E = np.load("embeddings.npy")   # (n, d), any encoder
A = np.load("attribute.npy")    # (n,)
Y = np.load("labels.npy")       # (n,)
result = conditional_audit(E, A, Y, config=AuditConfig(seed=42))  # groups=patient_ids when needed
print(result.results["probe"].p_value)
audit_report(result, "audit.pdf", attribute="sex", task="pneumonia")
```

The dashboard, the 19 exploratory detectors and the mitigation methods are documented under [Other package methods](methods/overview.md).

## Cite

```bibtex
@article{cajas2026shortkit,
  title={ShortKit-ML: an open toolkit for measuring shortcut potential in medical imaging embeddings},
  author={Cajas, Sebastian and Marzullo, Aldo and Kapadia, Sahil and Santos, Filipe and Ocampo Osorio, Felipe and Kong, Qingpeng and Quarta, Alessandro and Kuo, Po-Chih and Patel, Milit and Rojas Sillery, Raul Ignacio and Celi, Leo Anthony},
  journal={medRxiv},
  year={2026},
  doi={10.64898/2026.04.29.26352053}
}
```

MIT licence, see [LICENSE](https://github.com/criticaldata/ShortKit-ML/blob/main/LICENSE).
