"""Step 5. Write the audit as a self-contained HTML page and a two-page PDF.

Usage: python examples/walkthrough/05_report.py
Reads output/walkthrough/celeba_audit.pkl from step 3.
"""

import os
import pickle
import textwrap

from shortcut_detect.audit.report import audit_report, summary_text

with open("output/walkthrough/celeba_audit.pkl", "rb") as f:
    result = pickle.load(f)

for path in ["output/walkthrough/celeba_audit.html", "output/walkthrough/celeba_audit.pdf"]:
    audit_report(result, path, title="CelebA, ResNet-50: Male given Blond_Hair", attribute="Male", task="Blond_Hair")
    print(f"wrote {path}  ({os.path.getsize(path) // 1024} KB)")

print()
print(textwrap.fill(summary_text(result, attribute="Male", task="Blond_Hair"), 96))
