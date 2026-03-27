# Playwright E2E Tests

End-to-end tests for the Gradio dashboard using Playwright.

## Setup

```bash
pip install -e ".[e2e]"
playwright install chromium
```

## Running Tests

**Option 1: Dashboard already running**

```bash
# In one terminal:
GRADIO_SERVER_PORT=7860 python app.py

# In another:
GRADIO_TEST_URL=http://127.0.0.1:7860 pytest tests/e2e/ -v -m e2e
```

**Option 2: Auto-start dashboard**

```bash
pytest tests/e2e/ -v -m e2e
```

The conftest will start the dashboard if it is not already running at `http://127.0.0.1:7860`.

## Test Coverage

| Feature | Test | Status |
|---------|------|--------|
| D19 CAV | test_cav_standalone | Skipped (file upload selectors) |
| D20 SIS | test_detection_sis | Pass |
| D21 Causal Effect | test_detection_causal_effect | Pass |
| D22 VAE | test_vae_standalone | Skipped (file upload selectors) |
| M01 Shortcut Masking | test_shortcut_masking_m01_embedding | Skipped |
| M02 Background Randomization | test_background_randomization_m02 | Skipped |
| M03 GroupDRO | test_detection_groupdro | Pass |
| M04 Adversarial Debiasing | test_adversarial_debiasing_m04 | Pass |
| M05 Explanation Regularization | — | Not implemented |
| M06 Last Layer Retraining | test_last_layer_retraining_m06 | Pass |
| M07 Contrastive Debiasing | test_contrastive_debiasing_m07 | Pass |
| I01 I02 I03 Report/Dashboard/CSV | test_report_pdf_csv_download | Pass |
| I04 Multi-attribute | test_multi_attribute | Pass |
| I05 Intersectional | test_detection_intersectional | Pass |
| I06 Model Comparison | test_model_comparison | Skipped (slow) |

## Using cursor-ide-browser MCP (Manual Testing)

For exploratory testing with the browser MCP:

1. Start the dashboard: `python app.py`
2. `browser_navigate` to `http://127.0.0.1:7860`
3. `browser_lock` to interact
4. `browser_snapshot` to inspect UI
5. `browser_click` / `browser_type` for interactions

## CI

The e2e job in `.github/workflows/tests.yml` installs Playwright and runs `pytest tests/e2e/ -v -m e2e`.
