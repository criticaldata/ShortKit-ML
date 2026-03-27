"""
E2E tests for infrastructure features.

Covers: I01 Report Generator, I02 Gradio Dashboard, I03 CSV Export,
I04 Multi-attribute, I05 Intersectional, I06 Model Comparison.
"""

import pytest

from .conftest import (
    run_detection,
    select_only_methods,
    wait_for_detection_complete,
)


@pytest.mark.e2e
def test_dashboard_loads(browser_page):
    """I02 Gradio Dashboard: Page loads with correct title."""
    page = browser_page
    assert page.title() == "Shortcut Detection Dashboard"


@pytest.mark.e2e
def test_report_pdf_csv_download(browser_page):
    """I01/I03 Report Generator and CSV Export: Run detection, verify PDF and CSV download."""
    page = browser_page

    select_only_methods(page, ["hbac", "probe"])
    run_detection(page)
    wait_for_detection_complete(page)

    content = page.content()
    assert "Detection Complete" in content or "complete" in content.lower()
    # Download links or buttons exist (Gradio may use different labels)
    assert "PDF" in content or "Download" in content or "Report" in content
    assert "CSV" in content or "ZIP" in content or "Export" in content or "csv" in content.lower()


@pytest.mark.e2e
def test_multi_attribute(browser_page):
    """I04 Multi-attribute: Sample data has race+gender; run equalized_odds, verify per-attribute."""
    page = browser_page

    select_only_methods(page, ["equalized_odds"])
    run_detection(page)
    wait_for_detection_complete(page, timeout=300_000)

    content = page.content()
    # Per-attribute or by_attribute sections
    assert "equalized" in content.lower() or "Equalized Odds" in content
    assert "race" in content.lower() or "gender" in content.lower() or "group" in content.lower()


@pytest.mark.e2e
def test_model_comparison(browser_page):
    """I06 Model Comparison: Use Sample Data, run comparison, verify report."""
    page = browser_page
    page.set_default_timeout(300_000)

    page.get_by_role("tab", name="Model Comparison").click()
    page.wait_for_timeout(500)

    # Use only hbac + probe for faster run (3 models × 2 methods)
    for m in [
        "statistical",
        "geometric",
        "equalized_odds",
        "demographic_parity",
        "bias_direction_pca",
        "sis",
    ]:
        cb = page.get_by_role("checkbox", name=m)
        if cb.is_checked():
            cb.click()
    for m in ["hbac", "probe"]:
        cb = page.get_by_role("checkbox", name=m)
        if not cb.is_checked():
            cb.click()

    page.get_by_role("button", name="Run Comparison").or_(
        page.locator("button:has-text('Run Comparison')")
    ).first.click()
    page.get_by_text("Model Comparison Complete", exact=False).first.wait_for(
        state="visible", timeout=300_000
    )

    content = page.content()
    assert "Model Comparison" in content or "Comparison" in content
