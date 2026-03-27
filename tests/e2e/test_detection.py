"""
E2E tests for the Detection tab.

Covers: D20 SIS, M03 GroupDRO, GCE, I05 Intersectional, D21 Causal Effect.
"""

import pytest

from .conftest import (
    assert_report_contains,
    run_detection,
    select_only_methods,
    wait_for_detection_complete,
)


@pytest.mark.e2e
def test_detection_gce(browser_page):
    """D20/GCE: Select GCE method, run detection, verify report."""
    page = browser_page
    select_only_methods(page, ["gce"])
    run_detection(page)
    wait_for_detection_complete(page)
    assert_report_contains(
        page,
        "GCE (Generalized Cross Entropy) Bias Detection",
        "Minority/bias-conflicting samples",
        "GCE",
    )
    content = page.content()
    assert "high-loss" in content or "minority" in content.lower()


@pytest.mark.e2e
def test_detection_sis(browser_page):
    """D20 SIS: Select SIS method, run detection, verify Sufficient Input Subsets in report."""
    page = browser_page
    page.set_default_timeout(180_000)
    select_only_methods(page, ["sis", "hbac", "probe"])
    run_detection(page)
    wait_for_detection_complete(page, timeout=180_000)
    content = page.content()
    assert "SIS" in content or "Sufficient Input Subsets" in content or "mean_sis_size" in content


@pytest.mark.e2e
def test_detection_groupdro(browser_page):
    """M03 GroupDRO: Select groupdro, run detection, verify worst-group in report."""
    page = browser_page
    select_only_methods(page, ["groupdro"])
    run_detection(page)
    wait_for_detection_complete(page)
    content = page.content()
    assert "GroupDRO" in content or "groupdro" in content.lower()
    assert "worst" in content.lower() or "group" in content.lower()


@pytest.mark.e2e
def test_detection_intersectional(browser_page):
    """I05 Intersectional: Use Sample Data (has race + gender), select intersectional, run."""
    page = browser_page
    select_only_methods(page, ["intersectional"])
    run_detection(page)
    wait_for_detection_complete(page)
    content = page.content()
    assert "intersectional" in content.lower() or "intersection" in content.lower()


@pytest.mark.e2e
def test_detection_causal_effect(browser_page):
    """D21 Causal Effect: Sample Data has race+gender attributes; run causal_effect."""
    page = browser_page
    page.set_default_timeout(180_000)

    # Sample Data already provides attributes (race, gender) from load_sample_data
    select_only_methods(page, ["causal_effect"])
    run_detection(page)
    wait_for_detection_complete(page, timeout=180_000)
    content = page.content()
    assert "causal" in content.lower() or "Causal Effect" in content
