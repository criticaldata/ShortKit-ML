"""
Pytest fixtures for E2E dashboard tests.

Requires: pip install playwright && playwright install chromium
Dashboard: Started automatically by dashboard_process fixture, or already running
at GRADIO_TEST_URL (e.g. http://127.0.0.1:7860).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Base URL when dashboard is already running (e.g. started manually or by CI)
BASE_URL = os.environ.get("GRADIO_TEST_URL", "http://127.0.0.1:7860")
DASHBOARD_PORT = 7860
STARTUP_TIMEOUT = 120  # CI runners can be slow; allow 2 min for Gradio to start
POLL_INTERVAL = 0.5


def _playwright_available():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401

        return True
    except ImportError:
        return False


def _is_dashboard_ready(url: str) -> bool:
    """Check if the dashboard is accepting connections and serving the app."""
    try:
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            # Gradio serves HTML; ensure we get actual content, not just a connection
            return "gradio" in body.lower() or "Shortcut" in body or len(body) > 100
    except Exception:
        return False


@pytest.fixture(scope="session")
def dashboard_process():
    """
    Start the Gradio dashboard in a subprocess if not already running.
    Yields the base URL; terminates process on teardown.
    """
    if _is_dashboard_ready(BASE_URL):
        yield BASE_URL
        return

    project_root = Path(__file__).parent.parent.parent
    app_path = project_root / "app.py"
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(DASHBOARD_PORT)

    proc = subprocess.Popen(
        [sys.executable, str(app_path)],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        elapsed = 0
        while elapsed < STARTUP_TIMEOUT:
            if _is_dashboard_ready(BASE_URL):
                yield BASE_URL
                return
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        pytest.fail(f"Dashboard did not become ready at {BASE_URL} within {STARTUP_TIMEOUT}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def dashboard_url(dashboard_process):
    """Base URL for the dashboard (from fixture or env)."""
    return dashboard_process


@pytest.fixture
def browser_page(dashboard_url):
    """
    Provide a Playwright page with the dashboard loaded.
    Requires playwright to be installed.
    """
    if not _playwright_available():
        pytest.skip(
            "playwright not installed; pip install playwright && playwright install chromium"
        )

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            # 5 min default for detection-heavy operations (CI can be slow)
            page.set_default_timeout(300_000)
            page.goto(dashboard_url, wait_until="load")
            # Wait for Gradio to hydrate: Run Detection button is more reliable than title
            page.get_by_role("button", name="Run Detection").wait_for(
                state="visible", timeout=30_000
            )
            assert (
                page.title() == "Shortcut Detection Dashboard"
            ), f"Expected 'Shortcut Detection Dashboard', got '{page.title()}'"
            yield page
        finally:
            browser.close()


# Helper functions for tests
DEFAULT_METHODS = [
    "hbac",
    "probe",
    "statistical",
    "geometric",
    "equalized_odds",
    "groupdro",
    "demographic_parity",
    "intersectional",
    "bias_direction_pca",
    "early_epoch_clustering",
]


def select_methods(page, methods: list[str]) -> None:
    """Click checkboxes to select the given detection methods."""
    for method in methods:
        checkbox = page.get_by_role("checkbox", name=method)
        if not checkbox.is_checked():
            checkbox.click()
        assert checkbox.is_checked()


def select_only_methods(page, methods: list[str]) -> None:
    """Uncheck defaults, then select only the given methods (faster runs)."""
    for m in DEFAULT_METHODS:
        if m not in methods:
            cb = page.get_by_role("checkbox", name=m)
            if cb.is_checked():
                cb.click()
    for m in methods:
        cb = page.get_by_role("checkbox", name=m)
        if not cb.is_checked():
            cb.click()
        assert cb.is_checked()


def run_detection(page) -> None:
    """Click the Run Detection button."""
    page.get_by_role("button", name="Run Detection").click()


def wait_for_detection_complete(page, timeout: int = 300_000) -> None:
    """Wait for the detection to complete (success banner appears)."""
    page.get_by_text("Detection Complete", exact=False).first.wait_for(
        state="visible", timeout=timeout
    )


def assert_report_contains(page, *strings: str) -> None:
    """Assert the report HTML contains all of the given strings."""
    content = page.content()
    for s in strings:
        assert s in content, f"Report did not contain expected text: {s}"
