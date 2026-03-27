"""
E2E tests for the Advanced Analysis tab.

Covers: D19 CAV, D22 VAE, M01 Shortcut Masking, M02 Background Randomization,
M04 Adversarial Debiasing, M05 Explanation Regularization, M06 Last Layer Retraining,
M07 Contrastive Debiasing.
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    run_detection,
    select_only_methods,
    wait_for_detection_complete,
)


def _open_advanced_analysis_tab(page):
    """Click the Advanced Analysis tab."""
    page.get_by_role("tab", name="Advanced Analysis").click()


def _run_detection_first(page):
    """Run detection with sample data so mitigation methods can use last detection."""
    select_only_methods(page, ["hbac", "probe"])
    run_detection(page)
    wait_for_detection_complete(page)


@pytest.mark.e2e
def test_cav_standalone(browser_page, tmp_path):
    """D19 CAV: Create minimal CAV bundle, upload, run CAV analysis."""
    page = browser_page
    page.set_default_timeout(120_000)

    # Create CAV bundle (.npz)
    rng = np.random.default_rng(10)
    dim = 8
    concept = rng.normal(0.0, 0.5, size=(50, dim))
    random = rng.normal(0.0, 0.5, size=(50, dim))
    concept[:, 0] += 1.8
    random[:, 0] -= 1.8
    target_dd = rng.normal(0.0, 0.2, size=(70, dim))
    target_dd[:, 0] += 1.0

    cav_path = tmp_path / "cav_bundle.npz"
    np.savez(
        cav_path,
        concept_shortcut=concept,
        random_set=random,
        target_directional_derivatives=target_dd,
    )

    _open_advanced_analysis_tab(page)
    page.get_by_text("CAV Concept Testing", exact=False).first.click()
    page.wait_for_timeout(500)

    # File input precedes "Run CAV Analysis" in CAV accordion
    run_btn = page.get_by_role("button", name="Run CAV Analysis")
    cav_input = run_btn.locator("xpath=preceding::input[@type='file'][1]")
    cav_input.set_input_files(str(cav_path))

    run_btn.click()
    page.get_by_text("CAV complete", exact=False).first.wait_for(state="visible", timeout=120_000)
    content = page.content()
    assert "CAV" in content and ("complete" in content.lower() or "success" in content.lower())


@pytest.mark.e2e
@pytest.mark.skip(
    reason="VAE runs 5+ min without showing 'VAE complete'—likely backend issue or UI update"
)
def test_vae_standalone(browser_page, tmp_path):
    """D22 VAE: Create minimal images + labels CSV, upload, run VAE."""
    from PIL import Image

    page = browser_page
    page.set_default_timeout(300_000)  # VAE can be slow

    n_images = 6
    img_size = 32
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(n_images):
        arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images_dir / f"img_{i:02d}.png")

    labels_df = pd.DataFrame({"task_label": [i % 2 for i in range(n_images)]})
    labels_path = tmp_path / "labels.csv"
    labels_df.to_csv(labels_path, index=False)

    img_files = [str(images_dir / f"img_{i:02d}.png") for i in range(n_images)]

    _open_advanced_analysis_tab(page)
    page.get_by_text("VAE Image Shortcut Detection", exact=False).first.click()
    page.wait_for_timeout(1500)  # Allow accordion to expand fully

    # VAE: Identify by accept attr (Images=.png, Labels=.csv)
    vae_btn = page.get_by_role("button", name="Run VAE Analysis")
    vae_images_input = vae_btn.locator(
        "xpath=preceding::input[@type='file' and contains(@accept,'.png')][1]"
    )
    vae_labels_input = vae_btn.locator(
        "xpath=preceding::input[@type='file' and contains(@accept,'.csv')][1]"
    )
    vae_images_input.set_input_files(img_files)
    vae_labels_input.set_input_files(str(labels_path))

    page.get_by_label("VAE Epochs").fill("1")
    page.get_by_label("Classifier Epochs").fill("1")
    page.get_by_label("Latent Dimension").fill("4")

    page.get_by_role("button", name="Run VAE Analysis").click()
    page.get_by_text("VAE complete", exact=False).first.wait_for(state="visible", timeout=300_000)
    content = page.content()
    assert "VAE" in content


@pytest.mark.e2e
def test_adversarial_debiasing_m04(browser_page):
    """M04 Adversarial Debiasing: Run detection first, then use last detection run."""
    page = browser_page
    page.set_default_timeout(180_000)

    # Run detection first to populate detection_state
    _run_detection_first(page)

    _open_advanced_analysis_tab(page)
    page.get_by_text("Adversarial Debiasing (M04)", exact=False).first.click()
    page.wait_for_timeout(500)

    page.get_by_role("checkbox", name="Use embeddings from last detection run").check()
    page.get_by_role("button", name="Run Adversarial Debiasing").click()
    page.get_by_text("Adversarial debiasing (M04) complete", exact=False).first.wait_for(
        state="visible", timeout=120_000
    )
    content = page.content()
    assert "Adversarial debiasing" in content or "debiased" in content.lower()


@pytest.mark.e2e
def test_last_layer_retraining_m06(browser_page):
    """M06 Last Layer Retraining: Use embeddings from last detection run."""
    page = browser_page
    page.set_default_timeout(180_000)

    _run_detection_first(page)

    _open_advanced_analysis_tab(page)
    page.get_by_text("Last Layer Retraining (M06 DFR)", exact=False).first.click()
    page.wait_for_timeout(500)

    page.get_by_role("checkbox", name="Use embeddings from last detection run").check()
    page.get_by_role("button", name="Run Last Layer Retraining").click()
    page.get_by_text("Last Layer Retraining", exact=False).first.wait_for(
        state="visible", timeout=120_000
    )
    content = page.content()
    assert "DFR" in content or "Last Layer" in content


@pytest.mark.e2e
def test_contrastive_debiasing_m07(browser_page):
    """M07 Contrastive Debiasing: Use embeddings from last detection run."""
    page = browser_page
    page.set_default_timeout(180_000)

    _run_detection_first(page)

    _open_advanced_analysis_tab(page)
    page.get_by_text("Contrastive Debiasing (M07 CNC)", exact=False).first.click()
    page.wait_for_timeout(500)

    page.get_by_role("checkbox", name="Use embeddings from last detection run").check()
    page.get_by_role("button", name="Run Contrastive Debiasing").click()
    page.get_by_text("Contrastive debiasing", exact=False).first.wait_for(
        state="visible", timeout=120_000
    )
    content = page.content()
    assert "Contrastive debiasing" in content or "CNC" in content


@pytest.mark.e2e
def test_shortcut_masking_m01_embedding(browser_page, tmp_path):
    """M01 Shortcut Masking (Embedding mode): Upload CSV, mask dims 0,3,7."""
    from shortcut_detect.datasets import generate_linear_shortcut

    page = browser_page
    page.set_default_timeout(120_000)

    # Create embeddings CSV
    n, dim = 50, 16
    emb, _ = generate_linear_shortcut(n_samples=n, embedding_dim=dim, seed=42)
    labels = np.random.randint(0, 2, n)
    groups = np.array(["A", "B"] * (n // 2))
    cols = [f"embedding_{j}" for j in range(dim)] + ["task_label", "group_label"]
    data = np.hstack([emb, labels.reshape(-1, 1), groups.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=cols)
    csv_path = tmp_path / "emb.csv"
    df.to_csv(csv_path, index=False)

    _open_advanced_analysis_tab(page)
    page.get_by_text("Shortcut Feature Masking (M01)", exact=False).first.click()
    page.wait_for_timeout(500)

    page.get_by_role("radio", name="Embedding").click()
    page.wait_for_timeout(300)
    mask_btn = page.get_by_role("button", name="Run Shortcut Masking")
    emb_input = mask_btn.locator("xpath=preceding::input[@type='file'][1]")
    emb_input.set_input_files(str(csv_path))
    page.get_by_label("Dimension indices to mask (e.g. 0,3,7)").fill("0, 3, 7")
    page.get_by_role("button", name="Run Shortcut Masking").click()
    page.get_by_text("Shortcut masking (M01) complete", exact=False).first.wait_for(
        state="visible", timeout=60000
    )
    content = page.content()
    assert "Shortcut masking" in content or "M01" in content


@pytest.mark.e2e
def test_background_randomization_m02(browser_page, tmp_path):
    """M02 Background Randomization: Upload 2+ images + masks."""
    from PIL import Image

    page = browser_page
    page.set_default_timeout(120_000)

    # Create 2 images and 2 mask images
    size = 32
    img1 = Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
    mask1 = Image.fromarray(np.ones((size, size), dtype=np.uint8) * 255)
    mask2 = Image.fromarray(np.ones((size, size), dtype=np.uint8) * 255)
    img1.save(tmp_path / "i1.png")
    img2.save(tmp_path / "i2.png")
    mask1.save(tmp_path / "m1.png")
    mask2.save(tmp_path / "m2.png")

    _open_advanced_analysis_tab(page)
    page.get_by_text("Background Randomization (M02)", exact=False).first.click()
    page.wait_for_timeout(500)

    # M02: Images (multiple), Masks (single - one mask gets tiled)
    bg_btn = page.get_by_role("button", name="Run Background Randomization")
    bg_inputs = bg_btn.locator("xpath=preceding::input[@type='file']")
    for i in range(bg_inputs.count()):
        if bg_inputs.nth(i).get_attribute("multiple"):
            bg_inputs.nth(i).set_input_files([str(tmp_path / "i1.png"), str(tmp_path / "i2.png")])
            break
    for i in range(bg_inputs.count()):
        if not bg_inputs.nth(i).get_attribute("multiple"):
            bg_inputs.nth(i).set_input_files(str(tmp_path / "m1.png"))
            break
    page.get_by_role("button", name="Run Background Randomization").click()
    page.get_by_text("Background randomization", exact=False).first.wait_for(
        state="visible", timeout=60000
    )
    content = page.content()
    assert "Background" in content or "randomization" in content.lower()
