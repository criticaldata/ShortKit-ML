"""Plugin discovery for detector method registration."""

from __future__ import annotations

import importlib

_METHOD_PLUGIN_MODULES: dict[str, str] = {
    "bias_direction_pca": "shortcut_detect.geometric.plugin",
    "causal_effect": "shortcut_detect.causal.plugin",
    "cav": "shortcut_detect.xai.plugin",
    "demographic_parity": "shortcut_detect.fairness.plugin",
    "early_epoch_clustering": "shortcut_detect.training.plugin",
    "equalized_odds": "shortcut_detect.fairness.plugin",
    "frequency": "shortcut_detect.frequency.plugin",
    "gce": "shortcut_detect.gce.plugin",
    "geometric": "shortcut_detect.geometric.plugin",
    "generative_cvae": "shortcut_detect.causal.plugin",
    "groupdro": "shortcut_detect.groupdro.plugin",
    "gradcam_mask_overlap": "shortcut_detect.xai.plugin",
    "hbac": "shortcut_detect.clustering.plugin",
    "intersectional": "shortcut_detect.fairness.plugin",
    "probe": "shortcut_detect.probes.plugin",
    "sis": "shortcut_detect.xai.plugin",
    "ssa": "shortcut_detect.ssa.plugin",
    "statistical": "shortcut_detect.statistical.plugin",
    "vae": "shortcut_detect.vae.plugin",
}


def available_method_plugins() -> set[str]:
    """Return the known built-in detector method names."""
    return set(_METHOD_PLUGIN_MODULES)


def load_method_plugin(method: str) -> None:
    """Import the plugin module responsible for registering ``method``."""
    module_name = _METHOD_PLUGIN_MODULES.get(method)
    if module_name is None:
        raise ValueError(f"Unknown detector method '{method}'.")
    importlib.import_module(module_name)


def load_all_method_plugins() -> None:
    """Import all built-in plugin modules.

    This is kept for callers that want eager registration, but the package now
    prefers per-method lazy loading to avoid importing optional heavy stacks
    during a plain ``import shortcut_detect``.
    """
    for module_name in sorted(set(_METHOD_PLUGIN_MODULES.values())):
        importlib.import_module(module_name)
