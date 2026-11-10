"""ShortKit-ML's conditional probe: the primary test of the paper's audit.

An L2 logistic regression predicts A from [E, Y]; its out-of-fold AUROC is
compared with a null that shuffles A within Y strata. Score = observed AUROC
minus the null mean; flag = permutation p < 0.05.
"""

from shortcut_detect.audit import AuditConfig, conditional_audit

METADATA = {
    "name": "shortkit-conditional-probe",
    "title": "Conditional probe (ShortKit-ML audit)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://doi.org/10.64898/2026.04.29.26352053",
    "code": "https://github.com/criticaldata/ShortKit-ML",
    "description": "Logistic probe A ~ [E, Y], out-of-fold AUROC against a within-Y permutation null.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    cfg = AuditConfig(detectors=("probe",), n_permutations=100, cv_repeats=1, seed=seed)
    r = conditional_audit(embeddings, attribute, task_labels, config=cfg).results["probe"]
    return {"score": r.statistic - r.null_mean, "flag": r.p_value < 0.05}
