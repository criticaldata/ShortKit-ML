"""Marginal probe: predict A from E alone, ignoring the task label.

Kept as a cautionary baseline: when A is correlated with Y, the task signal in
E is enough to predict A, so this probe fires on null datasets (high FPR).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

METADATA = {
    "name": "marginal-probe",
    "title": "Marginal probe (E -> A)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/1610.01644",
    "description": "Out-of-fold AUROC of a logistic probe predicting A from E only; flags at AUROC > 0.6.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    X = StandardScaler().fit_transform(embeddings)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        LogisticRegression(max_iter=2000), X, attribute, cv=cv, method="predict_proba"
    )[:, 1]
    auc = float(roc_auc_score(attribute, p))
    return {"score": auc, "flag": auc > 0.6}
