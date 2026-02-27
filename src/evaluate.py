import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_binary_classifier(y_true, y_proba, threshold=0.5):
    """
    Évaluation complète d’un classifieur binaire.
    On transforme les probabilités en classes via un seuil donné.
    """

    # Conversion des probabilités en prédictions binaires
    y_pred = (y_proba >= threshold).astype(int)

    out = {}
    out["threshold"] = float(threshold)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    out["pr_auc"] = float(average_precision_score(y_true, y_proba))

    # Matrice de confusion pour analyse des erreurs
    cm = confusion_matrix(y_true, y_pred)
    out["confusion_matrix"] = cm.tolist()
    out["tn"], out["fp"], out["fn"], out["tp"] = map(int, cm.ravel())

    return out


def pretty_print_metrics(m):
    """
    Affichage formaté des métriques d’évaluation.
    """

    print("\n=== Évaluation (Baseline Layer 0) ===")
    print(f"Seuil      : {m['threshold']:.2f}")
    print(f"Accuracy   : {m['accuracy']:.4f}")
    print(f"Precision  : {m['precision']:.4f}")
    print(f"Recall     : {m['recall']:.4f}")
    print(f"F1-score   : {m['f1']:.4f}")
    print(f"ROC-AUC    : {m['roc_auc']:.4f}")
    print(f"PR-AUC     : {m['pr_auc']:.4f}")
    print("Matrice de confusion [[TN, FP],[FN, TP]]:")
    print(np.array(m["confusion_matrix"]))

from sklearn.metrics import confusion_matrix

def compute_cost_from_cm(cm, c_fn=5.0, c_fp=1.0):
    """
    Calcule un coût financier à partir d'une matrice de confusion.
    Coût = c_fn * FN + c_fp * FP
    """
    tn, fp, fn, tp = cm.ravel()
    return float(c_fn * fn + c_fp * fp)

def find_optimal_threshold(y_true, y_proba, c_fn=5.0, c_fp=1.0, t_min=0.05, t_max=0.95, n_grid=181):
    """
    Recherche du seuil optimal minimisant le coût asymétrique (FN vs FP).
    - y_proba : probabilités de la classe positive (défaut)
    - n_grid  : nombre de points sur la grille de seuils
    """
    thresholds = np.linspace(t_min, t_max, n_grid)
    best = {"threshold": None, "cost": np.inf, "cm": None}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cost = compute_cost_from_cm(cm, c_fn=c_fn, c_fp=c_fp)

        if cost < best["cost"]:
            best["threshold"] = float(t)
            best["cost"] = float(cost)
            best["cm"] = cm

    return best
