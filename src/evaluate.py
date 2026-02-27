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
