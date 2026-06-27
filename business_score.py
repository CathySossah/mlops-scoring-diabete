import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Score métier ─────────────────────────────────────────────────────────────
# Dans le contexte diabète :
# Faux Négatif (FN) : patient diabétique non détecté → coût élevé (5x)
# Faux Positif (FP) : patient sain mal classé → coût faible (1x)

COST_FN = 5   # Coût d'un faux négatif
COST_FP = 1   # Coût d'un faux positif

def business_score(y_true, y_pred):
    """
    Score métier basé sur les coûts relatifs des erreurs.
    Plus le score est bas, meilleur est le modèle.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = (COST_FN * fn) + (COST_FP * fp)
    total = len(y_true)
    normalized_score = 1 - (cost / (total * max(COST_FN, COST_FP)))
    return normalized_score, cost, tn, fp, fn, tp

def evaluate_model(model_name, y_true, y_pred, y_proba=None):
    """
    Évalue un modèle avec le score métier + métriques classiques.
    """
    score, cost, tn, fp, fn, tp = business_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"📊 Évaluation : {model_name}")
    print(f"{'='*50}")
    print(f"  ✅ Score métier     : {score:.4f} (plus proche de 1 = meilleur)")
    print(f"  💰 Coût total       : {cost}")
    print(f"  🔴 Faux Négatifs    : {fn} (diabétiques non détectés)")
    print(f"  🟡 Faux Positifs    : {fp} (sains mal classés)")
    print(f"  🟢 Vrais Positifs   : {tp}")
    print(f"  🟢 Vrais Négatifs   : {tn}")

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"  📈 AUC-ROC          : {auc:.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Sain', 'Diabétique'])}")
    return score, cost

# ─── Test du score métier sur un exemple ──────────────────────────────────────
if __name__ == "__main__":
    # Simulation de prédictions pour illustrer
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    print("🎯 Test du score métier — Contexte : Détection du diabète")
    print(f"   Coût Faux Négatif : {COST_FN}x")
    print(f"   Coût Faux Positif : {COST_FP}x")

    score, cost = evaluate_model("Modèle de démonstration", y_true, y_pred)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("scoring_diabete")

    with mlflow.start_run(run_name="test_score_metier"):
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        mlflow.log_metric("business_score", score)
        mlflow.log_metric("total_cost", cost)
        print("\n✅ Score métier loggé dans MLFlow !")