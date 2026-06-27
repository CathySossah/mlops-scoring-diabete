import pandas as pd
import numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from scipy import stats
import mlflow
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Chargement des données ────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

print("✅ Données chargées")
print(f"   Référence (train) : {X_train.shape}")
print(f"   Production (test) : {X_test.shape}")

# ─── 2. Simulation d'un drift ─────────────────────────────────────────────────
np.random.seed(42)
X_drift = X_test.copy()
X_drift["Glucose"] = X_drift["Glucose"] + np.random.normal(10, 5, len(X_drift))
X_drift["BMI"]     = X_drift["BMI"]     + np.random.normal(2,  1, len(X_drift))
X_drift["Age"]     = X_drift["Age"]     + np.random.normal(3,  1, len(X_drift))
print("⚠️  Données de production simulées avec drift artificiel")

# ─── 3. Définition du schéma ──────────────────────────────────────────────────
definition    = DataDefinition()
ref_dataset   = Dataset.from_pandas(X_train, data_definition=definition)
cur_dataset   = Dataset.from_pandas(X_test,  data_definition=definition)
drift_dataset = Dataset.from_pandas(X_drift, data_definition=definition)

# ─── 4. Rapport SANS drift ────────────────────────────────────────────────────
print("\n🔄 Génération rapport SANS drift...")
report_normal = Report(metrics=[DataDriftPreset()])
result_normal = report_normal.run(
    current_data=cur_dataset,
    reference_data=ref_dataset
)
result_normal.save_html("models/drift_report_normal.html")
print("✅ Rapport sauvegardé : models/drift_report_normal.html")

# ─── 5. Rapport AVEC drift ────────────────────────────────────────────────────
print("\n🔄 Génération rapport AVEC drift...")
report_drift = Report(metrics=[DataDriftPreset()])
result_drift = report_drift.run(
    current_data=drift_dataset,
    reference_data=ref_dataset
)
result_drift.save_html("models/drift_report_drift.html")
print("✅ Rapport sauvegardé : models/drift_report_drift.html")

# ─── 6. Analyse manuelle par feature (test KS) ───────────────────────────────
print("\n📊 Analyse du drift par feature (test de Kolmogorov-Smirnov) :")

drift_summary = []
for col in X_train.columns:
    stat, p_value = stats.ks_2samp(X_train[col], X_drift[col])
    drifted = p_value < 0.05
    drift_summary.append({
        "Feature":       col,
        "Drift détecté": "⚠️ OUI" if drifted else "✅ NON",
        "Score KS":      round(stat, 4),
        "P-value":       round(p_value, 4)
    })

df_drift = pd.DataFrame(drift_summary).sort_values("Score KS", ascending=False)
print(df_drift.to_string(index=False))

# ─── 7. Stratégie de réentraînement ──────────────────────────────────────────
print("\n" + "="*55)
print("📋 STRATÉGIE DE SURVEILLANCE ET RÉENTRAÎNEMENT")
print("="*55)
print("""
  1. MONITORING (hebdomadaire)
     → Calculer le score KS sur les nouvelles données
     → Seuil d'alerte : p_value < 0.05 sur 2+ features

  2. ALERTE AUTOMATIQUE
     → Si drift détecté : envoyer une notification
     → Logger l'événement dans MLFlow

  3. RÉENTRAÎNEMENT
     → Déclencher si score métier baisse de 5% ou plus
     → Réentraîner avec nouvelles données + anciennes
     → Comparer les modèles dans MLFlow avant déploiement

  4. VALIDATION
     → Tester le nouveau modèle sur un jeu de validation
     → Déployer uniquement si score métier >= 0.88
""")

# ─── 8. Log dans MLFlow ───────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("scoring_diabete")

drifted_features = df_drift[df_drift["Drift détecté"] == "⚠️ OUI"]["Feature"].tolist()

with mlflow.start_run(run_name="data_drift_analysis"):
    mlflow.log_metric("n_features_drifted", len(drifted_features))
    mlflow.log_metric("drift_ratio", len(drifted_features) / len(X_train.columns))
    mlflow.log_param("drifted_features", str(drifted_features))
    mlflow.log_artifact("models/drift_report_normal.html")
    mlflow.log_artifact("models/drift_report_drift.html")
    print("✅ Résultats drift loggés dans MLFlow !")

print("\n🎉 Analyse du data drift terminée !")
print("📂 Ouvre models/drift_report_drift.html dans ton navigateur pour le rapport complet.")