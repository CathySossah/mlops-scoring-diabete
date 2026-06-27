import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Chargement ───────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
model   = joblib.load("models/best_model.pkl")

# ─── SHAP ─────────────────────────────────────────────────────────────────────
print("🔄 Calcul des valeurs SHAP...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(
    shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values[1],
    X_test,
    feature_names=X_test.columns.tolist(),
    show=False
)
plt.title("SHAP — Importance des features (Random Forest)")
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150)
plt.close()
print("✅ Graphique SHAP sauvegardé : models/shap_summary.png")

# ─── Feature Importance classique ─────────────────────────────────────────────
importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\n📊 Feature Importance (Random Forest) :")
print(importances.round(4).to_string())

plt.figure(figsize=(10, 6))
importances.plot(kind="bar", color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150)
plt.close()
print("✅ Graphique sauvegardé : models/feature_importance.png")