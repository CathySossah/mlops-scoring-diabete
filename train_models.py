import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from business_score import business_score, evaluate_model
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Chargement des données ────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print(f"✅ Données chargées — Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 2. Gestion du déséquilibre avec SMOTE ───────────────────────────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"✅ SMOTE appliqué — Distribution : {dict(pd.Series(y_train_bal).value_counts())}")

# ─── 3. Configuration MLFlow ──────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("scoring_diabete")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── 4. Modèles et hyperparamètres ───────────────────────────────────────────
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, None],
            "min_samples_split": [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }
    }
}

# ─── 5. Baseline simple ───────────────────────────────────────────────────────
print("\n📊 Baseline — Prédiction majoritaire (classe 0)")
y_baseline = np.zeros(len(y_test), dtype=int)
baseline_score, baseline_cost = evaluate_model("Baseline (majorité)", y_test, y_baseline)

# ─── 6. Entraînement et logging ───────────────────────────────────────────────
best_models = {}

for name, config in models.items():
    print(f"\n🔄 Entraînement : {name}...")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid.fit(X_train_bal, y_train_bal)
    best_model = grid.best_estimator_

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    score, cost = evaluate_model(name, y_test, y_pred, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Log dans MLFlow
    with mlflow.start_run(run_name=name):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("business_score", score)
        mlflow.log_metric("total_cost", cost)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(best_model, name=name)

    best_models[name] = {"model": best_model, "score": score, "auc": auc}
    print(f"  ✅ Meilleurs params : {grid.best_params_}")

# ─── 7. Comparaison finale ────────────────────────────────────────────────────
print("\n" + "="*50)
print("🏆 COMPARAISON FINALE DES MODÈLES")
print("="*50)
print(f"{'Modèle':<25} {'Score Métier':>12} {'AUC-ROC':>10}")
print("-"*50)
print(f"{'Baseline':<25} {baseline_score:>12.4f} {'N/A':>10}")
for name, result in best_models.items():
    print(f"{name:<25} {result['score']:>12.4f} {result['auc']:>10.4f}")

# ─── 8. Sauvegarde du meilleur modèle ────────────────────────────────────────
best_name = max(best_models, key=lambda x: best_models[x]["score"])
best_model = best_models[best_name]["model"]
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"\n🥇 Meilleur modèle : {best_name}")
print(f"✅ Sauvegardé dans models/best_model.pkl")