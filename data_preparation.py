import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── 1. Chargement des données ───────────────────────────────────────────────
df = pd.read_csv("diabetes.csv")
print("✅ Dataset chargé — Shape :", df.shape)
print(df.head())

# ─── 2. Exploration initiale ─────────────────────────────────────────────────
print("\n📊 Informations générales :")
print(df.info())
print("\n📊 Statistiques descriptives :")
print(df.describe())
print("\n📊 Distribution de la cible :")
print(df["Outcome"].value_counts())

# ─── 3. Traitement des valeurs aberrantes ────────────────────────────────────
# Ces colonnes ne peuvent pas avoir la valeur 0 biologiquement
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\n⚠️ Valeurs à 0 (aberrantes) par colonne :")
for col in zero_cols:
    n_zeros = (df[col] == 0).sum()
    print(f"  {col} : {n_zeros} zéros")

# Remplacement des 0 par la médiane
for col in zero_cols:
    median = df[col].median()
    df[col] = df[col].replace(0, median)

print("\n✅ Valeurs aberrantes remplacées par la médiane")

# ─── 4. Feature Engineering ──────────────────────────────────────────────────
# Ratio Glucose / Insuline
df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)

# IMC par tranche d'âge
df["BMI_Age"] = df["BMI"] * df["Age"]

# Catégorie d'âge
df["Age_Group"] = pd.cut(df["Age"],
                          bins=[0, 30, 45, 60, 100],
                          labels=[0, 1, 2, 3]).astype(int)

print("✅ Nouvelles features créées :",
      ["Glucose_Insulin_Ratio", "BMI_Age", "Age_Group"])

# ─── 5. Séparation features / cible ──────────────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print(f"\n✅ Features : {X.shape[1]} colonnes")
print(f"✅ Cible : {y.value_counts().to_dict()}")

# ─── 6. Split train / test ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train : {X_train.shape[0]} lignes")
print(f"✅ Test  : {X_test.shape[0]} lignes")

# ─── 7. Normalisation ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("✅ Données normalisées")

# ─── 8. Sauvegarde ───────────────────────────────────────────────────────────
import joblib, os
os.makedirs("data", exist_ok=True)

pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled,  columns=X.columns).to_csv("data/X_test.csv",  index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
joblib.dump(scaler, "data/scaler.pkl")

print("\n✅ Données sauvegardées dans le dossier /data")
print("🎉 Préparation des données terminée !")