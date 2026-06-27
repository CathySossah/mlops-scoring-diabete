from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# ─── Chargement du modèle et du scaler ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "data", "scaler.pkl"))

app = FastAPI(
    title="API Scoring Diabète",
    description="Prédit le risque de diabète à partir de données médicales",
    version="1.0.0"
)

# ─── Schéma des données d'entrée ─────────────────────────────────────────────
class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

# ─── Route de santé ──────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "✅ API opérationnelle", "modele": "Random Forest"}

# ─── Route de prédiction ─────────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientData):
    # Reconstruction des features engineerisées
    data = pd.DataFrame([{
        "Pregnancies":               patient.Pregnancies,
        "Glucose":                   patient.Glucose,
        "BloodPressure":             patient.BloodPressure,
        "SkinThickness":             patient.SkinThickness,
        "Insulin":                   patient.Insulin,
        "BMI":                       patient.BMI,
        "DiabetesPedigreeFunction":  patient.DiabetesPedigreeFunction,
        "Age":                       patient.Age,
        "Glucose_Insulin_Ratio":     patient.Glucose / (patient.Insulin + 1),
        "BMI_Age":                   patient.BMI * patient.Age,
        "Age_Group":                 int(pd.cut([patient.Age],
                                        bins=[0, 30, 45, 60, 100],
                                        labels=[0, 1, 2, 3])[0])
    }])

    # Normalisation
    data_scaled = scaler.transform(data)

    # Prédiction
    prediction  = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "prediction":  int(prediction),
        "label":       "Diabétique" if prediction == 1 else "Non diabétique",
        "probabilite": round(float(probability), 4),
        "risque":      "Élevé" if probability > 0.6 else "Modéré" if probability > 0.4 else "Faible"
    }

# ─── Route métriques ─────────────────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    return {
        "modele":         "Random Forest",
        "auc_roc":        0.8280,
        "business_score": 0.8909,
        "f1_score":       0.70,
        "dataset":        "Pima Indians Diabetes"
    }