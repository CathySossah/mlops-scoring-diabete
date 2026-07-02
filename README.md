# DiabèteScore — Détection du risque diabétique (MLOps)

Système complet de prédiction du risque de diabète : entraînement du modèle, API de scoring déployée, application web interactive avec explicabilité, suivi d'expériences et surveillance du data drift.

🔗 **API en production** : https://mlops-scoring-diabete.onrender.com

## 🎯 Objectif

Prédire, à partir de données cliniques d'un patient (glycémie, IMC, pression artérielle, etc.), le risque qu'il développe un diabète — avec une explication transparente de la prédiction.

## 🛠️ Stack technique

- **Scikit-learn** (Random Forest) pour la modélisation
- **Feature engineering** : création de variables dérivées (ratio glucose/insuline, interaction IMC × âge, tranches d'âge)
- **FastAPI** pour l'API de scoring en production
- **Streamlit** pour l'interface utilisateur (formulaire clinique, jauge de risque, dashboard de résultats)
- **SHAP** pour l'explicabilité des prédictions (quelles variables influencent le score, et dans quel sens)
- **MLflow** pour le tracking des expériences et versions de modèle
- **Surveillance du data drift** pour détecter une dérive des données en production
- **Déploiement** sur Render (API + web service, via `Procfile` / `render.yaml`)

## 📊 Performance du modèle

| Métrique | Valeur |
|---|---|
| AUC-ROC | 0.828 |
| F1-score | 0.70 |
| Score métier (business_score) | 0.891 |
| Dataset | Pima Indians Diabetes |
| Variables utilisées | 11 (8 cliniques + 3 dérivées) |

## 📦 Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Vérification que l'API est opérationnelle |
| `POST` | `/predict` | Prédiction du risque pour un patient |
| `GET` | `/metrics` | Métriques du modèle en production |

## 🚀 Utilisation en local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn api:app --reload

# Lancer l'application Streamlit (dans un autre terminal)
streamlit run app.py
```

## 📁 Structure du projet

- `data_preparation.py` — nettoyage et préparation du dataset
- `feature_importance.py` — analyse de l'importance des variables
- `train_models.py` — entraînement et comparaison de modèles
- `business_score.py` — métrique métier personnalisée pour évaluer le modèle
- `data_drift.py` — détection de dérive des données en production
- `mlflow_server.py` / `test_mlflow.py` — tracking des expériences MLflow
- `api.py` — API FastAPI de scoring
- `app.py` — interface Streamlit avec visualisation SHAP

---
*Projet réalisé dans le cadre de ma formation en Informatique de Gestion (UCAO), couvrant le cycle complet MLOps : de l'entraînement du modèle à son déploiement et sa surveillance en production.*
