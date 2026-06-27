import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Détection du Diabète",
    page_icon="🩺",
    layout="wide"
)

# ─── Chargement modèle et scaler ─────────────────────────────────────────────
model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("data/scaler.pkl")

st.title("🩺 Système de Détection du Diabète")
st.markdown("Renseignez les données médicales du patient pour obtenir une prédiction.")

# ─── Formulaire ───────────────────────────────────────────────────────────────
st.subheader("📋 Données du patient")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Grossesses",               min_value=0,   max_value=20,  value=1)
    glucose     = st.number_input("Glucose (mg/dL)",          min_value=0,   max_value=300, value=120)
    blood_press = st.number_input("Pression artérielle",      min_value=0,   max_value=150, value=70)
    skin        = st.number_input("Épaisseur peau (mm)",      min_value=0,   max_value=100, value=20)

with col2:
    insulin     = st.number_input("Insuline (µU/mL)",         min_value=0,   max_value=900, value=80)
    bmi         = st.number_input("IMC (BMI)",                min_value=0.0, max_value=70.0,value=25.0, step=0.1)
    dpf         = st.number_input("Fonction pedigree diabète",min_value=0.0, max_value=3.0, value=0.5,  step=0.01)
    age         = st.number_input("Âge",                      min_value=1,   max_value=120, value=30)

st.markdown("---")

if st.button("🔍 Analyser le risque", use_container_width=True):

    payload = {
        "Pregnancies":              pregnancies,
        "Glucose":                  glucose,
        "BloodPressure":            blood_press,
        "SkinThickness":            skin,
        "Insulin":                  insulin,
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                      age
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result   = response.json()

        # ─── Résultat principal ───────────────────────────────────────────────
        st.subheader("📊 Résultat de l'analyse")
        col_res, col_jauge = st.columns([1, 1])

        with col_res:
            if result["prediction"] == 1:
                st.error(f"⚠️ **{result['label']}** — Risque {result['risque']}")
            else:
                st.success(f"✅ **{result['label']}** — Risque {result['risque']}")

            prob = result["probabilite"]
            st.metric("Probabilité de diabète", f"{prob*100:.1f}%")

            if result["risque"] == "Élevé":
                st.warning("🔴 Risque élevé — Consultation médicale fortement recommandée.")
            elif result["risque"] == "Modéré":
                st.warning("🟡 Risque modéré — Suivi médical conseillé.")
            else:
                st.info("🟢 Risque faible — Continuer un mode de vie sain.")

            with st.expander("📈 Performance du modèle"):
                st.write("**Modèle :** Random Forest")
                st.write("**AUC-ROC :** 0.8280")
                st.write("**Score Métier :** 0.8909")
                st.write("**Dataset :** Pima Indians Diabetes")

        with col_jauge:
            # ─── Jauge demi-cercle ────────────────────────────────────────────
            fig_jauge, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=True))
            ax.set_theta_offset(np.pi)
            ax.set_theta_direction(-1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)

            # Fond gris
            ax.barh(1, np.pi, left=0, height=0.5, color="#e0e0e0")

            # Arc coloré selon probabilité
            color = "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.4 else "#2ecc71"
            ax.barh(1, prob * np.pi, left=0, height=0.5, color=color)

            # Aiguille
            angle = prob * np.pi
            ax.annotate("", xy=(angle, 1), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

            # Labels
            ax.text(0,        1.7, "0%",   ha="center", fontsize=9, color="gray")
            ax.text(np.pi/2,  1.9, "50%",  ha="center", fontsize=9, color="gray")
            ax.text(np.pi,    1.7, "100%", ha="center", fontsize=9, color="gray")

            # Pourcentage au centre
            ax.text(np.pi/2, 0.1, f"{prob*100:.1f}%",
                    ha="center", va="center", fontsize=18,
                    fontweight="bold", color=color, transform=ax.transData)

            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.spines["polar"].set_visible(False)
            ax.grid(False)
            ax.set_title("Jauge de risque", fontsize=12, pad=15)

            st.pyplot(fig_jauge)
            plt.close()

        # ─── SHAP ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Explication SHAP — Facteurs influençant la prédiction")

        # Reconstruction features
        data = pd.DataFrame([{
            "Pregnancies":               pregnancies,
            "Glucose":                   glucose,
            "BloodPressure":             blood_press,
            "SkinThickness":             skin,
            "Insulin":                   insulin,
            "BMI":                       bmi,
            "DiabetesPedigreeFunction":  dpf,
            "Age":                       age,
            "Glucose_Insulin_Ratio":     glucose / (insulin + 1),
            "BMI_Age":                   bmi * age,
            "Age_Group":                 int(pd.cut([age], bins=[0,30,45,60,100], labels=[0,1,2,3])[0])
        }])

        data_scaled = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns
        )

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(data_scaled)

        # Valeurs SHAP pour classe 1 (diabétique)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0, :, 1]

        feature_names = data.columns.tolist()

        # ─── Tableau explicatif ───────────────────────────────────────────────
        df_shap = pd.DataFrame({
            "Feature": feature_names,
            "Valeur patient": [
                pregnancies, glucose, blood_press, skin,
                insulin, bmi, dpf, age,
                round(glucose / (insulin + 1), 3),
                round(bmi * age, 2),
                int(pd.cut([age], bins=[0,30,45,60,100], labels=[0,1,2,3])[0])
            ],
            "Impact SHAP": sv,
            "Influence": ["🔴 Augmente risque" if v > 0 else "🟢 Réduit risque" for v in sv]
        })

        df_shap["Impact SHAP"] = df_shap["Impact SHAP"].round(4)
        df_shap = df_shap.reindex(df_shap["Impact SHAP"].abs().sort_values(ascending=False).index)
        df_shap = df_shap.reset_index(drop=True)

        st.dataframe(
            df_shap.style.background_gradient(subset=["Impact SHAP"], cmap="RdYlGn_r"),
            use_container_width=True
        )

        # ─── Graphique SHAP barres ────────────────────────────────────────────
        fig_shap, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df_shap["Impact SHAP"]]
        ax.barh(df_shap["Feature"], df_shap["Impact SHAP"], color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Valeur SHAP (impact sur la prédiction)")
        ax.set_title("Impact de chaque feature sur la prédiction")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        st.info("Vérifiez que l'API FastAPI tourne sur le port 8000.")