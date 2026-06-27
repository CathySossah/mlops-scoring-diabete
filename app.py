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
    page_title="DiabèteScore — Détection IA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background: #0a0f1e;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {
    padding: 0 2rem 2rem 2rem;
    max-width: 1200px;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2d5a 50%, #0d2240 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 179, 237, 0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #90b4d4;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99, 179, 237, 0.12);
    border: 1px solid rgba(99, 179, 237, 0.3);
    color: #63b3ed;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Result cards */
.result-positive {
    background: linear-gradient(135deg, #2d1515 0%, #3d1a1a 100%);
    border: 1px solid rgba(252, 129, 129, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #0f2d1f 0%, #143d28 100%);
    border: 1px solid rgba(72, 199, 142, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-icon {
    font-size: 2.5rem;
}
.risk-tag {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.risk-high   { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }
.risk-medium { background: rgba(246,173, 85,0.15); color: #f6ad55; border: 1px solid rgba(246,173, 85,0.3); }
.risk-low    { background: rgba( 72,199,142,0.15); color: #48c78e; border: 1px solid rgba( 72,199,142,0.3); }

/* Metric pill */
.metric-pill {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label {
    font-size: 0.78rem;
    color: #718096;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Model info */
.model-info {
    background: #0d1520;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1a2a40;
    font-size: 0.85rem;
}
.model-row:last-child { border-bottom: none; }
.model-key   { color: #718096; }
.model-val   { color: #e2e8f0; font-weight: 500; }

/* Alert boxes */
.alert-high   { background:#2d1515; border:1px solid #fc8181; border-radius:10px; padding:0.8rem 1rem; color:#fc8181;  font-size:0.88rem; margin-top:0.8rem; }
.alert-medium { background:#2d2010; border:1px solid #f6ad55; border-radius:10px; padding:0.8rem 1rem; color:#f6ad55;  font-size:0.88rem; margin-top:0.8rem; }
.alert-low    { background:#0f2d1f; border:1px solid #48c78e; border-radius:10px; padding:0.8rem 1rem; color:#48c78e;  font-size:0.88rem; margin-top:0.8rem; }

/* Inputs */
.stNumberInput input {
    background: #1a2035 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
.stNumberInput input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}
label { color: #a0aec0 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(49,130,206,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(49,130,206,0.4) !important;
}

/* Divider */
hr { border-color: #1f2937 !important; margin: 1.5rem 0 !important; }

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Chargement modèle et scaler ──────────────────────────────────────────────
model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("data/scaler.pkl")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 IA · Random Forest · Pima Indians Dataset</div>
    <h1 class="hero-title">🩺 DiabèteScore</h1>
    <p class="hero-subtitle">Système de détection du risque diabétique basé sur l'intelligence artificielle.<br>
    Renseignez les données biologiques du patient pour obtenir une analyse instantanée.</p>
</div>
""", unsafe_allow_html=True)

# ─── Formulaire ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Données cliniques du patient</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    pregnancies = st.number_input("🤰 Grossesses",          min_value=0,   max_value=20,   value=1)
    glucose     = st.number_input("🩸 Glucose (mg/dL)",     min_value=0,   max_value=300,  value=120)

with col2:
    blood_press = st.number_input("💓 Pression artérielle", min_value=0,   max_value=150,  value=70)
    skin        = st.number_input("📏 Épaisseur peau (mm)", min_value=0,   max_value=100,  value=20)

with col3:
    insulin     = st.number_input("💉 Insuline (µU/mL)",    min_value=0,   max_value=900,  value=80)
    bmi         = st.number_input("⚖️ IMC (BMI)",           min_value=0.0, max_value=70.0, value=25.0, step=0.1)

with col4:
    dpf         = st.number_input("🧬 Pedigree diabète",    min_value=0.0, max_value=3.0,  value=0.5,  step=0.01)
    age         = st.number_input("🎂 Âge",                 min_value=1,   max_value=120,  value=30)

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔍 Analyser le risque diabétique", use_container_width=True)

if analyze:
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
        API_URL = "https://mlops-scoring-diabete.onrender.com/predict"
        response = requests.post(API_URL, json=payload, timeout=30)
        result   = response.json()

        prob    = result["probabilite"]
        is_diabetic = result["prediction"] == 1
        risque  = result["risque"]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Résultats de l\'analyse</div>', unsafe_allow_html=True)

        col_res, col_gauge, col_info = st.columns([1.2, 1, 1])

        # ─── Résultat principal ───────────────────────────────────────────────
        with col_res:
            if is_diabetic:
                risk_class = "risk-high" if risque == "Élevé" else "risk-medium"
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-icon">⚠️</div>
                    <div class="result-label" style="color:#fc8181">{result['label']}</div>
                    <span class="risk-tag {risk_class}">Risque {risque}</span>
                </div>
                """, unsafe_allow_html=True)
                if risque == "Élevé":
                    st.markdown('<div class="alert-high">🔴 Consultation médicale fortement recommandée.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-medium">🟡 Suivi médical conseillé dans les prochaines semaines.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-icon">✅</div>
                    <div class="result-label" style="color:#48c78e">{result['label']}</div>
                    <span class="risk-tag risk-low">Risque {risque}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="alert-low">🟢 Continuer un mode de vie sain et un suivi annuel.</div>', unsafe_allow_html=True)

        # ─── Jauge ───────────────────────────────────────────────────────────
        with col_gauge:
            fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            ax.set_theta_offset(np.pi)
            ax.set_theta_direction(-1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)

            ax.barh(1, np.pi, left=0, height=0.5, color="#1f2937")
            color = "#fc8181" if prob > 0.6 else "#f6ad55" if prob > 0.4 else "#48c78e"
            ax.barh(1, prob * np.pi, left=0, height=0.5, color=color, alpha=0.9)

            angle = prob * np.pi
            ax.annotate("", xy=(angle, 1.1), xytext=(angle, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))

            ax.text(0,       1.8, "0%",   ha="center", fontsize=8, color="#718096")
            ax.text(np.pi/2, 2.0, "50%",  ha="center", fontsize=8, color="#718096")
            ax.text(np.pi,   1.8, "100%", ha="center", fontsize=8, color="#718096")
            ax.text(np.pi/2, 0.05, f"{prob*100:.1f}%",
                    ha="center", va="center", fontsize=20,
                    fontweight="bold", color=color, transform=ax.transData)

            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.spines["polar"].set_visible(False)
            ax.grid(False)
            ax.set_title("Probabilité de diabète", fontsize=10, color="#a0aec0", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ─── Infos modèle ─────────────────────────────────────────────────────
        with col_info:
            st.markdown("""
            <div class="metric-pill">
                <div class="metric-value">82.8%</div>
                <div class="metric-label">AUC-ROC</div>
            </div>
            <div class="metric-pill">
                <div class="metric-value">89.1%</div>
                <div class="metric-label">Score Métier</div>
            </div>
            <div class="model-info">
                <div class="model-row"><span class="model-key">Modèle</span><span class="model-val">Random Forest</span></div>
                <div class="model-row"><span class="model-key">Dataset</span><span class="model-val">Pima Indians</span></div>
                <div class="model-row"><span class="model-key">Features</span><span class="model-val">11 variables</span></div>
                <div class="model-row"><span class="model-key">Version</span><span class="model-val">v1.0</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ─── SHAP ─────────────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔬 Explication SHAP — Facteurs d\'influence</div>', unsafe_allow_html=True)

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

        data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns)
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(data_scaled)

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0, :, 1]

        feature_names = data.columns.tolist()

        df_shap = pd.DataFrame({
            "Feature":        feature_names,
            "Valeur patient": [pregnancies, glucose, blood_press, skin, insulin, bmi, dpf, age,
                               round(glucose/(insulin+1), 3), round(bmi*age, 2),
                               int(pd.cut([age], bins=[0,30,45,60,100], labels=[0,1,2,3])[0])],
            "Impact SHAP":    sv.round(4),
            "Influence":      ["🔴 Augmente risque" if v > 0 else "🟢 Réduit risque" for v in sv]
        })

        df_shap = df_shap.reindex(df_shap["Impact SHAP"].abs().sort_values(ascending=False).index).reset_index(drop=True)

        col_t, col_g = st.columns([1.1, 1])

        with col_t:
            st.dataframe(
                df_shap.style.background_gradient(subset=["Impact SHAP"], cmap="RdYlGn_r"),
                use_container_width=True, height=380
            )

        with col_g:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            fig2.patch.set_facecolor('#111827')
            ax2.set_facecolor('#111827')
            colors2 = ["#fc8181" if v > 0 else "#48c78e" for v in df_shap["Impact SHAP"]]
            bars = ax2.barh(df_shap["Feature"], df_shap["Impact SHAP"], color=colors2, height=0.6, edgecolor='none')
            ax2.axvline(x=0, color="#4a5568", linewidth=1)
            ax2.set_xlabel("Valeur SHAP", color="#a0aec0", fontsize=9)
            ax2.set_title("Impact sur la prédiction", color="#e2e8f0", fontsize=11, fontweight='bold', pad=12)
            ax2.tick_params(colors="#a0aec0", labelsize=8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#2d3748')
            ax2.spines['bottom'].set_color('#2d3748')
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    except Exception as e:
        st.markdown(f"""
        <div style="background:#2d1515;border:1px solid #fc8181;border-radius:12px;padding:1.2rem;color:#fc8181;">
            ❌ <strong>Erreur de connexion à l'API</strong><br>
            <small style="color:#a0aec0;">{e}</small>
        </div>
        """, unsafe_allow_html=True)