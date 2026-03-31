"""
🔮 Streamlit App — Prediction du Churn Client
Lancer: streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Page config ───
st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { 
    color: #a0a0ff !important; 
    font-weight: 500; 
    font-size: 0.85rem;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span { color: #333 !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div { color: #e0e0e0 !important; }
[data-testid="stSidebar"] h2 { 
    color: #fff !important; 
    font-size: 1.3rem;
    border-bottom: 2px solid rgba(255,255,255,0.1);
    padding-bottom: 0.5rem;
}
[data-testid="stSidebar"] h3 { 
    color: #c0c0ff !important; 
    font-size: 1rem;
    margin-top: 1.2rem;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
}
.hero h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    position: relative;
}
.hero p { 
    color: rgba(255,255,255,0.85); 
    font-size: 1rem; 
    margin-top: 0.5rem;
    position: relative;
}

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.1); }
.card h3 { margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #1a1a2e; }
.card p { margin: 0; color: #666; font-size: 0.9rem; line-height: 1.5; }
.card .icon { font-size: 2rem; margin-bottom: 0.5rem; }

/* Model badge */
.model-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    color: white;
    margin-top: 0.5rem;
    backdrop-filter: blur(5px);
}

/* Prediction boxes */
.pred-box {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.pred-churn {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
}
.pred-safe {
    background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
    color: white;
}
.pred-box h2 { margin: 0; font-size: 1.3rem; font-weight: 600; }
.pred-box .prob { font-size: 3.5rem; font-weight: 800; margin: 0.5rem 0; }
.pred-box p { margin: 0; opacity: 0.9; font-size: 0.95rem; }

/* Probability bar */
.prob-container {
    background: #f5f5f5;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}
.prob-bar-bg {
    background: #e0e0e0;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}
.prob-green { background: linear-gradient(90deg, #00b894, #00cec9); }
.prob-red { background: linear-gradient(90deg, #ff6b6b, #ee5a24); }

/* Stats grid */
.stat-box {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #e8e8ff;
}
.stat-box .number { font-size: 1.8rem; font-weight: 800; color: #667eea; }
.stat-box .label { font-size: 0.8rem; color: #888; margin-top: 0.3rem; }

/* Actions box */
.actions-box {
    background: linear-gradient(135deg, #fff9e6 0%, #fff3cc 100%);
    border-radius: 14px;
    padding: 1.2rem;
    border: 1px solid #ffe0a0;
    margin-top: 1rem;
}
.actions-box h4 { margin: 0 0 0.7rem 0; color: #d4a017; font-size: 1rem; }
.actions-box ul { margin: 0; padding-left: 1.2rem; }
.actions-box li { color: #666; font-size: 0.85rem; margin-bottom: 0.3rem; }

/* Button override */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important;
}

/* Summary table */
.summary-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 15px rgba(0,0,0,0.05);
}
.summary-table th {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
}
.summary-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #e0e0e0;
    font-size: 0.85rem;
    color: #333 !important;
    background: #ffffff !important;
}
.summary-table tr:nth-child(even) td { background: #f5f6ff !important; }
.summary-table tr:last-child td { border-bottom: none; }
.summary-table td strong { color: #1a1a2e !important; }

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #bbb;
    font-size: 0.8rem;
    border-top: 1px solid #f0f0f0;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load model ───
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = {k: os.path.join(base, 'models', f) for k, f in 
             [('m','best_model.pkl'),('s','scaler.pkl'),('f','feature_names.pkl')]}
    if not os.path.exists(paths['m']): return None, None, None, None
    m = joblib.load(paths['m'])
    # Detect model name from class
    name_map = {'LogisticRegression':'Logistic Regression','RandomForestClassifier':'Random Forest',
                'SVC':'SVM','KNeighborsClassifier':'k-NN','DecisionTreeClassifier':'Decision Tree'}
    model_name = name_map.get(type(m).__name__, type(m).__name__)
    return m, joblib.load(paths['s']), joblib.load(paths['f']), model_name

def preprocess_input(data, feat_names, scaler):
    df = pd.DataFrame([data])
    df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    for c, m in {'gender':{'Female':0,'Male':1},'Partner':{'No':0,'Yes':1},
                 'Dependents':{'No':0,'Yes':1},'PhoneService':{'No':0,'Yes':1},
                 'PaperlessBilling':{'No':0,'Yes':1}}.items():
        if c in df.columns: df[c] = df[c].map(m)
    for c in ['InternetService','Contract','PaymentMethod','MultipleLines',
              'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
              'StreamingTV','StreamingMovies']:
        if c in df.columns:
            dum = pd.get_dummies(df[c], prefix=c, drop_first=True)
            df = pd.concat([df, dum], axis=1)
            df.drop(c, axis=1, inplace=True)
    for f in feat_names:
        if f not in df.columns: df[f] = 0
    df = df[feat_names]
    return pd.DataFrame(scaler.transform(df), columns=feat_names)

model, scaler, feat_names, model_name = load_artifacts()

# ─── Hero ───
badge = f'<div class="model-badge">🤖 Modèle actif : <strong>{model_name}</strong> (meilleur parmi 5 testés)</div>' if model_name else ''
st.markdown(f"""
<div class="hero">
    <h1>🔮 Churn Predictor</h1>
    <p>Système intelligent de prédiction du churn client — Telco Customer Churn</p>
    {badge}
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Modèle non trouvé ! Exécutez d'abord : `python train_model.py`")
    st.stop()

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 👤 Profil Client")
    
    st.markdown("### 📋 Démographie")
    gender = st.selectbox("Genre", ["Male", "Female"])
    senior = st.selectbox("Citoyen Senior", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    partner = st.selectbox("Partenaire", ["Yes", "No"], format_func=lambda x: "Oui" if x=="Yes" else "Non")
    deps = st.selectbox("Personnes à charge", ["Yes", "No"], format_func=lambda x: "Oui" if x=="Yes" else "Non")
    
    st.markdown("### 📝 Contrat")
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Facturation en ligne", ["Yes", "No"], format_func=lambda x: "Oui" if x=="Yes" else "Non")
    payment = st.selectbox("Paiement", ["Electronic check", "Mailed check", 
                           "Bank transfer (automatic)", "Credit card (automatic)"])
    
    st.markdown("### 📡 Services")
    phone = st.selectbox("Téléphone", ["Yes", "No"], format_func=lambda x: "Oui" if x=="Yes" else "Non")
    multi_lines = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    sec = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Sauvegarde en ligne", ["Yes", "No", "No internet service"])
    devprot = st.selectbox("Protection appareil", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Support technique", ["Yes", "No", "No internet service"])
    stv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    smov = st.selectbox("Streaming Films", ["Yes", "No", "No internet service"])
    
    st.markdown("### 💰 Facturation")
    monthly = st.slider("Charges mensuelles ($)", 18.0, 120.0, 50.0, step=0.5)
    total = st.slider("Charges totales ($)", 18.0, 9000.0, 500.0, step=10.0)

# ─── Main content ───
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📊 Récapitulatif du Client")
    
    rows = [
        ("Genre", "👤", gender),
        ("Senior", "🧓", "Oui" if senior else "Non"),
        ("Partenaire", "💑", partner),
        ("Personnes à charge", "👨‍👩‍👧", deps),
        ("Ancienneté", "📅", f"{tenure} mois"),
        ("Contrat", "📝", contract),
        ("Internet", "🌐", internet),
        ("Téléphone", "📞", phone),
        ("Charges mensuelles", "💵", f"${monthly:.2f}"),
        ("Charges totales", "💰", f"${total:.2f}"),
        ("Paiement", "💳", payment),
    ]
    
    table_html = '<table class="summary-table"><tr><th>Paramètre</th><th>Valeur</th></tr>'
    for label, icon, val in rows:
        table_html += f'<tr><td>{icon} {label}</td><td><strong>{val}</strong></td></tr>'
    table_html += '</table>'
    st.markdown(table_html, unsafe_allow_html=True)

with col_right:
    st.markdown("### 🎯 Prédiction")
    
    if st.button("🔮  Analyser le Risque de Churn", use_container_width=True):
        inp = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
            'Dependents': deps, 'tenure': tenure, 'PhoneService': phone,
            'MultipleLines': multi_lines, 'InternetService': internet,
            'OnlineSecurity': sec, 'OnlineBackup': backup,
            'DeviceProtection': devprot, 'TechSupport': tech,
            'StreamingTV': stv, 'StreamingMovies': smov,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        
        try:
            Xi = preprocess_input(inp, feat_names, scaler)
            pred = model.predict(Xi)[0]
            prob = model.predict_proba(Xi)[0] if hasattr(model, 'predict_proba') else None
            cp = prob[1] * 100 if prob is not None else (100 if pred == 1 else 0)
            sp = 100 - cp
            
            if pred == 1:
                st.markdown(f"""
                <div class="pred-box pred-churn">
                    <h2>⚠️ RISQUE DE CHURN DÉTECTÉ</h2>
                    <div class="prob">{cp:.1f}%</div>
                    <p>Ce client a un risque élevé de résiliation</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box pred-safe">
                    <h2>✅ CLIENT FIDÈLE</h2>
                    <div class="prob">{sp:.1f}%</div>
                    <p>Ce client est susceptible de rester</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability bars
            st.markdown(f"""
            <div class="prob-container">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                    <span style="font-weight:600;color:#00b894">✅ Fidèle: {sp:.1f}%</span>
                    <span style="font-weight:600;color:#ee5a24">⚠️ Churn: {cp:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill {'prob-red' if pred==1 else 'prob-green'}" style="width:{cp if pred==1 else sp}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors
            if pred == 1:
                st.markdown("""
                <div class="actions-box">
                    <h4>💡 Actions Recommandées</h4>
                    <ul>
                        <li>📞 Contacter le client de manière proactive</li>
                        <li>🎁 Proposer une offre de fidélisation personnalisée</li>
                        <li>📝 Suggérer un contrat annuel avec avantages</li>
                        <li>🛡️ Offrir des services complémentaires gratuits</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# ─── Bottom stats ───
st.markdown("---")
st.markdown("### ℹ️ À propos")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="stat-box">
        <div class="number">7 043</div>
        <div class="label">Clients analysés</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-box">
        <div class="number">21</div>
        <div class="label">Variables utilisées</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-box">
        <div class="number">5</div>
        <div class="label">Modèles comparés</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="stat-box">
        <div class="number">~80%</div>
        <div class="label">Accuracy du modèle</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">Mini-Projet Systèmes Intelligents — Prédiction du Churn Client — 2025/2026</div>', unsafe_allow_html=True)
