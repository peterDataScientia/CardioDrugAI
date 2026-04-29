import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import base64
import os
from rdkit import Chem
from rdkit import RDLogger
from sklearn.neighbors import NearestNeighbors
from skfp.fingerprints import PubChemFingerprint
from StreamJSME import StreamJSME

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="CardioDrugAI", page_icon="icon.png", layout="wide")
RDLogger.DisableLog("rdApp.*")

if "page" not in st.session_state:
    st.session_state.page = "screen"

# =========================================================
# LOAD CSS & ASSETS
# =========================================================
def load_css():
    if os.path.exists("css.css"):
        with open("css.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found.")

load_css()

# =========================================================
# HEADER IMAGE
# =========================================================
if os.path.exists("header.png"):
    with open("header.png", "rb") as f:
        img = base64.b64encode(f.read()).decode()
    st.markdown(
        f'''
        <div style="width:100%; text-align:center; margin-bottom:10px;">
            <img src="data:image/png;base64,{img}" style="width:100%; max-height:140px; object-fit:contain; border-radius:10px; display:block;">
        </div>
        ''', unsafe_allow_html=True
    )
st.markdown('<div class="qsar-title">TNF-α Inhibitor Screening Engine</div>', unsafe_allow_html=True)

# =========================================================
# CACHED MODELS LOADING
# ==========================
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    bundle = joblib.load("qsar_ad_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, bundle, feature_cols

model, scaler, bundle, feature_cols = load_assets()

# Constants from bundle
h_star = bundle["h_star"]
knn_train_space = np.array(bundle["knn_train_space"])
train_fps = np.array(bundle["train_fingerprints"])
fp_gen = PubChemFingerprint()
indices = np.array([int(c.split("_")[1]) for c in feature_cols])

# =========================================================
# FUNCTIONS
# =========================================================
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    full_fp = np.array(fp_gen.transform([mol]))[0]
    return full_fp[indices]

def tanimoto_similarity(fp_vec):
    fp_vec = fp_vec.astype(bool)
    refs = train_fps.astype(bool)
    inter = np.sum(refs & fp_vec, axis=1)
    union = np.sum(refs | fp_vec, axis=1)
    sims = inter / (union + 1e-8)
    return np.max(sims)

def compute_ad(x, fp_vec):
    # Leverage (H) calculation
    Xh = np.hstack([np.ones((x.shape[0], 1)), x])
    XtX_inv = np.linalg.pinv(Xh.T @ Xh)
    h = np.sum((Xh @ XtX_inv) * Xh, axis=1)
    H_flag = (h <= h_star).astype(int)
    
    # KNN flag
    dists = np.linalg.norm(knn_train_space - x, axis=1)
    K_flag = (dists <= np.percentile(dists, 95)).astype(int)
    
    # Similarity flag
    T_sim = tanimoto_similarity(fp_vec)
    T_flag = np.array([T_sim > 0.6]).astype(int)
    
    ADI = 0.33 * (H_flag + K_flag + T_flag)
    return ADI

def classify(ad_score):
    if ad_score >= 1.0: return "Highly Reliable"
    elif ad_score >= 0.66: return "Reliable"
    return "Unreliable"

def ic50_from_pic50(pIC50):
    return (10 ** (-pIC50)) * 1e6

# =========================================================
# SIDEBAR
# =========================================================
if os.path.exists("icon.png"):
    st.sidebar.image("icon.png", width=100)

st.sidebar.markdown("## Input Mode")
mode = st.sidebar.radio(
    "", ["JSME Draw", "SMILES → JSME", "Paste SMILES", "Upload File"]
)

run_btn = st.sidebar.button("🚀 Run Screening")
st.sidebar.markdown("---")

if st.sidebar.button("🧬 About CradioDrugAI"):
    st.session_state.page = "about"
if st.sidebar.button("🏠 Screening Page"):
    st.session_state.page = "screen"

st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Drug Discovery System")

# =========================================================
# SCREENING ENGINE
# =========================================================
def run_screening():
    col1, col2 = st.columns([1, 2])
    smiles_list = []
    ids = []
    smiles_from_jsme = None

    # ---------------- INPUT ----------------
    with col1:
        st.markdown('<div class="qsar-title">Molecular Input Engine</div>', unsafe_allow_html=True)
        
        if mode == "JSME Draw":
            smiles_from_jsme = StreamJSME()
        
        elif mode == "SMILES → JSME":
            smi = st.text_input("Enter SMILES for Editor")
            if smi:
                smiles_from_jsme = StreamJSME(smiles=smi)
        
        elif mode == "Paste SMILES":
            text = st.text_area("SMILES (one per line)", placeholder="e.g., C1=CC=CC=C1")
            if text:
                smiles_list = [s.strip() for s in text.split("\n") if s.strip()]
                ids = [f"Mol_{i+1}" for i in range(len(smiles_list))]
        
        else: # Upload File
            file = st.file_uploader("Upload .csv | .txt | .xlsx", type=["csv", "txt", "xls", "xlsx"])
            if file:
                df = pd.read_csv(file) if file.name.endswith(("csv", "txt")) else pd.read_excel(file)
                if df.shape[1] < 2:
                    st.error("File must contain ID and SMILES columns")
                else:
                    ids = df.iloc[:, 0].astype(str).tolist()
                    smiles_list = df.iloc[:, 1].astype(str).tolist()
                    st.success(f"Loaded {len(smiles_list)} molecules.")

    # ---------------- OUTPUT ----------------
    with col2:
        st.markdown('<div class="qsar-title results">Screening Output Console</div>', unsafe_allow_html=True)
        
        # Determine if we should process
        input_ready = (smiles_from_jsme and smiles_from_jsme != "") or len(smiles_list) > 0
        
        if run_btn and input_ready:
            if smiles_from_jsme:
                smiles_list = [smiles_from_jsme]
                ids = ["JSME_Molecule"]

            results = []
            start_time = time.time()
            progress = st.progress(0)
            status_placeholder = st.empty()
            
            total = len(smiles_list)
            for i, smi in enumerate(smiles_list):
                fp_vec = smiles_to_fp(smi)
                if fp_vec is None:
                    results.append([ids[i], smi, None, None, "Invalid SMILES"])
                else:
                    X = scaler.transform(fp_vec.reshape(1, -1))
                    pIC50 = float(model.predict(X)[0])
                    ic50 = ic50_from_pic50(pIC50)
                    ADI = compute_ad(X, fp_vec)
                    reliability = classify(float(ADI[0]))
                    results.append([ids[i], smi, round(pIC50, 3), round(ic50, 3), reliability])

                # ETA Update
                done = i + 1
                progress.progress(done / total)
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                status_placeholder.markdown(f"**Speed:** {speed:.1f} mol/s | **ETA:** {eta:.1f}s")

            # Final Table Presentation
            df_out = pd.DataFrame(results, columns=["ID", "SMILES", "pIC50", "IC50 (µM)", "Reliability"])
            
            st.download_button("📥 Download Results", df_out.to_csv(index=False).encode(), "qsar_results.csv")
            
            # Using the custom scrollable HTML table defined in your CSS
            st.markdown(
                f'<div class="qsar-scroll-table">{df_out.to_html(classes="qsar-table", index=False)}</div>', 
                unsafe_allow_html=True
            )
        elif run_btn:
            st.info("Please provide chemical structures in the left panel to begin.")

# =========================================================
# ROUTER
# =========================================================
if st.session_state.page == "screen":
    run_screening()
elif st.session_state.page == "about":
    try:
        from About import render
        render()
    except ImportError:
        st.error("About.py module not found. Please ensure the file exists.")
