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
# LOAD CSS
# =========================================================
def load_css():
    with open("css.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================================================
# HEADER IMAGE (CLEAN VERSION)
# =========================================================
if os.path.exists("header.png"):
    import base64
    with open("header.png", "rb") as f:
        img = base64.b64encode(f.read()).decode()

    st.markdown(
        f'''
        <div style="width:100%; text-align:center; margin-bottom:10px;">
            <img src="data:image/png;base64,{img}"
                 style="
                    width:100%;
                    max-height:140px;
                    object-fit:contain;
                    border-radius:10px;
                    display:block;
                 ">
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown('<div class="qsar-title">TNF-α Inhibitor Screening Engine</div>', unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
bundle = joblib.load("qsar_ad_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

h_star = bundle["h_star"]
knn_train_space = np.array(bundle["knn_train_space"])
train_fps = np.array(bundle["train_fingerprints"])

fp = PubChemFingerprint()
indices = np.array([int(c.split("_")[1]) for c in feature_cols])

# =========================================================
# FUNCTIONS
# =========================================================

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    full_fp = np.array(fp.transform([mol]))[0]
    return full_fp[indices]


def tanimoto_similarity(fp_vec):
    fp_vec = fp_vec.astype(bool)
    refs = train_fps.astype(bool)

    inter = np.sum(refs & fp_vec, axis=1)
    union = np.sum(refs | fp_vec, axis=1)

    sims = inter / (union + 1e-8)
    return np.max(sims)


def compute_ad(x, fp_vec):

    Xh = np.hstack([np.ones((x.shape[0], 1)), x])
    XtX_inv = np.linalg.pinv(Xh.T @ Xh)
    h = np.sum((Xh @ XtX_inv) * Xh, axis=1)
    H_flag = (h <= h_star).astype(int)

    dists = np.linalg.norm(knn_train_space - x, axis=1)
    K_flag = (dists <= np.percentile(dists, 95)).astype(int)

    T_sim = tanimoto_similarity(fp_vec)
    T_flag = np.array([T_sim > 0.6]).astype(int)

    ADI = 0.33 * (H_flag + K_flag + T_flag)
    return ADI


def classify(ad_score):
    if ad_score >= 1.0:
        return "Highly Reliable"
    elif ad_score >= 0.66:
        return "Reliable"
    return "Unreliable"


def ic50_from_pic50(pIC50):
    return (10 ** (-pIC50)) * 1e6

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.image("icon.png", width=100)

st.sidebar.markdown("## Input Mode")

mode = st.sidebar.radio(
    "",
    ["JSME Draw", "SMILES → JSME", "Paste SMILES", "Upload File"]
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
            smi = st.text_input("Enter SMILES")
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    smiles_from_jsme = StreamJSME(smiles=smi)

        elif mode == "Paste SMILES":
            text = st.text_area("SMILES (one per line)")
            if text:
                smiles_list = [s.strip() for s in text.split("\n") if s.strip()]
                ids = [f"Mol_{i+1}" for i in range(len(smiles_list))]

        # =====================================================
        # FILE UPLOAD (FINAL REQUIRED FORMAT)
        # =====================================================
        else:
            file = st.file_uploader(
                "Upload .csv | / .txt | .xlsx | .xls",
                type=["csv", "txt", "xls", "xlsx"]
            )

            if file:
                df = (
                    pd.read_csv(file)
                    if file.name.endswith(("csv", "txt"))
                    else pd.read_excel(file)
                )

                if df.shape[1] < 2:
                    st.error("File must contain ID/Name and SMILES columns")
                    smiles_list, ids = [], []
                else:
                    st.dataframe(df.head(5), use_container_width=True)
                    ids = df.iloc[:, 0].astype(str).tolist()
                    smiles_list = df.iloc[:, 1].astype(str).tolist()

        if smiles_from_jsme:
            smiles_list = [smiles_from_jsme]
            ids = ["JSMe_Molecule"]

        run_trigger = run_btn or bool(smiles_from_jsme)

    # ---------------- OUTPUT ----------------
    with col2:
        st.markdown('<div class="qsar-title results">Screening Output Console</div>', unsafe_allow_html=True)

        if run_trigger and smiles_list:

            results = []
            start_time = time.time()

            progress = st.progress(0)
            status_text = st.empty()

            total = len(smiles_list)

            for i, smi in enumerate(smiles_list):

                fp_vec = smiles_to_fp(smi)

                if fp_vec is None:
                    results.append([ids[i], smi, None, None, "Invalid"])
                else:
                    X = scaler.transform(fp_vec.reshape(1, -1))

                    pIC50 = model.predict(X)[0]
                    ic50 = ic50_from_pic50(pIC50)

                    ADI = compute_ad(X, fp_vec)
                    ad_score = float(ADI[0])
                    reliability = classify(ad_score)

                    results.append([
                        ids[i],
                        smi,
                        round(pIC50, 4),
                        round(ic50, 4),
                        reliability
                    ])

                # ================= ETA SYSTEM =================
                elapsed = time.time() - start_time
                done = i + 1

                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0

                progress.progress(done / total)

                status_text.markdown(
                    f"""
                    **Processed:** {done}/{total}  
                    **Speed:** {speed:.2f} mol/sec  
                    **ETA:** {eta:.1f} sec remaining
                    """
                )

            df_out = pd.DataFrame(
                results,
                columns=["ID", "SMILES", "pIC50", "IC50 (µM)", "Reliability"]
            )

            st.download_button(
                "Download Results",
                df_out.to_csv(index=False).encode(),
                "qsar_results.csv"
            )

            st.dataframe(df_out, use_container_width=True)

# =========================================================
# ROUTER
# =========================================================
if st.session_state.page == "screen":
    run_screening()

elif st.session_state.page == "about":
    from About import render
    render()