import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import joblib
import base64
from io import StringIO

# ========== STREAMLIT CLOUD OPTIMIZATIONS ==========
import sys
import os
sys.setrecursionlimit(10000)
# ===================================================

# Try to import RDKit – if missing, show a warning but allow other parts to run
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, Draw
    from rdkit.Chem.QED import qed
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ========== MUST BE FIRST STREAMLIT COMMAND ==========
st.set_page_config(
    page_title="MIND-EGFR",
    page_icon="🧬",
    layout="wide"
)
# =====================================================

# ------------------------------
# Custom CSS for Professional Styling with Academic Background
# ------------------------------
def local_css():
    st.markdown("""
    <style>
        /* Light academic background color */
        html, body, [class*="css"] {
            background-color: #f8f7f3;
            font-family: 'Segoe UI', 'Helvetica', sans-serif;
            font-size: 16px;
        }
        
        /* Main container padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            background-color: #f8f7f3;
        }
        
        /* Headers - Professional and Readable */
        h1 {
            color: #1f4788;
            font-weight: 800;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 0.8rem;
            font-size: 2.5rem;
        }
        h2 {
            color: #2e5c8a;
            font-weight: 700;
            margin-top: 1.8rem;
            border-left: 5px solid #0066cc;
            padding-left: 1rem;
            font-size: 1.8rem;
        }
        h3 {
            color: #1f4788;
            font-weight: 600;
            font-size: 1.4rem;
        }
        h4 {
            color: #2e5c8a;
            font-weight: 600;
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #e8f1f9, #eae8e0);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15);
            border-left: 5px solid #0066cc;
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            box-shadow: 0 8px 20px rgba(0, 102, 204, 0.25);
            transform: translateY(-2px);
        }
        div[data-testid="metric-container"] label {
            color: #1f4788 !important;
            font-weight: 700;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #0066cc !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ede9df 0%, #e8f1f9 100%);
            border-right: 2px solid #0066cc;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            background-color: transparent;
        }
        section[data-testid="stSidebar"] h2 {
            color: #1f4788;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #0066cc !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.7rem 2rem !important;
            font-weight: 700 !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.25);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0052a3 !important;
            box-shadow: 0 6px 18px rgba(0, 102, 204, 0.35);
            transform: translateY(-2px);
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #0066cc;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.1);
            background-color: #fafaf8;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #e8f1f9;
            border-radius: 8px;
            font-weight: 700;
            color: #1f4788;
            border-left: 4px solid #0066cc;
            padding-left: 0.8rem;
        }
        
        /* Introduction card */
        .intro-box {
            background: linear-gradient(135deg, #eae8e0 0%, #e8f1f9 100%);
            border-radius: 15px;
            padding: 2.5rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 6px 16px rgba(0, 102, 204, 0.12);
            border: 2px solid #0066cc;
        }
        .intro-box h3 {
            color: #1f4788;
            margin-bottom: 1rem;
            font-size: 1.8rem;
            font-weight: 800;
        }
        .intro-box p {
            color: #3a5a7a;
            line-height: 1.7;
            font-size: 1rem;
        }
        .short-intro-box {
            background: linear-gradient(135deg, #e8f1f9 0%, #d4e4f7 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border-left: 5px solid #0066cc;
            border-top: 1px solid #0066cc;
        }
        .short-intro-box p {
            color: #2e5c8a;
            line-height: 1.6;
            margin: 0.5rem 0;
            font-size: 0.95rem;
        }
        .feature-highlight {
            background: linear-gradient(135deg, #e8f1f9 0%, #d4e4f7 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            border: 1px solid #0066cc;
        }
        .feature-item {
            background: #fafaf8;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 102, 204, 0.1);
            font-weight: 600;
            color: #1f4788;
            border: 2px solid #d4e4f7;
            transition: all 0.3s ease;
        }
        .feature-item:hover {
            border-color: #0066cc;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.2);
            transform: translateY(-3px);
        }
        
        /* Footer */
        .footer {
            margin-top: 3rem;
            padding: 2rem;
            text-align: center;
            border-top: 2px solid #0066cc;
            background: linear-gradient(135deg, #eae8e0 0%, #e8f1f9 100%);
            border-radius: 10px;
            color: #2e5c8a;
            font-size: 0.95rem;
            line-height: 1.8;
        }
        .footer .names {
            font-weight: 700;
            color: #0066cc;
        }
        .footer-section {
            margin: 0.8rem 0;
        }
        
        /* Radio buttons (navigation) */
        div[role="radiogroup"] label {
            background-color: #e8f1f9;
            padding: 0.5rem 1.2rem;
            border-radius: 8px;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            transition: all 0.2s;
            color: #1f4788;
            font-weight: 600;
            border: 2px solid #d4e4f7;
        }
        div[role="radiogroup"] label:hover {
            background-color: #d4e4f7;
            border-color: #0066cc;
        }
        
        /* Text input and selectbox */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select {
            border-radius: 8px;
            border: 2px solid #0066cc !important;
            background-color: #fafaf8 !important;
            color: #1f4788 !important;
            box-shadow: 0 2px 8px rgba(0, 102, 204, 0.1);
            padding: 0.6rem 1rem !important;
        }
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #0052a3 !important;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.2);
        }
        
        /* Text area */
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #0066cc !important;
            background-color: #fafaf8 !important;
            color: #1f4788 !important;
            box-shadow: 0 2px 8px rgba(0, 102, 204, 0.1);
            padding: 0.8rem 1rem !important;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: #0052a3 !important;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.2);
        }
        
        /* File uploader */
        .stFileUploader > div {
            border-radius: 8px;
            border: 2px dashed #0066cc;
            background-color: rgba(0, 102, 204, 0.03);
            padding: 1.5rem;
        }
        
        /* Sliders */
        div[data-testid="stSlider"] > div {
            padding-top: 0.5rem;
        }
        
        /* Success/Warning/Error messages */
        .stAlert {
            border-radius: 8px;
            border-left: 5px solid #0066cc;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ------------------------------
# Configuration
# ------------------------------
DB_DIR = Path("database")
DB_DIR.mkdir(exist_ok=True)

# Look for the database file in root first, then in database folder
if Path("egfr_npdb.db").exists():
    DB_PATH = Path("egfr_npdb.db")
elif (DB_DIR / "egfr_npdb.db").exists():
    DB_PATH = DB_DIR / "egfr_npdb.db"
else:
    DB_PATH = DB_DIR / "egfr_npdb.db"  # fallback path (will trigger error later)

# ------------------------------
# Database Setup
# ------------------------------
def init_db():
    """Initialize database connection - use existing egfr_npdb.db file (root or database folder)"""
    if not DB_PATH.exists():
        st.error(f"❌ Database file not found at {DB_PATH.absolute()}")
        st.error("Please ensure egfr_npdb.db is in the root folder or inside 'database/'")
        st.stop()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compounds'")
        if cursor.fetchone() is None:
            conn.close()
            st.error("Database exists but 'compounds' table not found")
            st.stop()
        conn.close()
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.stop()

init_db()

@st.cache_data(ttl=600)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM compounds", conn)
    conn.close()
    return df

df = load_data()

def safe_unique(series):
    vals = series.dropna().unique().tolist()
    return sorted([str(v) for v in vals if str(v).strip() != ""])

# ------------------------------
# Load Pre-trained Models (Cached)
# ------------------------------
@st.cache_resource
def load_models():
    """Load all .pkl models and feature indices."""
    models = {}
    model_names = ['rf', 'svm', 'knn', 'xgboost', 'lightgbm', 'et', 'ensemble']
    for name in model_names:
        try:
            models[name] = joblib.load(f"{name}.pkl")
        except FileNotFoundError:
            models[name] = None
        except Exception as e:
            st.warning(f"Error loading {name}.pkl: {str(e)[:100]}")
            models[name] = None

    # Load selected feature indices
    feature_indices = None
    try:
        feature_indices = np.load("selected_feature_indices.npy")
    except:
        pass

    # Check if at least one model loaded successfully
    if all(v is None for v in models.values()):
        st.error("No model files could be loaded. Please check that .pkl files exist.")
        st.stop()
    
    return models, feature_indices

models, feature_indices = load_models()

# ------------------------------
# Helper: Generate Morgan Fingerprint and Molecular Properties
# ------------------------------
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(list(fp))

def compute_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    props = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
        'QED': qed(mol),
        'LipinskiViolations': (
            (1 if Descriptors.MolWt(mol) > 500 else 0) +
            (1 if Descriptors.MolLogP(mol) > 5 else 0) +
            (1 if Lipinski.NumHDonors(mol) > 5 else 0) +
            (1 if Lipinski.NumHAcceptors(mol) > 10 else 0)
        )
    }
    return props

# ------------------------------
# Prediction Function
# ------------------------------
def predict_smiles(smiles_list, include_properties=False):
    """Predict bioactivity for a list of SMILES."""
    results = []
    for smi in smiles_list:
        smi = smi.strip()
        if not smi:
            continue
        
        row = {'SMILES': smi}
        
        # Add molecular properties if requested
        if include_properties:
            props = compute_molecular_properties(smi)
            if props is not None:
                row.update(props)
        
        fp = smiles_to_fingerprint(smi)
        if fp is None:
            row['Error'] = 'Invalid SMILES'
            results.append(row)
            continue

        # Apply feature selection
        if feature_indices is not None:
            fp_selected = fp[feature_indices]
        else:
            fp_selected = fp

        fp_2d = fp_selected.reshape(1, -1)

        for model_name, model in models.items():
            if model is None:
                row[f'{model_name}_prob'] = None
                row[f'{model_name}_pred'] = None
                continue

            try:
                # Get probability of active class (class 1)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(fp_2d)[0, 1]
                elif hasattr(model, "decision_function"):
                    # Fallback: use decision function and calibrate roughly
                    df_val = model.decision_function(fp_2d)[0]
                    proba = 1.0 / (1.0 + np.exp(-df_val))  # sigmoid
                else:
                    proba = None
                pred = model.predict(fp_2d)[0]
            except Exception as e:
                proba = None
                pred = None

            row[f'{model_name}_prob'] = proba
            row[f'{model_name}_pred'] = pred

        results.append(row)
    return pd.DataFrame(results)

# Helper function to get 2D structure image
def get_mol_image(smiles, size=(400, 300)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except:
        return None

# ------------------------------
# Mode Selection in Sidebar
# ------------------------------
st.sidebar.title("🧭 Navigation")
mode = st.sidebar.radio("Select Mode", ["📊 Database Browser", "🧪 Bioactivity Predictor"])

# ------------------------------
# MODE 1: Database Browser
# ------------------------------
if mode == "📊 Database Browser":
    # Main Title
    st.title("🎗️ MIND-EGFR")
    st.markdown("**MIND-EGFR is an integrated web resource that combines a curated natural product database with an interactive bioactivity predictor, both focused on the Epidermal Growth Factor Receptor (EGFR) as a therapeutic target (PDB: 8F1Y).**")
    st.markdown("---")

    # Enhanced Introduction Section
    with st.container():
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 📦 Database Construction")
            st.markdown("""
            - **276,518** natural products from the LOTUS initiative  
            - **142,035** passed Lipinski's Rule of Five (zero violations)  
            - **778** predicted as active by an ensemble of six ML classifiers  
            - **730** successfully docked into EGFR (8F1Y)  
            - **16** compounds exceed the best control drug (Poziotinib, −8.9 kcal/mol)  
            - **139** compounds outperform the mean control affinity (−7.9 kcal/mol)
            """)
        with col_right:
            st.markdown("#### ⚙️ Machine Learning Performance")
            st.markdown("""
            - **Ensemble model** – 5‑fold CV AUC: **0.9646 ± 0.0037**  
            - External validation accuracy: **88.05%**, AUC: **0.9411**  
            - Individual classifiers: RF, SVM, KNN, XGBoost, LightGBM, ExtraTrees  
            - All models trained on Morgan fingerprints (radius 2, 2048 bits)
            """)

        # Key Features with equal spacing using columns
        st.markdown("#### 🧬 Key Features")
        feat_cols = st.columns(3)
        features = [
            "🔎 Interactive Database Browser",
            "🧪 On‑the‑fly Bioactivity Predictor",
            "⚖️ Full ADME & Drug‑likeness Profiles",
            "⚠️ 20 Toxicity Endpoints",
            "🎯 Molecular Docking (AutoDock Vina)",
            "⬇️ Export Filtered Datasets"
        ]
        for i, feat in enumerate(features):
            with feat_cols[i % 3]:
                st.markdown(f'<div class="feature-item">{feat}</div>', unsafe_allow_html=True)

        # Detailed performance expander
        with st.expander("📊 View Detailed ML Performance (Cross‑Validation & External Validation)"):
            st.markdown("""
            | Model         | 5‑Fold CV AUC (mean ± SD) | External Accuracy | External AUC |
            |---------------|---------------------------|-------------------|--------------|
            | Random Forest | 0.9603 ± 0.0037           | 0.8810            | 0.9406       |
            | SVM           | 0.9548 ± 0.0054           | 0.8715            | 0.9250       |
            | KNN           | 0.9489 ± 0.0044           | 0.8580            | 0.9256       |
            | XGBoost       | 0.9635 ± 0.0037           | 0.8685            | 0.9345       |
            | LightGBM      | 0.9619 ± 0.0032           | 0.8585            | 0.9309       |
            | ExtraTrees    | 0.9619 ± 0.0036           | 0.8845            | 0.9411       |
            | **Ensemble**  | **0.9646 ± 0.0037**       | **0.8805**        | **0.9411**   |
            
            **Control docking benchmarks (AutoDock Vina, 8F1Y):**  
            - Best control (Poziotinib): −8.9 kcal/mol  
            - Mean of all control drugs: −7.98 kcal/mol
            """)

    st.markdown("---")

    # Sidebar filters (only shown in DB mode)
    st.sidebar.subheader("🔍 Filters")

    search_text = st.sidebar.text_input(
        "Search LOTUS ID / Name / Organism",
        placeholder="e.g., quercetin"
    )

    st.sidebar.subheader("🧪 Docking")
    docking_success_opts = ["All"] + safe_unique(df["docking_success"]) if "docking_success" in df else ["All"]
    docking_success = st.sidebar.selectbox("Docking success", docking_success_opts)

    better_best_opts = ["All"] + safe_unique(df["better_than_best_control"]) if "better_than_best_control" in df else ["All"]
    better_best = st.sidebar.selectbox("Better than best control", better_best_opts)

    better_mean_opts = ["All"] + safe_unique(df["better_than_mean_control"]) if "better_than_mean_control" in df else ["All"]
    better_mean = st.sidebar.selectbox("Better than mean control", better_mean_opts)

    st.sidebar.subheader("💊 ADME")
    gi_opts = ["All"] + safe_unique(df["gi_absorption"]) if "gi_absorption" in df else ["All"]
    gi_absorption = st.sidebar.selectbox("GI absorption", gi_opts)

    bbb_opts = ["All"] + safe_unique(df["bbb_permeant"]) if "bbb_permeant" in df else ["All"]
    bbb_permeant = st.sidebar.selectbox("BBB permeant", bbb_opts)

    st.sidebar.subheader("📊 Ranges")
    qed_min, qed_max = None, None
    if "qed_score" in df.columns and df["qed_score"].notna().any():
        qmin, qmax = float(df["qed_score"].min()), float(df["qed_score"].max())
        qed_min, qed_max = st.sidebar.slider("QED score", qmin, qmax, (qmin, qmax))

    dock_min, dock_max = None, None
    if "docking_affinity" in df.columns and df["docking_affinity"].notna().any():
        dmin, dmax = float(df["docking_affinity"].min()), float(df["docking_affinity"].max())
        dock_min, dock_max = st.sidebar.slider("Docking affinity", dmin, dmax, (dmin, dmax))

    st.sidebar.subheader("⚠️ Toxicity Flags")
    tox_flags = {}
    for tox_col in ["herg_blockers_prediction", "dili_prediction", "ames_toxicity_prediction",
                    "carcinogenicity_prediction", "hepatotoxicity_prediction"]:
        if tox_col in df.columns:
            opts = ["All"] + safe_unique(df[tox_col])
            tox_flags[tox_col] = st.sidebar.selectbox(
                tox_col.replace("_", " ").title(),
                opts,
                key=tox_col
            )

    # Apply filters
    filtered_df = df.copy()

    if search_text:
        q = search_text.strip().lower()
        mask = (
            filtered_df["lotus_id"].astype(str).str.lower().str.contains(q, na=False) |
            filtered_df["compound_name"].astype(str).str.lower().str.contains(q, na=False) |
            filtered_df["source_organism"].astype(str).str.lower().str.contains(q, na=False)
        )
        filtered_df = filtered_df[mask]

    if docking_success != "All" and "docking_success" in filtered_df:
        filtered_df = filtered_df[filtered_df["docking_success"] == docking_success]
    if better_best != "All" and "better_than_best_control" in filtered_df:
        filtered_df = filtered_df[filtered_df["better_than_best_control"] == better_best]
    if better_mean != "All" and "better_than_mean_control" in filtered_df:
        filtered_df = filtered_df[filtered_df["better_than_mean_control"] == better_mean]
    if gi_absorption != "All" and "gi_absorption" in filtered_df:
        filtered_df = filtered_df[filtered_df["gi_absorption"] == gi_absorption]
    if bbb_permeant != "All" and "bbb_permeant" in filtered_df:
        filtered_df = filtered_df[filtered_df["bbb_permeant"] == bbb_permeant]

    if qed_min is not None and "qed_score" in filtered_df:
        filtered_df = filtered_df[filtered_df["qed_score"].between(qed_min, qed_max)]
    if dock_min is not None and "docking_affinity" in filtered_df:
        filtered_df = filtered_df[
            filtered_df["docking_affinity"].isna() |
            filtered_df["docking_affinity"].between(dock_min, dock_max)
        ]

    for col, val in tox_flags.items():
        if val != "All" and col in filtered_df:
            filtered_df = filtered_df[filtered_df[col] == val]

    # Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total compounds", len(df))
    col2.metric("Filtered compounds", len(filtered_df))

    docked = (df["docking_success"] == "Yes").sum() if "docking_success" in df else 0
    col3.metric("Docked", docked)

    better_best_count = (df["better_than_best_control"] == "Yes").sum() if "better_than_best_control" in df else 0
    col4.metric("Better than best control", better_best_count)

    st.markdown("---")

    st.subheader("📋 Compound Browser")
    browser_defaults = [
        "lotus_id", "compound_name", "source_organism",
        "ensemble_probability", "ensemble_prediction",
        "qed_score", "docking_affinity",
        "better_than_best_control", "better_than_mean_control", "docking_success"
    ]
    available_browser_cols = [c for c in browser_defaults if c in filtered_df.columns]

    # Download button only
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered results as CSV",
        data=csv,
        file_name="egfr_npdb_filtered.csv",
        mime="text/csv"
    )

    # Display table
    st.dataframe(filtered_df[available_browser_cols], height=400, use_container_width=True)

    st.markdown("---")

    st.subheader("🔬 Compound Details")
    lotus_ids = filtered_df["lotus_id"].dropna().astype(str).tolist()
    if lotus_ids:
        selected = st.selectbox("Select a LOTUS ID", lotus_ids)
        row = filtered_df[filtered_df["lotus_id"].astype(str) == selected].iloc[0]

        # Display chemical structure if SMILES available
        if "smiles" in row.index and pd.notna(row["smiles"]) and RDKIT_AVAILABLE:
            st.markdown("#### 🧪 Chemical Structure")
            smiles = row["smiles"]
            img = get_mol_image(smiles, size=(400, 300))
            if img:
                st.image(img, caption=f"2D structure of {row.get('compound_name', selected)}", width=400)
            else:
                st.warning("Could not generate structure image from SMILES.")

        groups = {
            "📌 Basic Information": [
                "lotus_id", "compound_name", "source_organism", "formula", "mw", "smiles"
            ],
            "⚖️ Physicochemical Properties": [
                "heavy_atoms", "aromatic_heavy_atoms", "fraction_csp3", "rotatable_bonds",
                "hbond_acceptors", "hbond_donors", "molar_refractivity", "tpsa"
            ],
            "💧 Lipophilicity & Solubility": [
                "ilogp", "xlogp3", "wlogp", "mlogp", "silicos_it_logp", "consensus_logp",
                "esol_logs", "esol_solubility_mg_ml", "esol_solubility_mol_l", "esol_class",
                "ali_logs", "ali_solubility_mg_ml", "ali_solubility_mol_l", "ali_class",
                "silicos_it_logsw", "silicos_it_solubility_mg_ml", "silicos_it_solubility_mol_l", "silicos_it_class"
            ],
            "🧪 Drug‑likeness": [
                "gi_absorption", "bbb_permeant", "pgp_substrate",
                "cyp1a2_inhibitor", "cyp2c19_inhibitor", "cyp2c9_inhibitor",
                "cyp2d6_inhibitor", "cyp3a4_inhibitor", "log_kp",
                "lipinski_violations", "ghose_violations", "veber_violations",
                "egan_violations", "muegge_violations", "bioavailability_score",
                "pains_alerts", "brenk_alerts", "leadlikeness_violations",
                "synthetic_accessibility", "qed_score"
            ],
            "⚙️ ML Predictions (EGFR Activity)": [
                "ensemble_probability", "ensemble_prediction",
                "rf_probability", "rf_prediction",
                "svm_probability", "svm_prediction",
                "knn_probability", "knn_prediction",
                "xgboost_probability", "xgboost_prediction",
                "lightgbm_probability", "lightgbm_prediction",
                "et_probability", "et_prediction"
            ],
            "🎯 Docking Results": [
                "docking_affinity", "docking_success",
                "better_than_best_control", "better_than_mean_control"
            ],
            "⚠️ Toxicity Predictions": [
                "herg_blockers_probability", "herg_blockers_prediction",
                "herg_blockers_10um_probability", "herg_blockers_10um_prediction",
                "dili_probability", "dili_prediction",
                "ames_toxicity_probability", "ames_toxicity_prediction",
                "rat_oral_acute_toxicity_probability", "rat_oral_acute_toxicity_prediction",
                "fdamdd_probability", "fdamdd_prediction",
                "skin_sensitization_probability", "skin_sensitization_prediction",
                "carcinogenicity_probability", "carcinogenicity_prediction",
                "eye_corrosion_probability", "eye_corrosion_prediction",
                "eye_irritation_probability", "eye_irritation_prediction",
                "respiratory_toxicity_probability", "respiratory_toxicity_prediction",
                "hepatotoxicity_probability", "hepatotoxicity_prediction",
                "neurotoxicity_di_probability", "neurotoxicity_di_prediction",
                "ototoxicity_probability", "ototoxicity_prediction",
                "hematotoxicity_probability", "hematotoxicity_prediction",
                "nephrotoxicity_di_probability", "nephrotoxicity_di_prediction",
                "genotoxicity_probability", "genotoxicity_prediction",
                "rpmi_8226_immunotoxicity_probability", "rpmi_8226_immunotoxicity_prediction",
                "a549_cytotoxicity_probability", "a549_cytotoxicity_prediction",
                "hek293_cytotoxicity_probability", "hek293_cytotoxicity_prediction"
            ]
        }

        for group_name, fields in groups.items():
            existing_fields = [f for f in fields if f in row.index]
            if existing_fields:
                with st.expander(group_name, expanded=(group_name == "📌 Basic Information")):
                    cols = st.columns(2)
                    for i, field in enumerate(existing_fields):
                        with cols[i % 2]:
                            st.write(f"**{field.replace('_', ' ').title()}:** {row[field]}")
    else:
        st.info("No compounds match the current filters.")

# ------------------------------
# MODE 2: Bioactivity Predictor
# ------------------------------
elif mode == "🧪 Bioactivity Predictor":
    st.title("🧪 EGFR Bioactivity Predictor")
    st.markdown("Predict EGFR inhibitory activity and molecular properties from SMILES strings.")
    st.markdown("---")

    # Short intro box for predictor
    st.markdown("""
    <div class="short-intro-box">
        <p><strong>Quick Guide:</strong> Enter or upload SMILES strings to receive EGFR bioactivity predictions along with molecular properties.</p>
    </div>
    """, unsafe_allow_html=True)

    if not RDKIT_AVAILABLE:
        st.error("❌ RDKit is not installed. Please install it to use this feature: `pip install rdkit`")
        st.stop()

    st.markdown("### 📥 Input SMILES")

    input_method = st.radio("Input method", ["Paste SMILES", "Upload CSV"])

    smiles_list = []

    if input_method == "Paste SMILES":
        smiles_text = st.text_area("Paste SMILES here (one per line, up to 1000)", height=200)
        if smiles_text:
            smiles_list = [s.strip() for s in smiles_text.splitlines() if s.strip()]
            if len(smiles_list) > 1000:
                st.warning("⚠️ More than 1000 SMILES entered. Only the first 1000 will be processed.")
                smiles_list = smiles_list[:1000]
    else:
        uploaded_file = st.file_uploader("Upload CSV file with 'SMILES' column", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if 'SMILES' not in df_upload.columns:
                    st.error("CSV must contain a 'SMILES' column.")
                else:
                    smiles_list = df_upload['SMILES'].dropna().astype(str).tolist()
                    if len(smiles_list) > 1000:
                        st.warning("⚠️ More than 1000 SMILES found. Only the first 1000 will be processed.")
                        smiles_list = smiles_list[:1000]
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if st.button("🚀 Predict Bioactivity & Compute Properties") and smiles_list:
        # Initialize progress tracking
        total_smiles = len(smiles_list)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        results = []
        
        with st.spinner("Generating fingerprints and running predictions..."):
            for idx, smi in enumerate(smiles_list):
                # Process each SMILES
                smi = smi.strip()
                if not smi:
                    continue
                
                row = {'SMILES': smi}
                
                # Add molecular properties
                props = compute_molecular_properties(smi)
                if props is not None:
                    row.update(props)
                
                fp = smiles_to_fingerprint(smi)
                if fp is None:
                    row['Error'] = 'Invalid SMILES'
                    results.append(row)
                else:
                    # Apply feature selection
                    if feature_indices is not None:
                        fp_selected = fp[feature_indices]
                    else:
                        fp_selected = fp

                    fp_2d = fp_selected.reshape(1, -1)

                    for model_name, model in models.items():
                        if model is None:
                            row[f'{model_name}_prob'] = None
                            row[f'{model_name}_pred'] = None
                            continue

                        try:
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(fp_2d)[0, 1]
                            elif hasattr(model, "decision_function"):
                                df_val = model.decision_function(fp_2d)[0]
                                proba = 1.0 / (1.0 + np.exp(-df_val))
                            else:
                                proba = None
                            pred = model.predict(fp_2d)[0]
                        except Exception as e:
                            proba = None
                            pred = None

                        row[f'{model_name}_prob'] = proba
                        row[f'{model_name}_pred'] = pred

                    results.append(row)
                
                # Update progress bar
                current_progress = (idx + 1) / total_smiles
                progress_bar.progress(current_progress)
                percentage = int(current_progress * 100)
                progress_text.text(f"📊 Progress: {percentage}% ({idx + 1}/{total_smiles} compounds processed)")
        
        results_df = pd.DataFrame(results)

        st.success(f"✅ Predictions completed for {len(results_df)} compounds.")
        
        # Display columns for preview
        display_cols = ['SMILES', 'QED', 'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 
                        'LipinskiViolations', 'ensemble_prob', 'ensemble_pred']
        available_display = [c for c in display_cols if c in results_df.columns]
        
        # Download button for preview
        csv_data = results_df[available_display].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name="bioactivity_predictions.csv",
            mime="text/csv"
        )
        
        # Display table
        st.dataframe(results_df[available_display], height=400, use_container_width=True)

        # Full results expander
        with st.expander("🔍 View Full Results (All Models & Properties)"):
            st.dataframe(results_df, use_container_width=True)

        # Download full results
        csv_full = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Complete Results as CSV",
            data=csv_full,
            file_name="bioactivity_adme_predictions.csv",
            mime="text/csv"
        )

        # Show summary
        if 'ensemble_pred' in results_df.columns:
            active_count = (results_df['ensemble_pred'] == 1).sum()
            st.metric("Predicted Active (Ensemble)", active_count)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <div class="footer-section">
        <span class="names">Sheikh Sunzid Ahmed</span> and <span class="names">M. Oliur Rahman</span><br>
        Plant Taxonomy and Ethnobotany Laboratory, Department of Botany, University of Dhaka
    </div>
    <div class="footer-section">
        Built with Streamlit, SQLite, and RDKit | Ensemble ML Models (RF, SVM, KNN, XGBoost, LightGBM, ExtraTrees)
    </div>
    <div class="footer-section">
        Data Source: LOTUS Initiative (276,518 natural products) | Docking: AutoDock Vina | Target: EGFR (PDB: 8F1Y)
    </div>
    <div class="footer-section" style="opacity:0.7; margin-top:1rem;">
        © 2026 MIND-EGFR — ML‑Guided Natural Product Database
    </div>
</div>
""", unsafe_allow_html=True)
