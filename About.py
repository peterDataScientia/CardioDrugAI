import streamlit as st
import os

def render():

    # =========================================================
    # CSS (SAFE ONLY)
    # =========================================================
    st.markdown("""
    <style>

    .about-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
        color: #111827;
    }

    .about-card {
        background: white;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.6;
    }

    .bio-card {
        background: #f8fafc;
        padding: 14px;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        font-size: 12px;
        line-height: 1.6;
        margin-top: 10px;
    }

    .dev-name {
        font-size: 17px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 4px;
    }

    .dev-role {
        font-size: 13px;
        color: #4b5563;
        margin-bottom: 6px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # TITLE
    # =========================================================
    # =========================================================
    # TABS
    # =========================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview",
        "🫀 Biological Insight",
        "👨‍💻 Technology",
        "👨‍🔬 Team"
    ])

    # =========================================================
    # OVERVIEW
    # =========================================================
    with tab1:
        st.markdown("""
        <div class="about-card">
        This platform is a machine learning-based virtual screening system designed to predict 
        potential <b>TNF-α inhibitors</b> from chemical structures (SMILES format).

        It supports early-stage drug discovery using QSAR modeling and molecular fingerprints.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # BIOLOGY INSIGHT
    # =========================================================
    with tab2:
        st.markdown("""
        <div class="bio-card">
        <b>TNF-α (Tumor Necrosis Factor-alpha)</b> is a pro-inflammatory cytokine involved in chronic disease.
        <br><br>
        <b>Biological roles:</b><br>
        • Atherosclerosis and vascular inflammation<br>
        • Endothelial dysfunction<br>
        • Heart failure progression<br>
        • Autoimmune disorders<br>
        <br>
        <b>Drug relevance:</b><br>
        TNF-α inhibition is a validated strategy for controlling inflammation.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # TECHNOLOGY
    # =========================================================
    with tab3:
        st.markdown("""
        <div class="about-card">
        Python • Streamlit • RDKit • Scikit-learn • PubChem Fingerprints • QSAR Modeling
        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # TEAM (FIXED + PROFESSIONAL LAYOUT)
    # =========================================================
    with tab4:

        def render_profile(name, role, affiliation, expertise, email, img_paths):

            col1, col2 = st.columns([1.3, 3])

            # ---------------- IMAGE ----------------
            with col1:
                shown = False
                for img in img_paths:
                    if os.path.exists(img):
                        st.image(img, width=160)
                        shown = True
                        break

                if not shown:
                    st.markdown("👤")

            # ---------------- DETAILS ----------------
            with col2:
                st.markdown(f"### {name}")
                st.markdown(f"**Position:** {role}")
                st.markdown(f"**Affiliation:** {affiliation}")
                st.markdown(f"**Expertise:** {expertise}")
                st.markdown(f"**Contact:** {email}")

        # =========================================================
        # PROFILE 1
        # =========================================================
        render_profile(
            name="Anthony Peter",
            role="Tutorial Assistant in Chemistry",
            affiliation="Department of Chemistry, Mkwawa University College of Education, University of Dar es Salaam, Tanzania",
            expertise="Pharmacology, Chemistry, QSAR Modeling, Cheminformatics, Machine Learning in Drug Discovery",
            email="anthony.peter@muce.ac.tz",
            img_paths=["developer1.jpg", "developer1.jpeg"]
        )

        st.divider()

        # =========================================================
        # PROFILE 2
        # =========================================================
        render_profile(
            name="Sarah Profess",
            role="Professor of Chemoinformatics",
            affiliation="Liberty University",
            expertise="Artificial Intelligence in Drug Discovery, Computational Chemoinformatics",
            email="sprofess@liberty.edu",
            img_paths=["developer2.jpeg", "developer2.jpg"]
        )

    # =========================================================
    # BACK BUTTON
    # =========================================================
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬅ Back to Screening"):
        st.session_state.page = "screen"
        st.rerun()