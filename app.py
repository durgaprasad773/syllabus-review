import streamlit as st
import pandas as pd
import numpy as np
import io
import re

from sentence_transformers import SentenceTransformer, util

# ==============================
# Load free embedding model
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ==============================
# Utility functions
# ==============================
def read_excel(file):
    try:
        return pd.read_excel(file)
    except:
        return None


def embed_text_list(text_list):
    text_list = ["" if pd.isna(t) else str(t) for t in text_list]
    return model.encode(text_list, convert_to_tensor=True, show_progress_bar=False)


def fix_quotes(text: str):
    """Convert 'word' → `word` while avoiding contractions."""
    if pd.isna(text):
        return text

    def repl(match):
        inner = match.group(1)
        return f"`{inner}`"

    return re.sub(r"'([A-Za-z0-9_+-]+)'", repl, str(text))


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="MCQ Review Tool (Free Embeddings)", layout="wide")

st.title("MCQ Review & Clean Tool (Open Source Embeddings — No Token Use)")

# Upload files
generated_file = st.file_uploader("Upload Generated MCQ Excel", type=["xlsx"])
master_file = st.file_uploader("Upload Master Questions Excel", type=["xlsx"])

st.markdown("---")

syllabus_text = st.text_area("Paste Syllabus (Optional)", height=200)

fix_quotes_toggle = st.checkbox("Convert 'break' → `break`", value=True)

sim_threshold = st.slider("Duplicate Detection Threshold", 0.50, 0.95, 0.75)
syllabus_threshold = st.slider("Syllabus Relevance Threshold", 0.30, 0.95, 0.55)

run_btn = st.button("Run Cleaning")

# ==============================
# MAIN PROCESSING
# ==============================
if run_btn:

    if generated_file is None:
        st.error("Please upload the Generated MCQ Excel first.")
        st.stop()

    # Load generated questions
    gen_df = read_excel(generated_file)
    if gen_df is None:
        st.error("Could not read generated MCQ file.")
        st.stop()

    st.success("Generated MCQ file loaded successfully.")
    st.write(gen_df.head())

    # Load master questions
    master_df = None
    if master_file:
        master_df = read_excel(master_file)
        st.success("Master file loaded.")
        st.write(master_df.head())

    # Detect question column
    q_col = None
    for col in gen_df.columns:
        if "question" in col.lower():
            q_col = col
            break
    if q_col is None:
        q_col = gen_df.columns[0]

    st.info(f"Detected question column: **{q_col}**")

    # Apply quote fix
    if fix_quotes_toggle:
        for col in gen_df.columns:
            if gen_df[col].dtype == object:
                gen_df[col] = gen_df[col].apply(fix_quotes)
        st.info("Quotes fixed across text columns.")

    # ==============================
    # Embedding Computation
    # ==============================
    questions = gen_df[q_col].astype(str).tolist()
    st.info("Embedding generated questions...")
    gen_embeds = embed_text_list(questions)

    remove_mask = np.zeros(len(gen_df), dtype=bool)

    # ==============================
    # 1️⃣ Master Similarity Filtering
    # ==============================
    if master_df is not None:
        m_col = None
        for col in master_df.columns:
            if "question" in col.lower():
                m_col = col
                break
        if m_col is None:
            m_col = master_df.columns[0]

        master_qs = master_df[m_col].astype(str).tolist()

        st.info("Embedding master questions...")
        master_embeds = embed_text_list(master_qs)

        st.info("Comparing with master questions...")
        sim_matrix = util.cos_sim(gen_embeds, master_embeds)

        max_similarities = sim_matrix.max(dim=1).values.cpu().numpy()

        duplicates = max_similarities >= sim_threshold
        remove_mask = remove_mask | duplicates

        st.success(f"Duplicate questions found & removed: {duplicates.sum()}")

    # ==============================
    # 2️⃣ Syllabus Filtering
    # ==============================
    if syllabus_text.strip():
        st.info("Checking syllabus relevance...")

        syllabus_emb = embed_text_list([syllabus_text])[0]

        sims = util.cos_sim(gen_embeds, syllabus_emb).cpu().numpy()

        outside_syllabus = sims < syllabus_threshold
        remove_mask = remove_mask | outside_syllabus.reshape(-1)

        st.success(f"Questions removed due to syllabus mismatch: {outside_syllabus.sum()}")

    # ==============================
    # 3️⃣ Final Cleanup
    # ==============================
    cleaned_df = gen_df[~remove_mask].reset_index(drop=True)

    st.success(f"Total questions removed: {remove_mask.sum()}")
    st.subheader("Cleaned Questions Preview")
    st.dataframe(cleaned_df.head(50))

    # Download
    buffer = io.BytesIO()
    cleaned_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="Download Cleaned Excel",
        data=buffer,
        file_name="cleaned_mcq.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.caption("App uses Sentence Transformers (free open-source embeddings) — zero token cost.")
 
