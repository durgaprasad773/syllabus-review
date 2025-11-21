import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity

# ---- SETUP ----
st.title("MCQ Question Reviewer with AI Similarity & Syllabus Check")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("OpenAI API Key is required for syllabus check.")

uploaded_file = st.file_uploader("Upload MCQ CSV (questions, options, answer, explanation)", type=["csv"])
bulk_master = st.text_area("Paste Master Questions (one per line)")
syllabus_text = st.text_area("Paste Syllabus Content")

run_btn = st.button("Run Review")

if run_btn and uploaded_file and bulk_master and syllabus_text and openai_api_key:

    # Load data
    df = pd.read_csv(uploaded_file)
    master_questions = [q.strip() for q in bulk_master.split('\n') if q.strip()]
    syllabus = syllabus_text.strip()

    # ---- EMBEDDINGS ----
    # Load locally, no API key needed
    model = SentenceTransformer("BAAI/bge-m3")  # Use BGE-M3 locally

    # For similarity, use question strings only.
    sheet_questions = df["Question"].astype(str).tolist()
    master_embeddings = model.encode(master_questions, batch_size=64, normalize_embeddings=True)
    sheet_embeddings = model.encode(sheet_questions, batch_size=64, normalize_embeddings=True)

    # ---- SIMILARITY MATCH ----
    similarity = cosine_similarity(sheet_embeddings, master_embeddings)
    sim_threshold = 0.75  # 75% similarity for duplicates

    # Find which sheet questions match to master questions
    duplicate_indices = []
    duplicate_details = []

    for idx, sim_scores in enumerate(similarity):
        max_sim = max(sim_scores)
        match_idx = sim_scores.argmax()
        if max_sim >= sim_threshold:
            duplicate_indices.append(idx)
            duplicate_details.append({
                "sheet": sheet_questions[idx],
                "master": master_questions[match_idx],
                "score": max_sim
            })

    # ---- REMOVE DUPLICATES ----
    df_nodup = df.drop(index=duplicate_indices).reset_index(drop=True)
    st.subheader("Duplicates Removed")
    st.write(f"{len(duplicate_indices)} duplicates found and removed.")
    st.table(pd.DataFrame(duplicate_details))

    # ---- SYLLABUS VALIDATION ----
    def check_in_syllabus(question, syllabus, openai_key):
        prompt = f"Is the following question within the syllabus? Return only Yes or No.\n\nSyllabus:\n{syllabus}\n\nQuestion:\n{question}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                api_key=openai_key,
            )
            return response.choices[0].message['content'].strip().lower().startswith('yes')
        except Exception as e:
            return False

    # Only validate non-duplicates
    st.subheader("Out-of-Syllabus Questions Removed")
    syllabus_indices = []
    outsyllabus_details = []
    for i, row in df_nodup.iterrows():
        q_text = str(row["Question"])
        in_syllabus = check_in_syllabus(q_text, syllabus, openai_api_key)
        if not in_syllabus:
            syllabus_indices.append(i)
            outsyllabus_details.append({
                "question": q_text
            })
    df_final = df_nodup.drop(index=syllabus_indices).reset_index(drop=True)
    st.write(f"{len(syllabus_indices)} out-of-syllabus questions removed.")
    st.table(pd.DataFrame(outsyllabus_details))

    # ---- QUOTE NORMALIZATION ----
    def normalize_quotes(s):
        return str(s).replace("''", "`")
    for col in ['Question', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'explanation']:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(normalize_quotes)

    # ---- DOWNLOAD RESULT ----
    st.subheader("Final Cleaned MCQ Sheet")
    st.write(f"{df_final.shape[0]} questions remain after filtering.")
    st.dataframe(df_final)
    st.download_button(
        "Download Final CSV",
        df_final.to_csv(index=False).encode('utf-8'),
        file_name="MCQ_cleaned.csv"
    )
