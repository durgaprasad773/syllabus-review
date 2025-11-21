import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MCQ Reviewer App", layout="wide")
st.title("MCQ Reviewer with OpenAI & BGE Local Similarity")

# --- Input Section ---
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.file_uploader("Upload MCQ CSV (must have 'Question' column)", type=["csv"])
bulk_master = st.text_area("Bulk Master Questions (one per line)")
syllabus_text = st.text_area("Syllabus Content")

if st.button("Run Review") and uploaded_file and openai_api_key and bulk_master and syllabus_text:
    df = pd.read_csv(uploaded_file)
    if "Question" not in df.columns:
        st.error("'Question' column not found in the uploaded CSV.")
        st.stop()

    # --- Step 1: Similarity/Duplicates Removal ---
    st.write("Loading BGE-M3 embedding model, please wait...")
    model = SentenceTransformer("BAAI/bge-m3")  # This downloads and caches model locally

    sheet_questions = df["Question"].astype(str).tolist()
    master_questions = [q.strip() for q in bulk_master.split("\n") if q.strip()]
    master_embeddings = model.encode(master_questions, batch_size=64, normalize_embeddings=True)
    sheet_embeddings = model.encode(sheet_questions, batch_size=64, normalize_embeddings=True)
    similarity = cosine_similarity(sheet_embeddings, master_embeddings)
    sim_threshold = 0.75

    dup_indices = []
    dup_details = []
    for idx, sim_scores in enumerate(similarity):
        max_sim = max(sim_scores)
        match_idx = sim_scores.argmax()
        if max_sim >= sim_threshold:
            dup_indices.append(idx)
            dup_details.append({
                "Sheet Question": sheet_questions[idx],
                "Matched Master": master_questions[match_idx],
                "Similarity": f"{max_sim:.2f}"
            })

    df_nodup = df.drop(index=dup_indices).reset_index(drop=True)
    st.subheader(f"Duplicates Removed ({len(dup_indices)})")
    st.dataframe(pd.DataFrame(dup_details))

    # --- Step 2: Syllabus Validation via OpenAI ---
    def check_in_syllabus(question, syllabus, openai_key):
        prompt = f"Is the following question within the syllabus? Return only Yes or No.\n\nSyllabus:\n{syllabus}\n\nQuestion:\n{question}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                api_key=openai_key,
            )
            return "yes" in response.choices[0].message['content'].strip().lower()
        except Exception as e:
            return False

    out_indices = []
    out_details = []
    st.write("Checking syllabus alignment, may take a few seconds per question...")
    for i, row in df_nodup.iterrows():
        q_text = str(row["Question"])
        if not check_in_syllabus(q_text, syllabus_text, openai_api_key):
            out_indices.append(i)
            out_details.append({"Out-of-Syllabus Question": q_text})

    df_final = df_nodup.drop(index=out_indices).reset_index(drop=True)
    st.subheader(f"Out-of-Syllabus Questions Removed ({len(out_indices)})")
    st.dataframe(pd.DataFrame(out_details))

    # --- Step 3: Normalize Quotes ---
    def normalize_quotes(text):
        return str(text).replace("''", "`")

    for col in ['Question', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'explanation']:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(normalize_quotes)

    st.subheader(f"Final Cleaned MCQ Sheet ({df_final.shape[0]} questions remaining)")
    st.dataframe(df_final)

    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("Download Final CSV", csv, file_name="MCQ_cleaned.csv")

    # --- Stats Summary ---
    st.write(f"**Summary:**\n- {len(dup_indices)} duplicates removed\n- {len(out_indices)} out-of-syllabus removed\n- {df_final.shape[0]} questions remaining")

else:
    st.info("Upload a CSV, enter your OpenAI key, paste master questions and syllabus, then click 'Run Review'.")
