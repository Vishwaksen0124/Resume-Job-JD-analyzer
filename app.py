import os
import streamlit as st
from dotenv import load_dotenv
from utils.doc_loader import load_text
from qa_chain import build_comparison_prompt, run_analysis

load_dotenv()

st.set_page_config(page_title="Resume & JD Analyzer")
st.title("📄 Resume + Job Description Analyzer")

resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])

if resume_file and jd_file and os.getenv("MISTRAL_API_KEY"):
    with st.spinner("Processing..."):
        resume_path = f"data/resumes/{resume_file.name}"
        jd_path = f"data/jds/{jd_file.name}"
        resume_docs = load_text(resume_file, resume_path)
        jd_docs = load_text(jd_file, jd_path)

        resume_text = "\n".join([doc.page_content for doc in resume_docs])
        jd_text = "\n".join([doc.page_content for doc in jd_docs])

        prompt = build_comparison_prompt(resume_text, jd_text)
        response = run_analysis(prompt)

        st.markdown("### 📊 Analysis Result")
        st.write(response)
else:
    st.warning("Please upload both the files")
