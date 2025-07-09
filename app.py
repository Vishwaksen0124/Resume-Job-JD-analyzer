# app.py
import streamlit as st
import os
from utils.doc_loader import load_document, chunk_documents
from qa_chain import run_rag_analysis

st.set_page_config(page_title="🧠 Resume + JD Analyzer (RAG)")
st.title("📄 Resume + Job Description Analyzer (RAG)")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if resume_file and jd_file:
    with st.spinner("Processing files..."):
        # Save temporarily
        os.makedirs("temp", exist_ok=True)
        resume_path = os.path.join("temp", resume_file.name)
        jd_path = os.path.join("temp", jd_file.name)
        with open(resume_path, "wb") as f: f.write(resume_file.read())
        with open(jd_path, "wb") as f: f.write(jd_file.read())

        resume_docs = load_document(resume_path)
        jd_docs = load_document(jd_path)

        resume_chunks = chunk_documents(resume_docs)
        jd_chunks = chunk_documents(jd_docs)

        response = run_rag_analysis(resume_chunks, jd_chunks)

    st.markdown("### ✅ Analysis Output")
    st.write(response)
else:
    st.info("Please upload both Resume and Job Description to begin.")
