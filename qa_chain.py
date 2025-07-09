# qa_chain.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai import Mistral

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_NAME = "mistral-medium"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(docs, db_path):
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(db_path)
    return vectordb

def query_llm(prompt: str) -> str:
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model=MISTRAL_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def run_rag_analysis(resume_chunks, jd_chunks):
    resume_db = build_vectorstore(resume_chunks, "data/db/resume_db")
    jd_db = build_vectorstore(jd_chunks, "data/db/jd_db")

    # cross-query: JD → Resume and vice versa
    jd_context = "\n".join([doc.page_content for doc in resume_db.similarity_search(" ".join([d.page_content for d in jd_chunks]), k=4)])
    resume_context = "\n".join([doc.page_content for doc in jd_db.similarity_search(" ".join([d.page_content for d in resume_chunks]), k=4)])

    prompt = f"""
You are an HR assistant. Analyze the resume and job description below and return:
1. Match Percentage (0–100%)
2. Summary of match
3. Missing Skills
4. Suggestions

[RESUME SNIPPETS]
{jd_context}

[JOB DESCRIPTION SNIPPETS]
{resume_context}
"""
    return query_llm(prompt)
