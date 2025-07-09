# 📄 Resume + Job JD Analyzer

**Live Demo:** 👉 [https://resumexyz.streamlit.app](https://resumexyz.streamlit.app)

A smart AI assistant that compares your **Resume** with any **Job Description** and provides:

- ✅ Match Percentage  
- ✅ Summary of Strengths  
- ✅ Missing Skills / Qualifications  
- ✅ Suggestions to Improve

---

## 🚀 Features

- 📄 Upload Resume and JD (PDF or DOCX)
- 🧠 Uses **Mistral LLM API** directly (no HuggingFace)
- 🧩 Smart prompt engineering to generate detailed analysis
- 💡 Built with **Streamlit** for an interactive UI
- 🔐 API key managed securely with `.env` file

---

## 🖥️ Tech Stack

- 🦜 LangChain (text splitting)
- 🧠 Mistral LLM via [mistralai/client-python](https://github.com/mistralai/client-python)
- 📄 PyMuPDF + python-docx for file parsing
- ⚙️ Streamlit for frontend
- 📦 python-dotenv for API key handling

---

