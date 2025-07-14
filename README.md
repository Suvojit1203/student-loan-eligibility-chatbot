# 🧠 Student Loan Eligibility Chatbot

A smart Q&A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about student loan eligibility. Built with Streamlit, FAISS, SentenceTransformers, and Hugging Face Transformers.

---

## 📘 Project Description

**Student Loan Eligibility Chatbot** is an AI-powered Q&A assistant that helps users understand loan approval criteria based on student academic and financial profiles. Built using a Retrieval-Augmented Generation (RAG) pipeline, it leverages a dataset of 100+ mock student records and answers real-time natural language questions.

This project combines:
- **Sentence Transformers** for embedding student records
- **FAISS** for fast document retrieval
- **Flan-T5 (Hugging Face Transformers)** for intelligent answer generation
- **Streamlit** for a clean, interactive UI

It’s ideal for:
- Demonstrating real-world RAG applications
- Exploring NLP + tabular data integration
- Educational and AI internship projects

---

## 📊 What It Does

This chatbot uses a dataset of student profiles to answer questions like:

- "Can a student with GPA 6.0 and income 15000 get a loan?"
- "Why was the loan rejected for STU045?"
- "What are the eligibility criteria for a student loan?"

---

## 🚀 How to Run

1. **Clone the repo:**
```bash
git clone https://github.com/<your-username>/student-loan-eligibility-chatbot.git
cd student-loan-eligibility-chatbot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run app.py
```

---

## 🧾 Files Included

- `app.py` — Streamlit frontend
- `rag_chatbot.py` — RAG logic using FAISS + HuggingFace
- `student_loan_dataset.csv` — Mock dataset of student loan applicants
- `requirements.txt` — Python package dependencies
- `README.md` — You’re reading it

---

## 👨‍💻 Author

Made with ❤️ by Suvojit
