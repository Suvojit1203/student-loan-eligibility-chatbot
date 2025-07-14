# RAG Chatbot for Student Loan Eligibility using Hugging Face Embeddings and LLM

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import streamlit as st

# Load the dataset
df = pd.read_csv("student_loan_dataset.csv")
df.fillna("unknown", inplace=True)

# Convert each row to a summary string
def row_to_summary(row):
    return f"Student {row['Student_ID']} is a {row['Age']} year old {row['Degree']} student with a GPA of {row['GPA']}, income ₹{row['Income']}, and credit score {row['Credit_Score']}. Requested loan: ₹{row['Loan_Amount']}. Loan was {row['Loan_Status'].lower()}."

documents = df.apply(row_to_summary, axis=1).tolist()

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)

# Index using FAISS
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# QA Model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def query_rag_bot(question, k=5):
    question_embedding = model.encode([question], convert_to_numpy=True)
    _, indices = index.search(question_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question based on the following student data:\n{context}\n\nQuestion: {question}"
    result = qa_model(prompt, max_new_tokens=256, do_sample=False)
    return result[0]['generated_text'].strip()

# Streamlit UI
st.set_page_config(page_title="Student Loan Q&A Chatbot", layout="centered")
st.markdown("""
<style>
    .big-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #004080;
        text-align: center;
        margin-top: 10px;
    }
    .suggestion-box {
        background-color: #f0f4ff;
        padding: 10px 20px;
        border-left: 5px solid #004080;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1.05rem;
    }
    .response-box {
        background-color: #f9f9f9;
        padding: 16px;
        border-left: 5px solid #28a745;
        border-radius: 6px;
        margin-top: 20px;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# Heading only once
st.markdown("<h1 class='big-title'>Student Loan Eligibility Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask any question about student loan eligibility based on academic and financial criteria.")

# Suggested questions (display only)
suggested_questions = [
    "What are the eligibility criteria for a student loan?",
    "Can a student with GPA 6.0 and income 15000 get a loan?",
    "Why was the loan rejected for STU045?",
    "Tell me about postgraduate students who were approved.",
    "What credit score is needed for approval?"
]

st.markdown("### 💡 Suggested Questions:")
for q in suggested_questions:
    st.markdown(f"<div class='suggestion-box'>{q}</div>", unsafe_allow_html=True)

# Single input box
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Generating answer..."):
        answer = query_rag_bot(question)
        st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ by Suvojit</p>", unsafe_allow_html=True)
