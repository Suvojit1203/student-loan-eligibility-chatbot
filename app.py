import streamlit as st
from rag_chatbot import query_rag_bot

st.set_page_config(page_title="Student Loan Q&A Chatbot", layout="centered")
st.title("🎓 Student Loan Eligibility Chatbot")

st.markdown("Ask any question about student loan eligibility based on academic and financial criteria.")

question = st.text_input("Enter your question here:")

if question:
    with st.spinner("Thinking..."):
        answer = query_rag_bot(question)
        st.success("Answer:")
        st.write(answer)