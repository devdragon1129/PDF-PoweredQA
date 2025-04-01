import streamlit as st
from utils import extract_text_from_pdf, create_vectorstore
from qa import load_qa_chain

st.set_page_config(page_title="PDF ChatBot", layout="centered")
st.title("📄 Ask Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        create_vectorstore(raw_text)
        st.success("PDF processed. You can now ask questions!")

    question = st.text_input("Ask a question based on the PDF:")
    if question:
        with st.spinner("Searching for answer..."):
            qa_chain = load_qa_chain()
            answer = qa_chain.run(question)
            st.write("🧠 **Answer:**", answer)
