import os
import streamlit as st

from utils.pdf_loader import load_pdf
from utils.chunker import split_documents
from utils.vector_store import create_vector_store
from utils.rag import get_retriever
from utils.llm import get_llm

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF RAG Chatbot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join("uploads", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully!")

    # Load PDF
    documents = load_pdf(pdf_path)

    # Split into chunks
    chunks = split_documents(documents)

    # Create Vector Store
    st.write("Creating Vector Store...")
    create_vector_store(chunks)
    st.success("✅ Vector Store Created Successfully!")

    retriever = get_retriever()
    st.write("API Key Found:", bool(os.getenv("GOOGLE_API_KEY")))
    st.write("Secret Found:", "GOOGLE_API_KEY" in st.secrets)
    llm = get_llm()

    # PDF Information
    st.subheader("Number of Pages")
    st.write(len(documents))

    st.subheader("Number of Chunks")
    st.write(len(chunks))

    st.subheader("First Chunk")
    st.write(chunks[0].page_content)

    st.divider()

    # Chat Input
    question = st.chat_input("Ask a question about your PDF")

    if question:

        # Show User Message
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            prompt = f"""
Answer the question only from the given context.

If the answer is not present in the context, reply:
"I don't know based on the uploaded PDF."

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

        # Extract Answer
        answer = ""

        if isinstance(response.content, list):
            for item in response.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    answer += item["text"]
        else:
            answer = response.content

        # Show Assistant Message
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Retrieved Chunks
        st.divider()
        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(docs):
            st.markdown(f"### Chunk {i+1}")
            st.write(doc.page_content)