import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF Question Answering Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# Load embedding model and LLM
@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embeddings, tokenizer, model


# Process uploaded PDF
@st.cache_resource
def create_vectorstore(pdf_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(pages)

    embeddings, _, _ = load_models()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# PDF upload
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    with st.spinner("Processing your PDF..."):

        vectorstore = create_vectorstore(
            uploaded_file.getvalue()
        )

    st.success("PDF processed successfully!")

    # User question
    query = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("Get Answer"):

        if query.strip():

            with st.spinner("Generating answer..."):

                # Retrieve relevant documents
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

                results = retriever.invoke(query)

                # Create context
                context = "\n\n".join(
                    f"Page {doc.metadata.get('page', 0) + 1}:\n"
                    f"{doc.page_content}"
                    for doc in results
                )

                # Create prompt
                prompt = f"""
                You are an AI tutor.

                Answer the question using only the context below.
                Explain in simple English.
                If the context does not contain the answer,
                say that the information is not available.

                Context:
                {context}

                Question:
                {query}

                Answer:
                """

                # Load LLM
                _, tokenizer, model = load_models()

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True
                )

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150
                )

                answer = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            # Display answer
            st.subheader("🤖 Answer")

            st.write(answer)

            # Display retrieved sources
            with st.expander("View retrieved PDF content"):

                for doc in results:

                    page = doc.metadata.get("page", 0) + 1

                    st.write(f"**Page {page}**")

                    st.write(doc.page_content)

                    st.divider()

        else:

            st.warning("Please enter a question.")