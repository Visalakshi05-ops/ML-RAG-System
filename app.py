import os
import streamlit as st

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma


# -----------------------------------
# Load environment variables
# -----------------------------------

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    st.error("COHERE_API_KEY not found in .env file")
    st.stop()


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="ML Notes RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("🤖 Machine Learning Notes Chatbot")

st.write(
    "Ask questions about the **Introduction to Machine Learning** study notes."
)


# -----------------------------------
# Load Cohere Embeddings
# -----------------------------------

embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-v4.0"
)


# -----------------------------------
# Load ChromaDB
# -----------------------------------

vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="ml_notes",
    embedding_function=embeddings
)


# -----------------------------------
# Load Cohere Chat Model
# -----------------------------------

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-08-2024",
    temperature=0.2
)


# -----------------------------------
# Chat input
# -----------------------------------

question = st.chat_input(
    "Ask something about Machine Learning..."
)


# -----------------------------------
# RAG pipeline
# -----------------------------------

if question:

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    # Retrieve relevant documents
    documents = vectorstore.similarity_search(
        question,
        k=4
    )

    # Combine retrieved content
    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    # Prompt
    prompt = f"""
You are a helpful assistant that answers questions using ONLY
the provided Machine Learning study notes.

If the answer cannot be found in the provided notes,
say:

"I couldn't find this information in the provided ML notes."

Do not invent information.

Study Notes:
{context}

Question:
{question}

Answer clearly and in simple language.
"""

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the ML notes..."):

            response = llm.invoke(prompt)

            st.write(response.content)