# 🤖 ML RAG System – Machine Learning Notes Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about Machine Learning study notes and receive answers based only on the information available in the uploaded PDF.

The project combines **Cohere, LangChain, ChromaDB, and Streamlit** to build a simple document-based question-answering system.

## 🚀 Features

* 📄 Loads Machine Learning notes from a PDF
* ✂️ Splits the document into smaller text chunks
* 🔍 Performs semantic search using Cohere embeddings
* 🗄️ Stores document embeddings in ChromaDB
* 🤖 Generates answers using Cohere's language model
* 💬 Provides an interactive Streamlit chatbot interface
* 🛡️ Reduces hallucination by instructing the chatbot to answer only from the provided notes
* ❌ Returns a suitable response when information is not available in the document

## 🏗️ RAG Architecture

```text
                PDF Study Notes
                       │
                       ▼
                PDF Document Loader
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
              Cohere Embeddings
                       │
                       ▼
                  ChromaDB
                Vector Database
                       │
                       │
User Question ────────►│
                       ▼
                Similarity Search
                       │
                       ▼
             Relevant Document Chunks
                       │
                       ▼
                Cohere LLM
                       │
                       ▼
                 Final Answer
                       │
                       ▼
              Streamlit Chatbot
```

## 🛠️ Technologies Used

* **Python**
* **Cohere**
* **LangChain**
* **ChromaDB**
* **Streamlit**
* **PyPDF**
* **python-dotenv**

## 📂 Project Structure

```text
ML-RAG-System/
│
├── data/
│   └── Introduction_to_ML_Notes.pdf
│
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> `chroma_db/` is generated locally during document ingestion and is excluded from GitHub.

> `.env` contains the Cohere API key and is excluded from GitHub for security.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Visalakshi05-ops/ML-RAG-System.git
```

```bash
cd ML-RAG-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Cohere API key

Create a `.env` file in the project root:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

Do not upload the `.env` file to GitHub.

### 4. Add your PDF

Place the PDF inside the `data/` folder.

The default project uses:

```text
data/Introduction_to_ML_Notes.pdf
```

### 5. Create the vector database

Run:

```bash
python ingest.py
```

This loads the PDF, creates text chunks, generates embeddings, and stores them in ChromaDB.

### 6. Start the chatbot

Run:

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 💬 Example Questions

You can ask questions such as:

```text
What are the four types of Machine Learning?

What is supervised learning?

What is the difference between classification and regression?

What are some real-world applications of Machine Learning?
```

The chatbot is designed to respond that the information cannot be found in the provided ML notes when a question is outside the document.

## 🔐 Security

The Cohere API key is stored in `.env` and is intentionally excluded from GitHub using `.gitignore`.

The repository contains `.env.example` as a template:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

Never commit or share your real API key.

## 🎯 Project Objective

The main objective of this project is to understand and implement a basic **Retrieval-Augmented Generation (RAG)** pipeline that connects document retrieval with a Large Language Model to provide grounded question-answering.

## 🔮 Future Enhancements

* Support multiple PDF documents
* Add PDF upload directly through the Streamlit interface
* Display document/page sources for each answer
* Add conversation history
* Improve retrieval using reranking
* Add support for different document formats
* Deploy the chatbot as a web application

## 👩‍💻 Author

**Visalakshi Ganaparthi**

B.Tech – Electronics & Communication Engineering

GitHub: https://github.com/Visalakshi05-ops
