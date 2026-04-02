# LangGraph RAG Chatbot with Tools

An AI-powered chatbot built using **LangGraph**, **RAG (Retrieval-Augmented Generation)**, and **tool calling**, featuring a modular backend and frontend architecture.

## 🚀 Features

- **RAG Pipeline**
  - PDF ingestion and document parsing
  - Context retrieval for question answering
  - Thread-aware conversations

- **Tool Calling**
  - Web search via DuckDuckGo
  - Calculator tool
  - Stock price lookup
  - Custom RAG tool integration

- **Backend + Frontend Separation**
  - Clean project structure with dedicated backend and frontend folders
  - Easy to extend and deploy

- **Environment-based Configuration**
  - API keys managed securely using `.env`

---

## 📂 Project Structure

```bash
ChatBot_deployed/
│── backend/
│   ├── langgraph_rag.py
│   └── __pycache__/   # ignored
│
│── frontend/
│   └── ... frontend files ...
│
│── arith_server.py
│── requirements.txt
│── .env               # ignored
│── .env.example
│── .gitignore
│── README.md
