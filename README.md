# 🎓 AI Data Science Mentor

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pothugantimeghasri-ai/antigravity-project)
[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=pothugantimeghasri-ai/antigravity-project&branch=main&mainModule=frontend/app.py)

Welcome to the **AI Data Science Mentor**, a premium, state-of-the-art interactive learning portal, code evaluator, and study planner designed to guide students from raw foundations to production-ready Machine Learning engineering.

Built with a **FastAPI backend** (supporting local RAG, token keyword similarity, and Ollama integration) and a custom **Streamlit frontend** featuring a dark glassmorphic Cyberpunk design.

---

## 🚀 Cloud Deployment

You can deploy the app to the cloud using the deployment buttons above!

### 1. Deploy Backend (FastAPI) to Render
- Click the **Deploy to Render** button.
- Render will automatically read `render.yaml` and provision a web service running your FastAPI backend.
- Once deployed, copy your Render service URL (e.g., `https://your-backend-service.onrender.com`).

### 2. Deploy Frontend (Streamlit) to Streamlit Cloud
- Click the **Deploy on Streamlit** button.
- In the Streamlit deployment page, ensure the **Main file path** is:
  ```text
  frontend/app.py
  ```
- **Optional**: To link the Streamlit app to your deployed Render API, add your Render backend URL in the Streamlit Cloud dashboard under **App Settings -> Secrets**:
  ```toml
  BACKEND_URL = "https://your-backend-service.onrender.com"
  ```
  *(Note: If you do not specify a `BACKEND_URL`, the Streamlit frontend will automatically run all backend functionalities locally in the same container, requiring no separate backend server).*

---

## ✨ Features

- **🏠 Interactive Feature Hub**: A dynamic, autoplaying sliding carousel dashboard. Hover over any card for zoom transitions, and click cards directly to jump into specific agents.
- **📅 Learning Path Generator**: Enter your existing skills and available weeks to generate a week-by-week curriculum. Mapped to ingested materials when available.
- **💻 Code Evaluator**: Submit Python/Pandas operations to receive constructive critiques, code complexity feedback, and optimized alternatives (e.g. replacing slow row-loops with vectorized operations).
- **🧠 Interactive Quiz Sandbox**: Conceptual multiple-choice assessments covering Python, SQL, Statistics, and ML with comprehensive explanation logs.
- **🚀 Project Recommendations**: Custom project blueprints matching Beginner, Intermediate, and Advanced difficulties.
- **💬 Ask Mentor (RAG Document Q&A)**: Ask questions strictly grounded in your custom uploaded study guides, PDFs, CSVs, or markdown files.
- **📚 Knowledge Base**: Ingest custom learning resources, monitor chunk count, and manage your vector database store.
- **📊 Interactive Visualization**: Upload datasets, select columns, and instantly generate correlation heatmaps, box plots, scatter matrices, and AI-driven distribution summaries.
- **🎙️ Voice Mentor**: Practice voice interviews by recording spoken answers directly in the UI and receiving text & spoken audio feedback.
- **🤝 Interview Trainer**: Simulated chat-based technical interviews covering modeling, statistics, or SQL with a graded scorecard report.
- **💻 Challenge Generator**: Write code inside an interactive sandbox and run live verification tests to grade your solution.

---

## 📁 Project Structure

```
antigravity-project/
├── backend/
│   ├── main.py              # FastAPI endpoints (evaluator, quizzes, challenges, interview mock engines)
│   ├── main_grounded.py      # Standalone grounded FastAPI backend logic
│   ├── rag_service.py       # RAG Vector Database, chunking logic, TF-IDF keyword & embeddings search
│   └── test_api.py          # Pytest API validation suite (18 automated tests)
├── frontend/
│   ├── app.py               # Streamlit frontend with full interactive cyberpunk dashboard
│   ├── app_grounded.py      # Standalone grounded document QA interface
│   └── package.json         # Node package descriptors
├── test_notes.txt           # Sample study notes for testing overfitting/regularization ingestion
├── requirements.txt         # Consolidated python dependencies
├── render.yaml              # Render deployment configuration
└── README.md                # System documentation
```

---

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** (optional, for local Llama 3 model execution)

### 2. Backend Installation & Startup
Navigate to the `backend` directory:
```bash
cd backend
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn main_grounded:app --reload --port 8000
```
The API documentation will be available at `http://localhost:8000/docs`.

### 3. Frontend Installation & Startup
Navigate to the `frontend` directory:
```bash
cd ../frontend
pip install -r requirements.txt
```

Start the Streamlit application:
```bash
streamlit run app.py
```
The frontend portal will open at `http://localhost:8501`.

---

## 🛠️ API Reference Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health Check |
| `POST` | `/api/upload` | Ingest a PDF/TXT/CSV/MD file into the RAG vector store |
| `GET` | `/api/rag/documents` | List all ingested sources and their chunk count |
| `POST` | `/api/rag/query` | Query the document-grounded RAG service |
| `POST` | `/api/chat` | Chat session grounded in retrieved document context |
| `POST` | `/api/v1/evaluate` | Evaluate student code submission for Pandas anti-patterns |
| `POST` | `/api/v1/learning-path` | Generate study syllabus matching skills and timeline |
| `POST` | `/api/v1/quiz` | Generate multiple choice assessments on selected topics |

---

## 🧪 Running Automated Tests

A comprehensive unit testing suite is provided under the backend module using `pytest`. It covers health checks, file uploads, RAG queries, code evaluator rules, and quiz generation.

To execute the tests:
```bash
cd backend
pytest
```
