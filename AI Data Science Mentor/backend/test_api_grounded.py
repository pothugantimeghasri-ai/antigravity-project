import json
import pytest
from fastapi.testclient import TestClient
from main import app
import rag_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # Clear the vector db before each test to ensure a clean state
    rag_service.vector_db.clear()
    rag_service.documents_db.clear()
    yield

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("[OK] Health check test passed.")

def test_upload_and_ingestion():
    # Upload a test text document
    file_content = b"Overfitting occurs when a machine learning model learns noise. Regularization prevents overfitting. It is essential to balance bias and variance to get the best model performance."
    response = client.post(
        "/api/upload",
        files={"file": ("test_doc.txt", file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_doc.txt"
    assert "successfully ingested" in data["status"]
    print("[OK] Upload and ingestion test passed.")

def test_rag_query_not_found():
    # Query without any document uploaded
    response = client.post("/api/rag/query", json={"query": "What is overfitting?"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "I could not find the answer to this question in the uploaded documents" in data["reply"]
    assert data["grounded"] is False
    print("[OK] RAG Query not found test passed.")

def test_rag_query_grounded_success():
    # 1. Ingest test document
    rag_service.add_document(
        "Overfitting is when a model fits the training data too closely. We can prevent overfitting by using regularization techniques. This is a very important concept in machine learning and data science.",
        "overfitting_guide.txt"
    )
    
    # 2. Query something present in the document
    response = client.post("/api/rag/query", json={"query": "Explain overfitting and regularization"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "Overfitting is when a model fits the training data too closely" in data["reply"]
    assert "regularization techniques" in data["reply"]
    assert data["grounded"] is True
    print("[OK] RAG Query grounded success test passed.")

def test_chat_grounded():
    # 1. Ingest test document
    rag_service.add_document(
        "Pandas introduces two main data structures: Series and DataFrame. DataFrames represent tabular spreadsheets. They are widely used for data manipulation and exploratory data analysis in python projects.",
        "pandas_guide.txt"
    )
    
    # 2. Chat query
    response = client.post("/api/chat", json={"message": "What is a DataFrame?"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "DataFrames represent tabular spreadsheets" in data["reply"]
    assert data["grounded"] is True
    print("[OK] Chat grounded query test passed.")

def test_learning_path_fallback():
    response = client.post("/api/v1/learning-path", json={"skills": ["Algorithms", "Data Science"]})
    assert response.status_code == 200
    data = response.json()
    assert data["mapped"] is True
    assert len(data["generated_path"]) == 4
    print("[OK] Learning Path Fallback test passed.")

def test_evaluate_dropna():
    response = client.post("/api/v1/evaluate", json={"submitted_code": "df.dropna()"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Needs Improvement"
    assert "inplace=True" in data["line_by_line_feedback"]
    assert "subset" in data["line_by_line_feedback"]
    print("[OK] Evaluate dropna test passed.")

def test_quiz():
    response = client.post("/api/v1/quiz", json={"topic": "Python"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "question" in data[0]
    print("[OK] Quiz generation API test passed.")

def test_projects():
    response = client.post("/api/v1/projects", json={"difficulty": "Beginner", "topic": "Pandas"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "title" in data[0]
    print("[OK] Project recommendations API test passed.")

if __name__ == "__main__":
    print("Running API validation tests...")
    test_health()
    test_upload_and_ingestion()
    test_rag_query_not_found()
    test_rag_query_grounded_success()
    test_chat_grounded()
    test_learning_path_fallback()
    test_evaluate_dropna()
    test_quiz()
    test_projects()
    print("All validation tests successfully passed!")
