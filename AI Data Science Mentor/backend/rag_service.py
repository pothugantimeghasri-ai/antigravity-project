import os
import re
import numpy as np

# Check if transformers should be disabled (useful if Hugging Face is blocked or slow to download)
DISABLE_TRANSFORMERS = os.getenv("DISABLE_TRANSFORMERS", "false").lower() == "true"
USE_EMBEDDINGS = False
model = None

if not DISABLE_TRANSFORMERS:
    try:
        from sentence_transformers import SentenceTransformer
        print("RAG Service: Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        USE_EMBEDDINGS = True
        print("RAG Service: Successfully loaded SentenceTransformer model 'all-MiniLM-L6-v2'")
    except Exception as e:
        print(f"RAG Service: SentenceTransformer failed to load or download ({e}). Falling back to local TF-IDF keyword search.")
else:
    print("RAG Service: Transformers disabled via environment configuration. Using local TF-IDF keyword search.")

# In-memory document storage: list of dicts with {"text": str, "source": str, "embedding": np.ndarray/None}
vector_db = []
documents_db = {}

# Pre-seeded Data Science Mentor Knowledge Base
DEFAULT_DOCUMENTS = [
    {
        "source": "data_science_fundamentals.txt",
        "text": (
            "Handling missing records and missing values is a crucial step in data preprocessing. "
            "In Pandas, we can identify missing records using df.isnull() or df.isna(). "
            "To handle missing records, we can drop them using df.dropna(), which removes any rows with null values. "
            "Alternatively, we can impute missing values using df.fillna(). For instance, filling null values with the "
            "median using df['column'].fillna(df['column'].median()) or the mean using df['column'].fillna(df['column'].mean()). "
            "Using inplace=True or reassigning the DataFrame is necessary to persist these changes. "
            "Advanced imputation includes forward fill (ffill) or backward fill (bfill) for time-series data."
        )
    },
    {
        "source": "overfitting_guide.txt",
        "text": (
            "Overfitting is when a model fits the training data too closely. We can prevent overfitting by using regularization techniques. "
            "This is a very important concept in machine learning and data science. Bias and variance trade-off is essential to get the best performance."
        )
    }
]

def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 100):
    """Splits a document text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    # Simple sliding window chunker
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if len(chunk_words) > 20: # Only save meaningful chunks
            chunks.append({
                "text": chunk_text,
                "source": source
            })
        if i + chunk_size >= len(words):
            break
            
    return chunks

def calculate_tf_idf_similarity(query: str, doc_text: str) -> float:
    """Fallback text similarity scorer (token intersection over union / TF-IDF style weighting)"""
    query_words = set(re.findall(r'\w+', query.lower()))
    doc_words = re.findall(r'\w+', doc_text.lower())
    doc_words_set = set(doc_words)
    
    if not query_words or not doc_words:
        return 0.0
        
    intersection = query_words.intersection(doc_words_set)
    # Simple Jaccard similarity as fallback
    score = len(intersection) / len(query_words.union(doc_words_set))
    
    # Add extra weight if query terms match consecutive phrases
    for qw in query_words:
        if qw in doc_text.lower():
            score += 0.05
            
    return min(score, 1.0)

def add_document(text: str, source: str):
    """Chunks a document and adds it to the local vector DB."""
    global vector_db, documents_db
    documents_db[source] = text
    chunks = chunk_text(text, source)
    
    for chunk in chunks:
        embedding = None
        if USE_EMBEDDINGS and model is not None:
            try:
                embedding = model.encode(chunk["text"])
            except Exception as e:
                print(f"Error encoding chunk: {e}")
                
        vector_db.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "embedding": embedding
        })
    print(f"RAG Service: Ingested document '{source}' -> Split into {len(chunks)} chunks.")

def search_rag(query: str, top_k: int = 2):
    """Searches the vector database for the most relevant context chunks."""
    if not vector_db:
        return []
        
    results = []
    
    if USE_EMBEDDINGS and model is not None:
        try:
            query_emb = model.encode(query)
            for item in vector_db:
                if item["embedding"] is not None:
                    # Cosine similarity
                    dot_product = np.dot(query_emb, item["embedding"])
                    norm_q = np.linalg.norm(query_emb)
                    norm_d = np.linalg.norm(item["embedding"])
                    similarity = dot_product / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0
                    results.append((item, float(similarity)))
                else:
                    # Fallback if specific item has no embedding
                    similarity = calculate_tf_idf_similarity(query, item["text"])
                    results.append((item, similarity))
        except Exception as e:
            print(f"Search vector error ({e}). Falling back to TF-IDF matching.")
            for item in vector_db:
                similarity = calculate_tf_idf_similarity(query, item["text"])
                results.append((item, similarity))
    else:
        # Standard keyword matching
        for item in vector_db:
            similarity = calculate_tf_idf_similarity(query, item["text"])
            results.append((item, similarity))
            
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in results[:top_k] if match[1] > 0.18]

# Initialize and pre-seed vector database
def initialize_database():
    if DEFAULT_DOCUMENTS:
        print("RAG Service: Seeding default guides...")
        for doc in DEFAULT_DOCUMENTS:
            add_document(doc["text"], doc["source"]) 
    else:
        print("RAG Service: No default guides to seed.")    
    # Check and load user's custom study notes PDF from downloads
    # (Disabled per user request to only keep uploaded files/datasets)
    pass

def list_documents() -> list:
    """Returns a list of unique document sources currently in the vector DB."""
    sources = list(set(item["source"] for item in vector_db))
    # Build summary: source -> chunk count
    source_counts = {}
    for item in vector_db:
        src = item["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    return [{"source": src, "chunks": source_counts[src]} for src in sorted(source_counts.keys())]

# Seed immediately on import
initialize_database()
