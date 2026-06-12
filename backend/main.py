import os
import runpy
from collections import Counter
import re
import requests
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import rag_service

app = FastAPI(title="AI Data Science Mentor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Pydantic Schemas
class SkillInput(BaseModel):
    skills: List[str]
    course_source: Optional[str] = None

class ChatInput(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class RAGQueryInput(BaseModel):
    query: str
    top_k: Optional[int] = 3

class AssignmentSubmission(BaseModel):
    submitted_code: str
    assignment_context: Optional[str] = None

class QuizRequest(BaseModel):
    topic: str

class ProjectRequest(BaseModel):
    difficulty: str
    topic: str

class InterviewStartRequest(BaseModel):
    topic: str

class InterviewEvaluateRequest(BaseModel):
    topic: str
    question: str
    user_answer: str

class InterviewAssessRequest(BaseModel):
    topic: str
    questions: List[str]
    answers: List[str]
    evaluations: List[Dict[str, Any]]

class ChallengeGenerateRequest(BaseModel):
    language: str  # Python, SQL, Data Analysis
    level: str     # Beginner, Intermediate, Advanced

class ChallengeEvaluateRequest(BaseModel):
    language: str
    level: str
    title: str
    description: str
    user_code: str



# Mock Llama 3 responses for typical data science queries if Ollama is not running locally
def mock_llama3_response(message: str, context_chunks: List[dict]) -> str:
    msg_lower = message.lower()
    
    # Determine sources string if available
    source_str = ""
    if context_chunks:
        sources = list(set([c['source'] for c in context_chunks]))
        source_str = ", ".join(sources)
        
    no_match_text = "I could not find the answer to this question in the uploaded documents."

    if not context_chunks:
        return no_match_text

    # Extract sentences from the text chunks
    all_sentences = []
    for chunk in context_chunks:
        # simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 10:
                all_sentences.append((s_clean, chunk['source']))
                
    # Tokenize query to find matching keywords
    query_words = set([w.lower() for w in re.findall(r'\w+', msg_lower) if len(w) > 2])
    
    # Rank sentences by keyword match count
    matched_sentences = []
    for s, src in all_sentences:
        s_lower = s.lower()
        match_count = sum(1 for w in query_words if w in s_lower)
        if match_count > 0:
            matched_sentences.append((s, src, match_count))
            
    # Sort by match count descending
    matched_sentences.sort(key=lambda x: x[2], reverse=True)
    
    if matched_sentences and matched_sentences[0][2] >= 1:
        # Take top matched sentences up to a reasonable limit
        unique_matches = []
        seen = set()
        for s, src, score in matched_sentences:
            if s not in seen:
                seen.add(s)
                unique_matches.append((s, src))
            if len(unique_matches) >= 4:
                break
                
        # Group by source
        answers_by_src = {}
        for s, src in unique_matches:
            if src not in answers_by_src:
                answers_by_src[src] = []
            answers_by_src[src].append(s)
            
        lines = []
        for src, sents in answers_by_src.items():
            joined_sents = " ".join(sents)
            lines.append(joined_sents)
        
        return "\n\n".join(lines)
        
    return no_match_text


QUIZ_TOPIC_KEYWORDS = {
    "python": ["python", "list", "tuple", "dict", "exception", "copy", "class", "init"],
    "pandas": ["pandas", "dataframe", "csv", "fillna", "merge", "isnull", "read_csv"],
    "sql": ["sql", "database", "join", "having", "group by", "primary key", "acid"],
    "statistics": ["statistics", "probability", "mean", "median", "variance", "iqr", "skewness", "kurtosis"],
    "machine learning": ["machine learning", "regression", "classification", "f1", "r-squared", "sigmoid"],
    "deep learning": ["deep learning", "neural", "relu", "softmax", "sigmoid", "neuron"],
    "nlp": ["nlp", "language", "token", "embedding", "text", "transformer"],
    "data visualization": ["visualization", "plot", "chart", "graph", "matplotlib", "seaborn"],
    "big data": ["big data", "spark", "pyspark", "distributed", "hadoop"],
}


def _normalize_topic_key(topic: str) -> str:
    topic_lower = topic.lower()
    if "python" in topic_lower:
        return "python"
    if "pandas" in topic_lower:
        return "pandas"
    if "sql" in topic_lower or "database" in topic_lower:
        return "sql"
    if "statistics" in topic_lower or "probability" in topic_lower:
        return "statistics"
    if "deep learning" in topic_lower or "neural" in topic_lower:
        return "deep learning"
    if "nlp" in topic_lower or "natural language" in topic_lower:
        return "nlp"
    if "visualization" in topic_lower or "viz" in topic_lower:
        return "data visualization"
    if "big data" in topic_lower or "pyspark" in topic_lower or "spark" in topic_lower:
        return "big data"
    return "machine learning"


def _filter_quiz_questions_for_topic(topic: str, questions: list) -> list:
    topic_key = _normalize_topic_key(topic)
    keywords = QUIZ_TOPIC_KEYWORDS.get(topic_key, [])
    if not keywords:
        return questions

    filtered_questions = []
    for question in questions:
        combined_text = " ".join(
            str(question.get(field, "")) for field in ("question", "explanation", "correct_answer")
        ).lower()
        option_text = " ".join(str(option) for option in question.get("options", [])).lower()
        if any(keyword in combined_text or keyword in option_text for keyword in keywords):
            filtered_questions.append(question)

    return filtered_questions


COURSE_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "your", "course", "lesson",
    "module", "section", "topic", "topics", "overview", "introduction", "advanced", "learn", "using",
    "based", "data", "science", "learning", "guide", "notes", "content", "week", "weeks", "chapter",
    "part", "level", "beginner", "intermediate", "advanced"
}

JAVA_LEARNING_PATH = [
    {
        "week": 1,
        "topic": "Java Fundamentals & Core Programming Concepts",
        "details": "Build a strong foundation in Java syntax, variables, data types, operators, control flow, methods, and basic program structure. Practice writing small console programs to reinforce input, output, and logical thinking.",
    },
    {
        "week": 2,
        "topic": "Object-Oriented Programming (OOP) & Class Design",
        "details": "Learn classes, objects, constructors, inheritance, encapsulation, polymorphism, abstraction, and interfaces. Focus on designing reusable code with clean class structures and proper access modifiers.",
    },
    {
        "week": 3,
        "topic": "Collections Framework & Exception Handling",
        "details": "Study ArrayList, LinkedList, HashMap, HashSet, iterators, and generics. Pair that with try-catch blocks, custom exceptions, and safe error handling patterns for real programs.",
    },
    {
        "week": 4,
        "topic": "Advanced Java Development & Project Implementation",
        "details": "Explore file handling, streams, lambda expressions, and project structure. Finish with a small Java project that combines OOP, collections, and exception handling into one practical application.",
    },
]

PYTHON_LEARNING_PATH = [
    {
        "week": 1,
        "topic": "Statistics",
        "details": "Build the statistical foundation needed for data work and model interpretation.",
        "subtopics": ["Descriptive Statistics", "Probability Distributions", "Sampling Techniques", "Hypothesis Testing"],
    },
    {
        "week": 2,
        "topic": "Pandas",
        "details": "Work with tabular data using pandas for transformation and analysis.",
        "subtopics": ["DataFrames", "Data Cleaning", "Grouping and Aggregation", "Merge and Join Operations"],
    },
    {
        "week": 3,
        "topic": "Visualization",
        "details": "Communicate insights clearly with charts and exploratory visuals.",
        "subtopics": ["Matplotlib Basics", "Seaborn Plots", "Chart Selection", "Dashboard Basics"],
    },
    {
        "week": 4,
        "topic": "ML Basics",
        "details": "Learn the first machine learning workflow from data split to model evaluation.",
        "subtopics": ["Train/Test Split", "Regression Basics", "Classification Basics", "Model Evaluation Metrics"],
    },
]

DATA_ANALYSIS_LEARNING_PATH = [
    {
        "week": 1,
        "topic": "Data Analysis Fundamentals & Data Profiling Techniques",
        "details": "Build the foundation for analytical thinking, data types, and profiling before deeper analysis.",
        "subtopics": ["Descriptive Analytics", "Data Types", "Data Profiling", "Summary Statistics"],
    },
    {
        "week": 2,
        "topic": "Data Cleaning, Transformation & Exploratory Data Analysis",
        "details": "Prepare datasets, transform fields, and explore distributions, relationships, and anomalies.",
        "subtopics": ["Missing Value Handling", "Type Conversion", "Feature Transformation", "Exploratory Charts"],
    },
    {
        "week": 3,
        "topic": "Data Manipulation, SQL & Dashboard Development",
        "details": "Practice SQL querying, joins, aggregations, and communicate findings through dashboards.",
        "subtopics": ["SQL Joins", "Aggregations", "Window Functions", "Dashboard Design"],
    },
    {
        "week": 4,
        "topic": "Statistical Analysis, Business Insights & Project Deployment",
        "details": "Turn analysis into business recommendations and package the final project for sharing.",
        "subtopics": ["Hypothesis Testing", "A/B Testing", "Business KPIs", "Project Presentation"],
    },
]


def _build_java_learning_path() -> list[dict]:
    return [dict(item) for item in JAVA_LEARNING_PATH]


def _build_python_learning_path() -> list[dict]:
    return [
        {
            **item,
            "subtopics": list(item.get("subtopics", [])),
        }
        for item in PYTHON_LEARNING_PATH
    ]


def _build_data_analysis_learning_path() -> list[dict]:
    return [
        {
            **item,
            "subtopics": list(item.get("subtopics", [])),
        }
        for item in DATA_ANALYSIS_LEARNING_PATH
    ]


def _match_course_source(requested_source: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not requested_source or not hasattr(rag_service, "documents_db"):
        return None, None

    requested_lower = requested_source.strip().lower()
    for source, text in rag_service.documents_db.items():
        if source.lower() == requested_lower:
            return source, text
    return None, None


def _split_course_text(text: str, parts: int = 4) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) >= parts:
        return paragraphs[:parts]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= parts:
        chunk_size = max(1, len(sentences) // parts)
        return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)][:parts]

    words = text.split()
    if not words:
        return []

    chunk_size = max(1, len(words) // parts)
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)][:parts]


def _infer_course_topic_title(text: str, source_name: str, week_number: int) -> str:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9+#-]*", text)]
    meaningful = [w for w in words if len(w) > 3 and w not in COURSE_STOPWORDS]
    counts = Counter(meaningful)
    top_terms = [term for term, _ in counts.most_common(3)]
    if top_terms:
        return " & ".join(term.replace("_", " ").title() for term in top_terms)

    base_name = re.sub(r"\.[^.]+$", "", source_name).replace("_", " ").replace("-", " ").strip()
    if base_name:
        return f"{base_name.title()} Core Concepts"
    return f"Course Week {week_number}"


def _build_course_specific_path(source_name: str, course_text: str, weeks: int = 4) -> list[dict]:
    chunks = _split_course_text(course_text, weeks)
    generated = []
    for idx, chunk in enumerate(chunks[:weeks]):
        cleaned_chunk = re.sub(r"\s+", " ", chunk).strip()
        title = _infer_course_topic_title(cleaned_chunk, source_name, idx + 1)
        generated.append({
            "week": idx + 1,
            "topic": title,
            "details": cleaned_chunk[:320]
        })

    return generated

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "AI Data Science Mentor Backend"}

DEFAULT_WEEK_PLAN = [
    {
        "week": 1,
        "topic": "Integrating Python & Data Workflows",
        "subtopics": [
            {
                "name": "Baseline Knowledge Synthesis",
                "details": "Master Python's core built-in data structures (Lists, Dictionaries, Sets), list comprehensions, and lambda functions for inline utility.",
                "tags": ["python", "lists", "dictionaries", "sets", "list comprehensions", "lambda", "loops"]
            },
            {
                "name": "The Core Data Science Stack",
                "details": "Foundational ecosystem libraries: NumPy for vectorized calculations, Pandas for DataFrames, and Matplotlib/Seaborn for visualization.",
                "tags": ["numpy", "pandas", "matplotlib", "seaborn", "python", "visualization", "data science stack", "libraries"]
            },
            {
                "name": "Data Ingestion Workflows",
                "details": "Ingesting from flat files (CSV), semi-structured APIs (JSON), and enterprise relational databases (SQL) using python connections.",
                "tags": ["csv", "json", "api", "sql", "ingestion", "database", "sqlite", "relational", "databases", "flat files"]
            }
        ]
    },
    {
        "week": 2,
        "topic": "Advanced Exploratory Data Analysis (EDA)",
        "subtopics": [
            {
                "name": "Statistical Analysis Foundations",
                "details": "Central tendency (mean, median, mode) and dispersion (variance, standard deviation, skewness, tail dominance, Kurtosis peaking).",
                "tags": ["statistics", "mean", "median", "mode", "variance", "std", "distribution", "kurtosis", "hypothesis", "stats"]
            },
            {
                "name": "Dataset Profiling Techniques",
                "details": "Systematic handling of missing values with dropna/fillna and structural casting to optimize schema performance.",
                "tags": ["profiling", "missing", "dropna", "fillna", "type casting", "pandas", "cleansing"]
            },
            {
                "name": "Spotting and Handling Outliers",
                "details": "Isolating anomalies using Interquartile Range (IQR) and Z-score variance thresholds.",
                "tags": ["outliers", "iqr", "z-score", "anomaly", "eda"]
            }
        ]
    },
    {
        "week": 3,
        "topic": "Introduction to Predictive Modeling",
        "subtopics": [
            {
                "name": "Linear Regression (Continuous Estimation)",
                "details": "Ordinary Least Squares parameter estimation, R2 score explanation, and performance error metrics (MSE, MAE).",
                "tags": ["linear regression", "regression", "ols", "r2", "mse", "mae", "modeling", "machine learning"]
            },
            {
                "name": "Logistic Regression (Categorical Classification)",
                "details": "Solving binary classification flag determinations using Sigmoid activation and log-loss cost minimization.",
                "tags": ["logistic regression", "regression", "sigmoid", "classification", "binary", "modeling", "machine learning"]
            },
            {
                "name": "Performance Evaluation Metrics",
                "details": "Quantifying model success using Precision, Recall, F1-Score harmonic mean, Confusion Matrix, and ROC-AUC curves.",
                "tags": ["metrics", "precision", "recall", "f1", "confusion matrix", "roc", "auc", "evaluation"]
            }
        ]
    },
    {
        "week": 4,
        "topic": "Project Deployment & Next Steps",
        "subtopics": [
            {
                "name": "Packaging Data Pipelines",
                "details": "Refactoring notebook code into modular classes and structured packages, utilizing virtual environments.",
                "tags": ["pipeline", "packaging", "modular", "python", "codebase", "package"]
            },
            {
                "name": "Light Deployment Frameworks",
                "details": "Building presentation APIs and user interfaces using Streamlit, Flask, or FastAPI.",
                "tags": ["deployment", "streamlit", "flask", "fastapi", "api", "serving", "web"]
            },
            {
                "name": "Future Professional Roadmap",
                "details": "Establishing version control with Git/GitHub, cloud deployment, and continuous model optimization.",
                "tags": ["roadmap", "git", "github", "cloud", "advanced", "version control"]
            }
        ]
    }
]

def extract_weeks_from_text(text: str):
    import re
    pattern = r'(Week\s+(\d+|[IVXLCDM]+)\s*[:\-]?\s*([^\n]+))'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return []
    weeks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        week_num_str = match.group(2)
        try:
            week_num = int(week_num_str)
        except ValueError:
            week_num = i + 1
        week_title = match.group(3).strip()
        week_content = text[match.end():end].strip()
        
        subtopic_matches = re.findall(r'^\s*(\d+)\.\s*([^\n]+)', week_content, re.MULTILINE)
        subtopics = []
        for s_idx, (num, sub_title) in enumerate(subtopic_matches):
            sub_start_pat = re.escape(f"{num}. {sub_title}")
            sub_match = re.search(sub_start_pat, week_content)
            if sub_match:
                sub_start_pos = sub_match.end()
                next_num = str(int(num) + 1)
                sub_end_match = re.search(r'^\s*' + re.escape(next_num) + r'\.\s*', week_content[sub_start_pos:], re.MULTILINE)
                if sub_end_match:
                    sub_end_pos = sub_start_pos + sub_end_match.start()
                else:
                    sub_end_pos = len(week_content)
                sub_detail = week_content[sub_start_pos:sub_end_pos].strip()
                sub_detail = re.sub(r'\s+', ' ', sub_detail)
                if len(sub_detail) > 180:
                    sub_detail = sub_detail[:177] + "..."
                subtopics.append({
                    "name": sub_title.strip(),
                    "details": sub_detail,
                    "tags": [sub_title.strip().lower()]
                })
        if not subtopics:
            lines = [line.strip() for line in week_content.split('\n') if line.strip()]
            for l_idx, line in enumerate(lines[:3]):
                subtopics.append({
                    "name": f"Topic {l_idx + 1}",
                    "details": line,
                    "tags": [line.lower()]
                })
        weeks.append({
            "week": week_num,
            "topic": week_title,
            "subtopics": subtopics
        })
    return weeks

@app.post("/api/v1/learning-path")
@app.post("/api/learning-path")
def generate_learning_path(input_data: SkillInput):
    skills = [s.strip().lower() for s in input_data.skills]

    if any("data analysis" in skill or "data analytics" in skill or "data analyst" in skill for skill in skills):
        return {
            "mapped": True,
            "skills": input_data.skills,
            "course_source": "data_analysis_learning_path",
            "generated_path": _build_data_analysis_learning_path()
        }

    if any("python" in skill or "sql" in skill for skill in skills):
        return {
            "mapped": True,
            "skills": input_data.skills,
            "course_source": "python_sql_learning_path",
            "generated_path": _build_python_learning_path()
        }

    if any("java" in skill for skill in skills):
        return {
            "mapped": True,
            "skills": input_data.skills,
            "course_source": "java_learning_path",
            "generated_path": _build_java_learning_path()
        }

    selected_course_name, selected_course_text = _match_course_source(input_data.course_source)

    if input_data.course_source and not selected_course_name:
        raise HTTPException(status_code=404, detail="Selected course source was not found in the knowledge base.")

    if selected_course_name and selected_course_text:
        course_path = _build_course_specific_path(selected_course_name, selected_course_text)

        if skills:
            filtered_course_path = []
            for item in course_path:
                item_text = f"{item.get('topic', '')} {item.get('details', '')}".lower()
                if not any(skill in item_text for skill in skills):
                    filtered_course_path.append(item)
            if filtered_course_path:
                course_path = filtered_course_path

        return {
            "mapped": True,
            "skills": input_data.skills,
            "course_source": selected_course_name,
            "generated_path": course_path
        }
    
    # 1. Parse all documents
    parsed_weeks = []
    if hasattr(rag_service, "documents_db") and rag_service.documents_db:
        for source, text in rag_service.documents_db.items():
            doc_weeks = extract_weeks_from_text(text)
            if doc_weeks:
                parsed_weeks.extend(doc_weeks)
                
    # 2. Fall back to standard defaults if no documents contain weekly guides
    if not parsed_weeks:
        parsed_weeks = DEFAULT_WEEK_PLAN
        
    # 3. Flatten subtopics
    all_subtopics = []
    for week in parsed_weeks:
        for sub in week["subtopics"]:
            all_subtopics.append({
                "name": sub["name"],
                "details": sub["details"],
                "tags": sub.get("tags", [])
            })
            
    # 4. Filter out topics matching entered skills
    remaining_subtopics = []
    for sub in all_subtopics:
        match = False
        for skill in skills:
            if not skill:
                continue
            # Check skill name matches subtopic name or details or tags
            if (skill in sub["name"].lower() or 
                skill in sub["details"].lower() or 
                any(skill in tag.lower() for tag in sub["tags"])):
                match = True
                break
        if not match:
            remaining_subtopics.append(sub)
            
    # 5. Distribute remaining subtopics across 4 weeks
    num_weeks = 4
    generated_path = []
    
    if not remaining_subtopics:
        generated_path = [
            {"week": 1, "topic": "Advanced Topics & Research", "details": "Since you already have a solid foundation in the core topics, focus on state-of-the-art models and production scaling."},
            {"week": 2, "topic": "Cloud Data Engineering", "details": "Learn about distributed storage, streaming architectures (Kafka), and cloud databases (BigQuery/Snowflake)."},
            {"week": 3, "topic": "Large Language Models & GenAI", "details": "Dive into transformers, fine-tuning LLMs, vector search embeddings, and agentic workflows."},
            {"week": 4, "topic": "MLOps & System Design", "details": "Model monitoring, CI/CD for machine learning, pipeline orchestration, and low-latency serving."}
        ]
    else:
        import math
        chunks = [[] for _ in range(num_weeks)]
        for idx, sub in enumerate(remaining_subtopics):
            chunks[idx % num_weeks].append(sub)
            
        for w in range(num_weeks):
            week_subs = chunks[w]
            if week_subs:
                topic_title = " & ".join([sub["name"] for sub in week_subs])
                if len(topic_title) > 65:
                    topic_title = topic_title[:62] + "..."
                details_list = []
                for sub in week_subs:
                    details_list.append(f"{sub['name']}: {sub['details']}")
                topic_details = "; ".join(details_list)
            else:
                advanced_topics = [
                    {"name": "MLOps", "details": "Pipeline orchestration, CI/CD, and model registry."},
                    {"name": "Big Data Scaling", "details": "Distributed systems, Spark, and cloud data warehouses."},
                    {"name": "Advanced Optimization", "details": "Deep learning architectures, hyperparameter tuning, and advanced estimators."}
                ]
                topic_item = advanced_topics[w % len(advanced_topics)]
                topic_title = topic_item["name"]
                topic_details = f"{topic_item['name']}: {topic_item['details']}"
                
            generated_path.append({
                "week": w + 1,
                "topic": topic_title,
                "details": topic_details
            })
            
    return {"mapped": True, "skills": input_data.skills, "course_source": None, "generated_path": generated_path}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        
        # Simple text extraction based on file format
        text = ""
        if filename.endswith(".txt") or filename.endswith(".csv") or filename.endswith(".md"):
            text = content.decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            import io
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(content))
                text_list = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_list.append(page_text)
                text = "\n".join(text_list)
            except Exception as pdf_err:
                print(f"Error extracting PDF via pypdf: {pdf_err}. Falling back to basic decode.")
                # Fallback text decoder for basic binary content parsing
                text = content.decode("latin1", errors="ignore")
                # Clean up binary streams
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        else:
            text = content.decode("utf-8", errors="ignore")
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty or unsupported.")
            
        rag_service.add_document(text, filename)
        return {"filename": filename, "status": "successfully ingested into vector storage", "chunks": len(rag_service.vector_db)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@app.get("/api/rag/documents")
def list_rag_documents():
    """Returns the list of all document sources currently ingested in the knowledge base."""
    docs = rag_service.list_documents()
    return {"documents": docs, "total_chunks": len(rag_service.vector_db)}

class RAGQueryInput(BaseModel):
    query: str
    top_k: Optional[int] = 3

def format_mentor_response(query: str, raw_reply: str, is_grounded: bool, source_str: str = "") -> str:
    # Clean up any existing header prefixes from the raw reply
    cleaned_reply = raw_reply
    
    # Remove standard no-match warnings
    cleaned_reply = cleaned_reply.replace("I could not find this information in the uploaded knowledge base.", "")
    cleaned_reply = cleaned_reply.replace("No matching document chunks found.", "")
    
    # Remove existing headers like "Source:", "Answer:", "General Knowledge Reference:"
    cleaned_reply = re.sub(r"📚 Source:.*?\n", "", cleaned_reply, flags=re.IGNORECASE)
    cleaned_reply = re.sub(r"📝 Answer:", "", cleaned_reply, flags=re.IGNORECASE)
    cleaned_reply = re.sub(r"🌐 General Knowledge Reference:", "", cleaned_reply, flags=re.IGNORECASE)
    
    cleaned_reply = cleaned_reply.strip()
    
    # Check if this is a "not found" response
    no_match_text = "I could not find the answer to this question in the uploaded documents."
    if no_match_text in cleaned_reply or "could not find this information" in cleaned_reply or not is_grounded:
        status = "❌ Not Found in Knowledge Base"
        explanation_content = no_match_text
        example_content = ""
        takeaway_content = "Please verify your question or upload a document containing the relevant information."
    else:
        if source_str:
            status = f"✅ Knowledge Base Answer (Source: {source_str})"
        else:
            status = "✅ Knowledge Base Answer"
            
        # Extract code blocks
        code_blocks = re.findall(r"```(?:python|sql|bash|json)?\n(.*?)\n```", cleaned_reply, re.DOTALL | re.IGNORECASE)
        
        # If code blocks exist, use the first one as our Example
        if code_blocks:
            first_code = code_blocks[0]
            lang = "sql" if "select" in first_code.lower() else "python"
            example_content = f"```{lang}\n{first_code.strip()}\n```"
            # Remove the first code block from explanation
            match = re.search(r"```(?:python|sql|bash|json)?\n.*?\n```", cleaned_reply, re.DOTALL | re.IGNORECASE)
            if match:
                explanation_content = cleaned_reply.replace(match.group(0), "").strip()
            else:
                explanation_content = cleaned_reply
        else:
            # No code block in reply, check query for standard default
            query_lower = query.lower()
            if "sql" in query_lower:
                example_content = "```sql\nSELECT * FROM dataset LIMIT 5;\n```"
            elif "python" in query_lower or "pandas" in query_lower or "dropna" in query_lower or "loop" in query_lower:
                example_content = "```python\n# Basic python usage\nprint(\"Learning Python & Data Science!\")\n```"
            elif "regression" in query_lower or "classification" in query_lower or "overfitting" in query_lower or "pca" in query_lower or "cross-validation" in query_lower:
                example_content = "```python\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n```"
            else:
                example_content = "```python\n# Explore concepts in Python or SQL\n```"
            explanation_content = cleaned_reply

    # Clean double newlines in explanation
    explanation_content = re.sub(r"\n{3,}", "\n\n", explanation_content).strip()

    # Extract Key Takeaway (last paragraph or last sentence)
    if "explanation_content" in locals() and explanation_content:
        paragraphs = [p.strip() for p in explanation_content.split("\n\n") if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', last_para) if s.strip()]
            if sentences:
                takeaway_content = sentences[-1]
            else:
                takeaway_content = last_para
        else:
            takeaway_content = "Solidify these concepts through coding exercises and practical projects."
    elif not takeaway_content:
        takeaway_content = "Solidify these concepts through coding exercises and practical projects."

    # Build response in requested format
    formatted = (
        f"📖 Query: {query}\n"
        f"{status}\n\n"
        f"📚 Explanation:\n{explanation_content}\n\n"
    )
    if example_content:
        formatted += f"💻 Example:\n{example_content}\n\n"
    formatted += f"🎯 Key Takeaway:\n{takeaway_content}"
    return formatted

@app.post("/api/rag/query")
def rag_query(input_data: RAGQueryInput):
    """Performs a RAG-grounded query against the knowledge base and returns a synthesized answer with source citations."""
    query = input_data.query
    top_k = min(input_data.top_k or 3, 5)
    
    # 1. Retrieve relevant chunks from the vector DB
    context_chunks = rag_service.search_rag(query, top_k=top_k)
    serializable_context = [{"source": c["source"], "text": c["text"]} for c in context_chunks]
    
    # Get source string
    source_str = ""
    if context_chunks:
        sources = list(set([c['source'] for c in context_chunks]))
        source_str = ", ".join(sources)
        
    # 2. Build grounded context prompt
    context_text = ""
    if context_chunks:
        context_text = "Use the following retrieved document excerpts as the authoritative source to answer the student's question:\n\n"
        for i, chunk in enumerate(context_chunks):
            context_text += f"[{i+1}] Source: {chunk['source']}\n{chunk['text']}\n\n"
    else:
        context_text = "No relevant context found in the database.\n\n"
    
    system_instruction = (
        "You are an AI assistant specializing in document Q&A. Your sole task is to answer the user's question using ONLY the provided retrieved context from the uploaded documents.\n\n"
        "Rules:\n"
        "1. Answer the question directly and concisely, using only facts directly mentioned in the retrieved context.\n"
        "2. Do not include any general knowledge, assumptions, or external information. If the answer cannot be found in the context, do not attempt to answer it using general knowledge.\n"
        "3. If the retrieved context does not contain the answer to the user's question, you must respond with exactly: \"I could not find the answer to this question in the uploaded documents.\"\n"
        "4. Do not summarize the entire document or output overall document details that are not directly relevant to the specific question asked.\n\n"
        "Response Format:\n"
        f"📖 Query: {query}\n"
        "[Insert '✅ Knowledge Base Answer' OR '❌ Not Found in Knowledge Base']\n"
        "📚 Explanation: {detailed explanation}\n"
        "💻 Example: {example code/query}\n"
        "🎯 Key Takeaway: {short summary}"
    )
    
    prompt = (
        f"System: {system_instruction}\n\n"
        f"Retrieved Context:\n{context_text}"
        f"Student Question: {query}\n"
        f"Mentor Answer:"
    )
    
    # 3. Try Llama 3 / Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            reply = response.json().get("response", "").strip()
            is_grounded = len(context_chunks) > 0 and "could not find the answer" not in reply.lower()
            if not ("📖 Query:" in reply and "📚 Explanation:" in reply):
                reply = format_mentor_response(query, reply, is_grounded, source_str)
            return {"reply": reply, "grounded": is_grounded, "context": serializable_context}
    except Exception as e:
        print(f"Ollama RAG query failed ({e}). Using mock fallback.")
    
    # 4. Fallback mock response
    reply = mock_llama3_response(query, context_chunks)
    is_grounded = len(context_chunks) > 0 and "could not find the answer" not in reply.lower()
    reply = format_mentor_response(query, reply, is_grounded, source_str)
    return {"reply": reply, "grounded": is_grounded, "context": serializable_context}

@app.post("/api/chat")
def chat_mentor(input_data: ChatInput):
    message = input_data.message
    
    # 1. Search vector database for grounded context (RAG)
    context_chunks = rag_service.search_rag(message, top_k=2)
    serializable_context = [{"source": c["source"], "text": c["text"]} for c in context_chunks]
    
    # Get source string
    source_str = ""
    if context_chunks:
        sources = list(set([c['source'] for c in context_chunks]))
        source_str = ", ".join(sources)
        
    # 2. Build grounded context prompt
    context_text = ""
    if context_chunks:
        context_text = "Use the following retrieved document excerpts as the authoritative source to answer the student's question:\n\n"
        for chunk in context_chunks:
            context_text += f"Source [{chunk['source']}]:\n{chunk['text']}\n\n"
    else:
        context_text = "No relevant context found in the database.\n\n"
            
    system_instruction = (
        "You are an AI assistant specializing in document Q&A. Your sole task is to answer the user's question using ONLY the provided retrieved context from the uploaded documents.\n\n"
        "Rules:\n"
        "1. Answer the question directly and concisely, using only facts directly mentioned in the retrieved context.\n"
        "2. Do not include any general knowledge, assumptions, or external information. If the answer cannot be found in the context, do not attempt to answer it using general knowledge.\n"
        "3. If the retrieved context does not contain the answer to the user's question, you must respond with exactly: \"I could not find the answer to this question in the uploaded documents.\"\n"
        "4. Do not summarize the entire document or output overall document details that are not directly relevant to the specific question asked.\n\n"
        "Response Format:\n"
        f"📖 Query: {message}\n"
        "[Insert '✅ Knowledge Base Answer' OR '❌ Not Found in Knowledge Base']\n"
        "📚 Explanation: {detailed explanation}\n"
        "💻 Example: {example code/query}\n"
        "🎯 Key Takeaway: {short summary}"
    )
    
    prompt = (
        f"System: {system_instruction}\n\n"
        f"Retrieved Context:\n{context_text}"
        f"Student Question: {message}\n"
        f"Mentor Answer:"
    )
    
    # 3. Call Llama 3 (Ollama)
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        
        if response.status_code == 200:
            reply = response.json().get("response", "").strip()
            is_grounded = len(context_chunks) > 0 and "could not find the answer" not in reply.lower()
            if not ("📖 Query:" in reply and "📚 Explanation:" in reply):
                reply = format_mentor_response(message, reply, is_grounded, source_str)
            return {"reply": reply, "grounded": is_grounded, "context": serializable_context}
    except Exception as e:
        print(f"Ollama connection failed ({e}). Using grounded mock engine.")
        
    # 4. Fallback to mock logic if Ollama isn't configured
    reply = mock_llama3_response(message, context_chunks)
    is_grounded = len(context_chunks) > 0 and "could not find the answer" not in reply.lower()
    reply = format_mentor_response(message, reply, is_grounded, source_str)
    return {"reply": reply, "grounded": is_grounded, "context": serializable_context}


def mock_evaluation_response(code: str) -> dict:
    code_lower = code.strip().lower()
    
    # Check for df.dropna()
    if "dropna(" in code_lower or "dropna" in code_lower:
        return {
            "status": "Needs Improvement",
            "review_summary": "Good approach — `dropna()` removes missing values correctly. However, your code has a few issues regarding state persistence and memory efficiency.",
            "line_by_line_feedback": (
                "- **State Persistence**: `df.dropna()` returns a copy by default. The original DataFrame 'df' remains unmodified. Consider reassigning it as `df = df.dropna()` or using `inplace=True`.\n"
                "- **Memory Optimization**: Use `df.dropna(inplace=True)` to modify the DataFrame directly without creating an unnecessary copy in memory (highly critical for large datasets).\n"
                "- **Targeted Cleanup**: Use the `subset` parameter (e.g., `df.dropna(subset=['column_name'])`) if you only want to drop rows where a specific column is null, avoiding accidental deletion of useful rows."
            ),
            "optimized_alternative_code": (
                "```python\n"
                "# Recommended: Reassign with subset to keep the DataFrame valid\n"
                "df = df.dropna(subset=['target_column'])\n\n"
                "# Alternative: Modify in-place for memory optimization on large datasets\n"
                "df.dropna(inplace=True)\n"
                "```"
            )
        }
        
    # Check for loops over rows (anti-pattern)
    if "for index" in code_lower or "iterrows()" in code_lower or "itertuples()" in code_lower:
        return {
            "status": "Needs Improvement",
            "review_summary": "Iterating over DataFrame rows using loops is a slow and inefficient anti-pattern in pandas. Vectorized operations should be used instead.",
            "line_by_line_feedback": (
                "- **Vectorization**: Instead of looping row-by-row using `iterrows()` or a standard for loop, perform operations directly on pandas Series. This leverages underlying C-optimized implementations.\n"
                "- **Alternative**: Use built-in vectorized functions or `.apply()` / `.map()` for more complex element-wise operations."
            ),
            "optimized_alternative_code": (
                "```python\n"
                "# Slow Loop Anti-pattern:\n"
                "# for index, row in df.iterrows():\n"
                "#     df.loc[index, 'total'] = row['a'] + row['b']\n\n"
                "# Fast Vectorized Solution:\n"
                "df['total'] = df['a'] + df['b']\n"
                "```"
            )
        }
        
    # Default fallback structured evaluation
    return {
        "status": "Correct",
        "review_summary": "The submitted code logic looks sound and follows standard Python and Pandas data science practices.",
        "line_by_line_feedback": "- No major anti-patterns or inefficiencies detected in the code structure.",
        "optimized_alternative_code": (
            "```python\n"
            "# Code is already optimal\n"
            f"{code}\n"
            "```"
        )
    }

@app.post("/api/v1/evaluate")
def evaluate_assignment(submission: AssignmentSubmission):
    code = submission.submitted_code
    context = submission.assignment_context or "None"
    
    # 1. Build system prompt for Llama 3
    system_prompt = (
        "You are an expert AI Engineer and Python Developer acting as a Senior Data Science Code Reviewer. "
        "You are encouraging yet uncompromising on best practices. Evaluate the student's code submission "
        "for correctness, memory efficiency, and proper API usage. Check for common data science anti-patterns "
        "(e.g., iterating over rows using loops instead of vectorized Pandas operations).\n\n"
        "Crucial evaluation rules:\n"
        "- IF the student submits a generic row/column drop operation like: `df.dropna()`\n"
        "  THEN praise the approach but explicitly instruct them on memory efficiency and targeted cleanup:\n"
        "    * Highlight using `df.dropna(inplace=True)` to modify the DataFrame directly without creating an unnecessary copy in memory (highly critical for large datasets).\n"
        "    * Highlight using the `subset` parameter (e.g., `df.dropna(subset=['column_name'])`) if they only want to drop rows where a specific column is null.\n\n"
        "You MUST respond ONLY with a raw JSON object matching the following structure. Do not wrap the output in markdown code blocks or add any other text outside the JSON:\n"
        "{\n"
        '  "status": "Correct" | "Incorrect" | "Needs Improvement",\n'
        '  "review_summary": "Overall summary of the review...",\n'
        '  "line_by_line_feedback": "Markdown bulleted points with specific line feedback...",\n'
        '  "optimized_alternative_code": "Optimized production-ready Python code, wrapped in standard markdown blocks like ```python\\n...\\n```"\n'
        "}"
    )
    
    prompt = (
        f"Assignment Context: {context}\n"
        f"Student Submitted Code:\n{code}\n"
    )
    
    # 2. Call Llama 3 via Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        
        if response.status_code == 200:
            text_result = response.json().get("response", "").strip()
            
            # Clean markdown code block wraps (e.g. ```json ... ```)
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
                
            parsed_response = json.loads(text_result)
            
            # Basic validation of expected keys
            required_keys = ["status", "review_summary", "line_by_line_feedback", "optimized_alternative_code"]
            if all(k in parsed_response for k in required_keys):
                return parsed_response
            
    except Exception as e:
        print(f"Ollama assignment evaluation failed ({e}). Using rules-based fallback engine.")
        
    # 3. Fallback to mock evaluation rules if Ollama fails or returns invalid response
    return mock_evaluation_response(code)

def mock_quiz_response(topic: str) -> list:
    topic_lower = topic.lower()
    if "workflow" in topic_lower or "ingestion" in topic_lower or "integrating python" in topic_lower:
        return [
            {
                "id": 1,
                "question": "What is the main performance benefit of using NumPy arrays over standard Python lists for numerical operations?",
                "options": [
                    "NumPy arrays support vectorization which avoids loop overhead in Python",
                    "NumPy arrays can store different data types in the same array",
                    "NumPy arrays automatically run on GPUs",
                    "NumPy arrays are mutable and lists are not"
                ],
                "correct_answer": "NumPy arrays support vectorization which avoids loop overhead in Python",
                "explanation": "NumPy arrays are homogeneous and stored in contiguous memory blocks, enabling vectorized calculations implemented in compiled C code, bypassing Python loop interpreter overhead."
            },
            {
                "id": 2,
                "question": "In Python's built-in libraries, which module is typically used to work with JSON data?",
                "options": ["json", "requests", "pandas", "urllib"],
                "correct_answer": "json",
                "explanation": "The json module provides standard methods like loads and dumps to serialize and deserialize JSON objects."
            },
            {
                "id": 3,
                "question": "What is the output of the dictionary comprehension {x: x**2 for x in (1, 2, 3)}?",
                "options": ["{1: 1, 2: 4, 3: 9}", "{1: 2, 2: 4, 3: 6}", "[1, 4, 9]", "{1, 4, 9}"],
                "correct_answer": "{1: 1, 2: 4, 3: 9}",
                "explanation": "The comprehension maps each value in the tuple (1, 2, 3) to its square, creating dictionary keys and values."
            },
            {
                "id": 4,
                "question": "Which database library is a built-in part of the Python standard library, allowing database queries without external installations?",
                "options": ["sqlite3", "psycopg2", "SQLAlchemy", "pymongo"],
                "correct_answer": "sqlite3",
                "explanation": "sqlite3 is part of Python's standard library and provides a DB-API 2.0 compliant interface for working with SQLite database files."
            },
            {
                "id": 5,
                "question": "In Pandas, what is the default behavior when calling df.to_csv('file.csv') regarding the index?",
                "options": [
                    "It writes the DataFrame index as the first column in the file",
                    "It ignores the index entirely",
                    "It raises an exception unless index=False is specified",
                    "It hashes the index for security"
                ],
                "correct_answer": "It writes the DataFrame index as the first column in the file",
                "explanation": "By default, to_csv() saves the index. If you do not want the index in your CSV, you must pass index=False."
            }
        ]
    elif "exploratory" in topic_lower or "eda" in topic_lower or "outliers" in topic_lower or "profiling" in topic_lower or "cleansing" in topic_lower or "handling outliers" in topic_lower or "statistical analysis" in topic_lower:
        return [
            {
                "id": 1,
                "question": "What is the definition of the Interquartile Range (IQR)?",
                "options": [
                    "The difference between the 75th percentile (Q3) and the 25th percentile (Q1)",
                    "The difference between the maximum and minimum values",
                    "The range between mean and standard deviation",
                    "The middle 95% of the data"
                ],
                "correct_answer": "The difference between the 75th percentile (Q3) and the 25th percentile (Q1)",
                "explanation": "The IQR measures statistical dispersion and represents the middle 50% of the dataset."
            },
            {
                "id": 2,
                "question": "When analyzing skewness of a distribution, if the tail is longer on the right side, the distribution is:",
                "options": [
                    "Positively skewed (right-skewed)",
                    "Negatively skewed (left-skewed)",
                    "Symmetric",
                    "Platykurtic"
                ],
                "correct_answer": "Positively skewed (right-skewed)",
                "explanation": "A right-skewed distribution has a long tail stretching to the right, meaning the mean is typically greater than the median."
            },
            {
                "id": 3,
                "question": "Which of the following is an anomaly detection method based on standard deviations from the mean?",
                "options": ["Z-score", "Interquartile Range (IQR)", "Elbow Method", "Silhouette Score"],
                "correct_answer": "Z-score",
                "explanation": "A Z-score indicates how many standard deviations a data point is from the mean. Points with |Z| > 3 are often considered outliers."
            },
            {
                "id": 4,
                "question": "In Pandas, what is the best way to handle missing values by filling them with the median of that column?",
                "options": [
                    "df['col'].fillna(df['col'].median(), inplace=True)",
                    "df['col'].dropna()",
                    "df['col'].replace(0)",
                    "df['col'].interpolate()"
                ],
                "correct_answer": "df['col'].fillna(df['col'].median(), inplace=True)",
                "explanation": "fillna() replaces NaN values with the computed median of the column."
            },
            {
                "id": 5,
                "question": "What does the term 'Kurtosis' measure in statistical data distributions?",
                "options": [
                    "The 'tailedness' or thickness of the distribution tails relative to a normal distribution",
                    "The horizontal shift of the peak",
                    "The correlation between two variables",
                    "The spread between min and max values"
                ],
                "correct_answer": "The 'tailedness' or thickness of the distribution tails relative to a normal distribution",
                "explanation": "Kurtosis measures the presence of outliers and tail thickness. High kurtosis indicates heavy tails (more extreme outliers)."
            }
        ]
    elif "predictive" in topic_lower or "modeling" in topic_lower or "regression" in topic_lower or "classification" in topic_lower or "continuous estimation" in topic_lower or "categorical classification" in topic_lower:
        return [
            {
                "id": 1,
                "question": "In Ordinary Least Squares (OLS) linear regression, what metric is minimized during training?",
                "options": [
                    "Sum of Squared Residuals (SSR)",
                    "Mean Absolute Error (MAE)",
                    "R-squared value",
                    "Log Loss"
                ],
                "correct_answer": "Sum of Squared Residuals (SSR)",
                "explanation": "OLS minimizes the sum of squared differences between observed values and values predicted by the linear model."
            },
            {
                "id": 2,
                "question": "Which activation function is used in Logistic Regression to map any real-valued number into a probability range between 0 and 1?",
                "options": ["Sigmoid function", "ReLU function", "Tanh function", "Softmax function"],
                "correct_answer": "Sigmoid function",
                "explanation": "The Sigmoid function 1 / (1 + e^-z) scales output values to [0, 1], which is interpreted as the probability of the positive class."
            },
            {
                "id": 3,
                "question": "If a classification model predicts everything as the negative class in a highly imbalanced dataset where 99% of samples are negative, what will be the model's accuracy?",
                "options": ["99%", "1%", "50%", "0%"],
                "correct_answer": "99%",
                "explanation": "Since 99% of samples are negative and the model predicts negative for all, it will get 99% of them correct, showcasing why accuracy is misleading for imbalanced data."
            },
            {
                "id": 4,
                "question": "What does the F1-Score represent?",
                "options": [
                    "The harmonic mean of Precision and Recall",
                    "The arithmetic mean of Accuracy and Precision",
                    "The product of True Positive Rate and True Negative Rate",
                    "The area under the ROC curve"
                ],
                "correct_answer": "The harmonic mean of Precision and Recall",
                "explanation": "The F1-Score is 2 * (Precision * Recall) / (Precision + Recall), providing a balanced metric for both false positives and false negatives."
            },
            {
                "id": 5,
                "question": "What does an R-squared value of 0.85 in a linear regression model indicate?",
                "options": [
                    "85% of the variance in the target variable is explained by the model's features",
                    "The model predictions are correct 85% of the time",
                    "The correlation coefficient between features is 0.85",
                    "The model error is 15%"
                ],
                "correct_answer": "85% of the variance in the target variable is explained by the model's features",
                "explanation": "R2 (coefficient of determination) measures the proportion of target variance explained by the independent variables."
            }
        ]
    elif "deployment" in topic_lower or "deploy" in topic_lower or "packaging" in topic_lower or "pipelines" in topic_lower or "framework" in topic_lower or "roadmap" in topic_lower or "serving" in topic_lower:
        return [
            {
                "id": 1,
                "question": "Why is it important to define a requirements.txt or pyproject.toml file in a Python data science repository?",
                "options": [
                    "To pin dependency packages and ensure reproducible environments",
                    "To speed up code execution",
                    "To compile Python scripts to native binaries",
                    "To track Git code versions"
                ],
                "correct_answer": "To pin dependency packages and ensure reproducible environments",
                "explanation": "Specifying package versions prevents code from breaking when dependencies are updated on other machines."
            },
            {
                "id": 2,
                "question": "Which micro-web framework is commonly used to create lightweight, interactive web dashboards in Python with minimal HTML/CSS code?",
                "options": ["Streamlit", "Django", "FastAPI", "Flask"],
                "correct_answer": "Streamlit",
                "explanation": "Streamlit is designed specifically for data scientists to quickly convert Python scripts into shareable web apps."
            },
            {
                "id": 3,
                "question": "In a REST API designed with FastAPI, which decorator is used to handle a GET request at the /predict path?",
                "options": ["@app.get('/predict')", "@app.post('/predict')", "@app.route('/predict')", "@app.predict()"],
                "correct_answer": "@app.get('/predict')",
                "explanation": "@app.get('/predict') routes GET requests to the decorated function in FastAPI."
            },
            {
                "id": 4,
                "question": "What is the role of a virtual environment (like venv or conda) in Python development?",
                "options": [
                    "To isolate project dependencies and avoid global library conflicts",
                    "To run code in a virtual sandbox on remote clouds",
                    "To double the memory allocated to Python scripts",
                    "To execute code inside a Docker container"
                ],
                "correct_answer": "To isolate project dependencies and avoid global library conflicts",
                "explanation": "Virtual environments keep libraries required by different projects in separate locations, preventing version conflicts."
            },
            {
                "id": 5,
                "question": "In version control using Git, what does the command git commit -m \"initial commit\" do?",
                "options": [
                    "Saves a snapshot of staged changes to the local repository with a message",
                    "Pushes the code changes to GitHub",
                    "Downloads the latest commits from the remote repository",
                    "Creates a new branch named 'initial commit'"
                ],
                "correct_answer": "Saves a snapshot of staged changes to the local repository with a message",
                "explanation": "git commit records staged changes in the local history, and -m attaches the message."
            }
        ]
    elif "python" in topic_lower:
        return [
            {
                "id": 1,
                "question": "What is the difference between a list and a tuple in Python?",
                "options": [
                    "Lists are mutable while tuples are immutable",
                    "Lists are immutable while tuples are mutable",
                    "Lists cannot store heterogeneous data types",
                    "Tuples cannot be indexed"
                ],
                "correct_answer": "Lists are mutable while tuples are immutable",
                "explanation": "Lists are mutable, meaning they can be modified after creation. Tuples are immutable and their contents cannot be changed."
            },
            {
                "id": 2,
                "question": "Which of the following is used to handle exceptions in Python?",
                "options": ["try-except", "catch-throw", "try-catch", "do-catch"],
                "correct_answer": "try-except",
                "explanation": "Python uses try-except blocks to catch and handle exceptions gracefully."
            },
            {
                "id": 3,
                "question": "What is the output of print(type([]) == list)?",
                "options": ["True", "False", "None", "Error"],
                "correct_answer": "True",
                "explanation": "[] creates a list object, and type([]) is indeed <class 'list'>, so comparing it to list returns True."
            },
            {
                "id": 4,
                "question": "What is the correct way to clone or copy a list in Python?",
                "options": [
                    "new_list = list(old_list)",
                    "new_list = old_list.copy()",
                    "new_list = old_list[:]",
                    "All of the above"
                ],
                "correct_answer": "All of the above",
                "explanation": "All of these methods create a shallow copy of the list: calling the list constructor, using the copy() method, or slicing the entire list."
            },
            {
                "id": 5,
                "question": "What is the difference between __init__ and __new__ in Python classes?",
                "options": [
                    "__new__ is the constructor that creates the instance; __init__ is the initializer that configures it",
                    "__init__ creates the instance; __new__ configures it",
                    "There is no difference; they are aliases",
                    "__new__ is only used in sub-classes"
                ],
                "correct_answer": "__new__ is the constructor that creates the instance; __init__ is the initializer that configures it",
                "explanation": "__new__ is static and responsible for creating and returning a new instance of the class. __init__ is then called to initialize the newly created instance."
            }
        ]
    elif "pandas" in topic_lower:
        return [
            {
                "id": 1,
                "question": "Which pandas function is used to read data from a CSV file?",
                "options": ["read_csv", "load_csv", "import_csv", "open_csv"],
                "correct_answer": "read_csv",
                "explanation": "pd.read_csv() is the standard function in pandas to parse CSV files into DataFrames."
            },
            {
                "id": 2,
                "question": "How do you select a single column named 'Age' from a pandas DataFrame 'df'?",
                "options": ["df['Age']", "df.select('Age')", "df.column('Age')", "df.get('Age')"],
                "correct_answer": "df['Age']",
                "explanation": "df['Age'] (or df.Age) is the standard syntax for selecting a column Series from a DataFrame."
            },
            {
                "id": 3,
                "question": "What does df.fillna(0) do?",
                "options": [
                    "Replaces all NaN values in the DataFrame with 0",
                    "Drops all rows containing 0",
                    "Replaces all 0 values with NaN",
                    "Counts the number of zero values"
                ],
                "correct_answer": "Replaces all NaN values in the DataFrame with 0",
                "explanation": "fillna() is used to fill null or missing values (NaN) with a specified value, in this case, 0."
            },
            {
                "id": 4,
                "question": "Which method is used to combine two pandas DataFrames by joining rows or columns based on key columns?",
                "options": ["pd.merge()", "pd.concat()", "df.join()", "All of the above"],
                "correct_answer": "All of the above",
                "explanation": "merge() is database-style joins, join() joins on indexes, and concat() concatenates along a particular axis (rows or columns)."
            },
            {
                "id": 5,
                "question": "How do you count the number of missing (NaN) values in each column of a pandas DataFrame?",
                "options": ["df.isnull().sum()", "df.isna().count()", "df.missing().sum()", "df.nulls().count()"],
                "correct_answer": "df.isnull().sum()",
                "explanation": "df.isnull() (or df.isna()) returns a boolean DataFrame of identical shape, and calling .sum() aggregates the True values (which evaluate to 1) per column."
            }
        ]
    elif "sql" in topic_lower or "database" in topic_lower:
        return [
            {
                "id": 1,
                "question": "Which SQL clause is used to filter records after aggregation?",
                "options": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"],
                "correct_answer": "HAVING",
                "explanation": "The HAVING clause is used to filter records after group aggregates have been calculated. WHERE filters before aggregation."
            },
            {
                "id": 2,
                "question": "What type of JOIN returns all records when there is a match in either left or right table?",
                "options": ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN"],
                "correct_answer": "FULL OUTER JOIN",
                "explanation": "FULL OUTER JOIN (or FULL JOIN) returns all records when there is a match in left or right table records."
            },
            {
                "id": 3,
                "question": "Which index structure is most commonly used in relational database management systems (RDBMS) for indexing?",
                "options": ["Hash Table", "B-Tree / B+ Tree", "Binary Search Tree", "Linked List"],
                "correct_answer": "B-Tree / B+ Tree",
                "explanation": "B-Tree (and B+ Tree) structures are the industry standard for database indexing because they keep data sorted and allow search, sequential access, insertions, and deletions in logarithmic time."
            },
            {
                "id": 4,
                "question": "Which SQL constraint uniquely identifies each record in a database table?",
                "options": ["PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK"],
                "correct_answer": "PRIMARY KEY",
                "explanation": "A PRIMARY KEY constraint uniquely identifies each record in a table. It must contain unique values and cannot contain NULL values."
            },
            {
                "id": 5,
                "question": "What does ACID stand for in database transaction management?",
                "options": [
                    "Atomicity, Consistency, Isolation, Durability",
                    "Accuracy, Consistency, Integration, Duplication",
                    "Atomicity, Concurrency, Isolation, Distribution",
                    "Access, Control, Indexing, Delivery"
                ],
                "correct_answer": "Atomicity, Consistency, Isolation, Durability",
                "explanation": "ACID describes the key properties of reliable transactions: Atomicity (all or nothing), Consistency (preserves database rules), Isolation (independent concurrent execution), and Durability (persistence)."
            }
        ]
    elif "statistics" in topic_lower or "probability" in topic_lower:
        return [
            {
                "id": 1,
                "question": "What does the Central Limit Theorem state about the sampling distribution of the sample mean?",
                "options": [
                    "It approaches a normal distribution as sample size increases, regardless of the population distribution shape",
                    "It is always identical to the population distribution shape",
                    "It becomes skewed as sample size increases",
                    "It has a mean equal to the population standard deviation"
                ],
                "correct_answer": "It approaches a normal distribution as sample size increases, regardless of the population distribution shape",
                "explanation": "The CLT states that the sample mean distribution approaches a normal distribution as the sample size grows (typically N >= 30), regardless of the underlying population distribution shape."
            },
            {
                "id": 2,
                "question": "If event A and event B are independent, how is the joint probability P(A and B) calculated?",
                "options": ["P(A) + P(B)", "P(A) * P(B)", "P(A) / P(B)", "P(A) - P(B)"],
                "correct_answer": "P(A) * P(B)",
                "explanation": "For independent events, the probability of both events occurring simultaneously is the product of their individual probabilities."
            },
            {
                "id": 3,
                "question": "What is a p-value in hypothesis testing?",
                "options": [
                    "The probability of the alternative hypothesis being true",
                    "The probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true",
                    "The probability of making a Type II error",
                    "The probability of rejecting the null hypothesis when it is true"
                ],
                "correct_answer": "The probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true",
                "explanation": "A p-value measures the probability of finding the observed, or more extreme, results when the null hypothesis (H0) is true. A lower p-value indicates stronger evidence against H0."
            },
            {
                "id": 4,
                "question": "What is Type I error in hypothesis testing?",
                "options": [
                    "Rejecting the null hypothesis when it is actually true (False Positive)",
                    "Failing to reject the null hypothesis when it is actually false (False Negative)",
                    "Accepting the null hypothesis when it is true",
                    "An arithmetic error in calculations"
                ],
                "correct_answer": "Rejecting the null hypothesis when it is actually true (False Positive)",
                "explanation": "Type I error (significance level alpha) occurs when we reject a true null hypothesis. Type II error (beta) occurs when we fail to reject a false null hypothesis."
            },
            {
                "id": 5,
                "question": "What is the key difference between covariance and correlation?",
                "options": [
                    "Covariance is scale-dependent; correlation is normalized between -1 and 1",
                    "Covariance is normalized; correlation is scale-dependent",
                    "Covariance measures non-linear relationships; correlation measures linear ones",
                    "They are identical metrics"
                ],
                "correct_answer": "Covariance is scale-dependent; correlation is normalized between -1 and 1",
                "explanation": "Covariance shows the direction of the linear relationship between two variables, but its magnitude depends on unit scales. Correlation standardizes covariance by the product of standard deviations, giving a scale-free value between -1 and 1."
            }
        ]
    elif "deep learning" in topic_lower or "neural" in topic_lower:
        return [
            {
                "id": 1,
                "question": "Which activation function outputs values in the range of 0 to 1, making it ideal for binary classification output layers?",
                "options": ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"],
                "correct_answer": "Sigmoid",
                "explanation": "The Sigmoid activation function maps any real-valued number into a range between 0 and 1, representing a probability for binary classification."
            },
            {
                "id": 2,
                "question": "What is the primary role of backpropagation in training artificial neural networks?",
                "options": [
                    "To generate random weights and biases for the network",
                    "To calculate the gradients of the loss function with respect to weights and biases to update them",
                    "To normalize input features before they pass through the model",
                    "To visual training metrics like ROC-AUC"
                ],
                "correct_answer": "To calculate the gradients of the loss function with respect to weights and biases to update them",
                "explanation": "Backpropagation computes the gradient of the loss function for a single weight-bias set using the chain rule, which optimizer algorithms (like SGD, Adam) use to update model weights."
            },
            {
                "id": 3,
                "question": "Which regularization technique prevents overfitting in deep networks by randomly setting a fraction of input units to 0 at each step?",
                "options": ["L1 Regularization", "L2 Regularization", "Dropout", "Early Stopping"],
                "correct_answer": "Dropout",
                "explanation": "Dropout randomly deactivates (sets to zero) a proportion of neurons during training, forcing the network to learn redundant representations and reducing overfitting."
            },
            {
                "id": 4,
                "question": "Which of the following optimizer algorithms adjusts individual learning rates for each parameter using historical gradient averages?",
                "options": ["Adam", "Stochastic Gradient Descent (SGD)", "Mini-batch SGD", "Momentum"],
                "correct_answer": "Adam",
                "explanation": "Adam (Adaptive Moment Estimation) combines the principles of RMSprop and Momentum by calculating adaptive learning rates for each parameter based on running averages of first and second moments of the gradients."
            },
            {
                "id": 5,
                "question": "What is the primary cause of the 'vanishing gradient problem' in deep networks using Sigmoid activation functions?",
                "options": [
                    "Sigmoid derivatives are maximum 0.25, so gradients shrink exponentially when backpropagated through many layers",
                    "The learning rate is too large",
                    "Weights are initialized to zero",
                    "Gradients grow exponentially and overflow"
                ],
                "correct_answer": "Sigmoid derivatives are maximum 0.25, so gradients shrink exponentially when backpropagated through many layers",
                "explanation": "Since the derivative of the Sigmoid function is at most 0.25, multiplying these small values repeatedly during backpropagation through deep layers causes the gradient to shrink rapidly to zero, preventing early layers from updating."
            }
        ]
    elif "nlp" in topic_lower or "natural language" in topic_lower:
        return [
            {
                "id": 1,
                "question": "What is the process of breaking down a text sequence into individual words, subwords, or characters called?",
                "options": ["Stemming", "Tokenization", "Lemmatization", "Vectorization"],
                "correct_answer": "Tokenization",
                "explanation": "Tokenization is the task of chopping a character sequence into pieces (tokens) while throwing away certain characters like punctuation."
            },
            {
                "id": 2,
                "question": "What does TF-IDF stand for in text feature representation?",
                "options": [
                    "Term Frequency-Inverse Document Frequency",
                    "Text Filtering-Inverse Document Filtration",
                    "Topic Finder-Indexing Document Format",
                    "Token Fitting-Iterative Document Feature"
                ],
                "correct_answer": "Term Frequency-Inverse Document Frequency",
                "explanation": "TF-IDF stands for Term Frequency-Inverse Document Frequency, which reflects how important a word is to a document in a collection or corpus."
            },
            {
                "id": 3,
                "question": "Which model architecture introduced in 2017 utilizes self-attention mechanisms and forms the foundation of modern Large Language Models (LLMs)?",
                "options": ["RNN", "LSTM", "Transformer", "CNN"],
                "correct_answer": "Transformer",
                "explanation": "The Transformer architecture, introduced in 'Attention Is All You Need', replaced recurrence and convolutions with self-attention mechanisms, powering modern LLMs like GPT and Llama."
            },
            {
                "id": 4,
                "question": "What is the primary difference between Word2Vec and BERT embeddings?",
                "options": [
                    "Word2Vec generates static embeddings regardless of context; BERT generates dynamic context-dependent embeddings",
                    "BERT generates static embeddings; Word2Vec is dynamic",
                    "Word2Vec is only for character level; BERT is for paragraph level",
                    "Word2Vec cannot handle English words"
                ],
                "correct_answer": "Word2Vec generates static embeddings regardless of context; BERT generates dynamic context-dependent embeddings",
                "explanation": "Word2Vec maps a word to a single vector regardless of context (e.g. 'bank' in 'river bank' vs 'bank account'). BERT uses self-attention to generate different vectors for 'bank' based on surrounding words."
            },
            {
                "id": 5,
                "question": "What is the purpose of the BLEU score in NLP evaluation?",
                "options": [
                    "To evaluate the quality of machine-translated text compared to human references",
                    "To count the number of tokens in a document",
                    "To measure the training speed of an NLP model",
                    "To calculate vocabulary diversity"
                ],
                "correct_answer": "To evaluate the quality of machine-translated text compared to human references",
                "explanation": "Bilingual Evaluation Understudy (BLEU) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another, comparing n-gram overlaps with reference translations."
            }
        ]
    elif "visualization" in topic_lower or "plotting" in topic_lower:
        return [
            {
                "id": 1,
                "question": "Which Python library acts as the low-level foundation for data visualization, and is the base on top of which Seaborn is built?",
                "options": ["Plotly", "Bokeh", "Matplotlib", "ggplot"],
                "correct_answer": "Matplotlib",
                "explanation": "Matplotlib is the core low-level plotting library in the Python data science ecosystem. Seaborn is a high-level wrapper built on top of Matplotlib."
            },
            {
                "id": 2,
                "question": "Which type of plot is best suited for visualizing the distribution of a single continuous numerical variable?",
                "options": ["Scatter Plot", "Bar Plot", "Histogram", "Line Plot"],
                "correct_answer": "Histogram",
                "explanation": "A histogram groups continuous numerical data into bins and plots the count/density of values in each bin, displaying the distribution's shape."
            },
            {
                "id": 3,
                "question": "In a box-and-whisker plot, what range of the dataset does the central box shape represent?",
                "options": [
                    "The range between the minimum and maximum values",
                    "The Interquartile Range (IQR), from the 25th percentile (Q1) to the 75th percentile (Q3)",
                    "Only the values within one standard deviation of the mean",
                    "The top 10% outlier values"
                ],
                "correct_answer": "The Interquartile Range (IQR), from the 25th percentile (Q1) to the 75th percentile (Q3)",
                "explanation": "The box in a box plot represents the middle 50% of the dataset, spanning from the 25th percentile (lower quartile) to the 75th percentile (upper quartile)."
            },
            {
                "id": 4,
                "question": "In Seaborn, which function is used to plot pairwise relationships in a dataset, generating a matrix of bivariate scatter plots?",
                "options": ["sns.pairplot()", "sns.heatmap()", "sns.jointplot()", "sns.catplot()"],
                "correct_answer": "sns.pairplot()",
                "explanation": "sns.pairplot() creates a grid of Axes such that each numeric variable in data is shared in the y-axis across a single row and in the x-axis across a single column."
            },
            {
                "id": 5,
                "question": "Which plotting technique is ideal for visualizing correlation coefficients between many continuous numerical variables?",
                "options": [
                    "Heatmap of correlation matrix",
                    "Histogram matrix",
                    "Multiple bar charts",
                    "Line plot matrix"
                ],
                "correct_answer": "Heatmap of correlation matrix",
                "explanation": "A color-coded heatmap (e.g., using sns.heatmap(df.corr())) is highly effective for visual scanning of strong positive or negative correlations across large numbers of variables."
            }
        ]
    elif "pyspark" in topic_lower or "big data" in topic_lower or "spark" in topic_lower:
        return [
            {
                "id": 1,
                "question": "What is the core distributed data structure in Apache Spark, representing an immutable, partitioned collection of records?",
                "options": ["Pandas DataFrame", "RDD (Resilient Distributed Dataset)", "NumPy Array", "Python Dictionary"],
                "correct_answer": "RDD (Resilient Distributed Dataset)",
                "explanation": "RDD is the fundamental abstraction of Apache Spark, allowing parallel operations on distributed clusters with fault tolerance."
            },
            {
                "id": 2,
                "question": "What does 'lazy evaluation' mean in Apache Spark / PySpark?",
                "options": [
                    "Transformations are run instantly, but actions are delayed",
                    "Execution is slow due to excessive logging",
                    "Transformations are not executed immediately; they are built into an execution plan and run only when an action is called",
                    "Spark waits for user keyboard input before processing"
                ],
                "correct_answer": "Transformations are not executed immediately; they are built into an execution plan and run only when an action is called",
                "explanation": "Lazy evaluation means Spark delays actual execution of transformations (like filter, map) until an action (like collect, count) is requested. This allows Spark to optimize the full execution plan."
            },
            {
                "id": 3,
                "question": "Which method is used to keep a PySpark DataFrame persisted in memory across multiple actions?",
                "options": ["df.save()", "df.persist() or df.cache()", "df.store()", "df.freeze()"],
                "correct_answer": "df.persist() or df.cache()",
                "explanation": "df.cache() (or df.persist() with specific storage levels) tells Spark to save the intermediate DataFrame in memory/disk so subsequent actions do not need to recompute it."
            },
            {
                "id": 4,
                "question": "What is the difference between PySpark DataFrame 'transformations' and 'actions'?",
                "options": [
                    "Transformations are lazy and define a new DataFrame; actions trigger computation and return results to the driver or write to disk",
                    "Actions are lazy; transformations are eager",
                    "Transformations require shuffling; actions do not",
                    "There is no difference; they are synonyms"
                ],
                "correct_answer": "Transformations are lazy and define a new DataFrame; actions trigger computation and return results to the driver or write to disk",
                "explanation": "Transformations (like select, filter) return a new DataFrame and are not computed immediately. Actions (like show, collect, count) trigger the Spark job execution to return a value to the driver program or write data out."
            },
            {
                "id": 5,
                "question": "What PySpark operation causes a 'shuffle' where data is redistributed across partitions/executors?",
                "options": ["groupBy() or join()", "select()", "filter()", "map()"],
                "correct_answer": "groupBy() or join()",
                "explanation": "Wide transformations like groupBy(), join(), and repartition() require Spark to relocate data across different executors so keys are grouped together, which involves heavy network and disk I/O (shuffling)."
            }
        ]
    elif "machine learning" in topic_lower or "ml" in topic_lower:
        return [
            {
                "id": 1,
                "question": "Which validation technique splits a dataset into K non-overlapping parts?",
                "options": ["K-Fold Cross Validation", "Holdout Validation", "Bootstrapping", "Stratified Splits"],
                "correct_answer": "K-Fold Cross Validation",
                "explanation": "K-Fold Cross Validation splits the dataset into K parts, validating on one fold and training on the remaining K-1 folds iteratively."
            },
            {
                "id": 2,
                "question": "Which metric evaluates classifier performance by plotting True Positive Rate vs. False Positive Rate?",
                "options": ["ROC-AUC Curve", "Precision-Recall Curve", "F1-Score", "Confusion Matrix"],
                "correct_answer": "ROC-AUC Curve",
                "explanation": "The Receiver Operating Characteristic (ROC) curve plots TPR vs FPR, and the Area Under the Curve (AUC) measures aggregate performance."
            },
            {
                "id": 3,
                "question": "What is the standard method to handle highly multicollinear numerical features in linear regression?",
                "options": [
                    "Drop one of the collinear features or use L2 regularization (Ridge)",
                    "Add more identical features",
                    "Increase learning rate",
                    "Convert all features to strings"
                ],
                "correct_answer": "Drop one of the collinear features or use L2 regularization (Ridge)",
                "explanation": "Multicollinearity can be mitigated by removing one of the highly correlated columns, using PCA, or applying Ridge (L2) regularization."
            },
            {
                "id": 4,
                "question": "What is the primary difference between L1 (Lasso) and L2 (Ridge) regularization?",
                "options": [
                    "L1 regularizes weights by absolute value and can yield sparse weights; L2 regularizes by squared value and yields small but non-zero weights",
                    "L2 regularizes weights by absolute value; L1 regularizes by squared value",
                    "L1 is for classification; L2 is for regression",
                    "L2 yields sparse weights; L1 does not"
                ],
                "correct_answer": "L1 regularizes weights by absolute value and can yield sparse weights; L2 regularizes by squared value and yields small but non-zero weights",
                "explanation": "L1 Lasso adds a penalty proportional to the absolute values of coefficients, driving some to exactly zero (feature selection). L2 Ridge adds a penalty proportional to the square of coefficients, shrinking them close to zero but not exactly."
            },
            {
                "id": 5,
                "question": "What does the bias-variance tradeoff describe in machine learning?",
                "options": [
                    "The balance between underfitting (high bias) and overfitting (high variance)",
                    "The speed of training vs model accuracy",
                    "The ratio of positive to negative samples in a dataset",
                    "The computation cost of different algorithms"
                ],
                "correct_answer": "The balance between underfitting (high bias) and overfitting (high variance)",
                "explanation": "High bias models are too simple and underfit (low training and test accuracy). High variance models are too complex and overfit (high training accuracy, poor test accuracy). Finding the sweet spot minimizes total error."
            }
        ]
    # General fallback quiz
    return [
        {
            "id": 1,
            "question": "Which validation technique splits a dataset into K non-overlapping parts?",
            "options": ["K-Fold Cross Validation", "Holdout Validation", "Bootstrapping", "Stratified Splits"],
            "correct_answer": "K-Fold Cross Validation",
            "explanation": "K-Fold Cross Validation splits the dataset into K parts, validating on one fold and training on the remaining K-1 folds iteratively."
        },
        {
            "id": 2,
            "question": "Which metric evaluates classifier performance by plotting True Positive Rate vs. False Positive Rate?",
            "options": ["ROC-AUC Curve", "Precision-Recall Curve", "F1-Score", "Confusion Matrix"],
            "correct_answer": "ROC-AUC Curve",
            "explanation": "The Receiver Operating Characteristic (ROC) curve plots TPR vs FPR, and the Area Under the Curve (AUC) measures aggregate performance."
        },
        {
            "id": 3,
            "question": "What is the standard method to handle highly multicollinear numerical features in linear regression?",
            "options": [
                "Drop one of the collinear features or use L2 regularization (Ridge)",
                "Add more identical features",
                "Increase learning rate",
                "Convert all features to strings"
            ],
            "correct_answer": "Drop one of the collinear features or use L2 regularization (Ridge)",
            "explanation": "Multicollinearity can be mitigated by removing one of the highly correlated columns, using PCA, or applying Ridge (L2) regularization."
        },
        {
            "id": 4,
            "question": "What is the primary difference between L1 (Lasso) and L2 (Ridge) regularization?",
            "options": [
                "L1 regularizes weights by absolute value and can yield sparse weights; L2 regularizes by squared value and yields small but non-zero weights",
                "L2 regularizes weights by absolute value; L1 regularizes by squared value",
                "L1 is for classification; L2 is for regression",
                "L2 yields sparse weights; L1 does not"
            ],
            "correct_answer": "L1 regularizes weights by absolute value and can yield sparse weights; L2 regularizes by squared value and yields small but non-zero weights",
            "explanation": "L1 Lasso adds a penalty proportional to the absolute values of coefficients, driving some to exactly zero (feature selection). L2 Ridge adds a penalty proportional to the square of coefficients, shrinking them close to zero but not exactly."
        },
        {
            "id": 5,
            "question": "What does the bias-variance tradeoff describe in machine learning?",
            "options": [
                "The balance between underfitting (high bias) and overfitting (high variance)",
                "The speed of training vs model accuracy",
                "The ratio of positive to negative samples in a dataset",
                "The computation cost of different algorithms"
            ],
            "correct_answer": "The balance between underfitting (high bias) and overfitting (high variance)",
            "explanation": "High bias models are too simple and underfit (low training and test accuracy). High variance models are too complex and overfit (high training accuracy, poor test accuracy). Finding the sweet spot minimizes total error."
        }
    ]

def mock_projects_response(difficulty: str, topic: str) -> list:
    diff_lower = difficulty.lower()
    topic_lower = topic.lower()
    
    # Check difficulty levels
    is_beg = "beginner" in diff_lower
    is_adv = "advanced" in diff_lower
    is_int = not is_beg and not is_adv
    
    # Topic matching
    if "exploratory" in topic_lower or "eda" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "Exploratory Data Analysis (EDA) on House Sales",
                    "description": "Clean, visualize, and analyze a dataset of house prices to identify key drivers of home values.",
                    "difficulty": "Beginner",
                    "tech_stack": ["Python", "Pandas", "Matplotlib", "Seaborn"],
                    "key_deliverables": [
                        "Data cleaning report handling missing prices and values",
                        "Correlation heatmap identifying variables related to house prices",
                        "Bar plots showing average price per zip code"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "High-Dimensional Gene Expression EDA",
                    "description": "Perform dimensionality reduction (t-SNE/UMAP) and clustered heatmaps on massive gene expression datasets.",
                    "difficulty": "Advanced",
                    "tech_stack": ["Python", "UMAP-learn", "Scikit-Learn", "Seaborn", "Scanpy"],
                    "key_deliverables": [
                        "UMAP and t-SNE clustering projections comparing cellular groups",
                        "Interactive clustered heatmap of top 50 highly variable genes",
                        "Outlier diagnostics pipeline handling batch effects"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Store Sales EDA & Outlier Detection",
                    "description": "Analyze multi-store retail sales datasets to perform cohort analysis and find anomalies.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["Python", "Pandas", "Plotly", "Seaborn"],
                    "key_deliverables": [
                        "Interactive cohort retention curve using Plotly",
                        "Isolation Forest or boxplot anomalies report of daily transactions",
                        "Statistical significance testing (t-test) on seasonal sales spikes"
                    ]
                }
            ]
            
    elif "unsupervised" not in topic_lower and "supervised" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "Titanic Survival Classifier",
                    "description": "Build a baseline classification model to predict passenger survival based on age, gender, and ticket class.",
                    "difficulty": "Beginner",
                    "tech_stack": ["Scikit-Learn", "Pandas", "Logistic Regression"],
                    "key_deliverables": [
                        "Categorical encoding for passenger gender and port of embarkation",
                        "Trained Logistic Regression model with >75% validation accuracy",
                        "Evaluation confusion matrix"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Credit Card Fraud Detection Pipeline",
                    "description": "Train a high-performance ensemble model to identify fraudulent transactions in a highly imbalanced dataset.",
                    "difficulty": "Advanced",
                    "tech_stack": ["XGBoost", "LightGBM", "Scikit-Learn", "SMOTE", "MLflow"],
                    "key_deliverables": [
                        "SMOTE or class-weight balancing handling fraud label skewness",
                        "Hyperparameter optimization tracking precision-recall AUC metrics",
                        "Custom cost-sensitive classifier threshold tuning report"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Employee Attrition Predictor",
                    "description": "Build an HR prediction classifier to determine if employees are likely to leave, analyzing feature importances.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["Scikit-Learn", "Pandas", "Random Forest", "SHAP"],
                    "key_deliverables": [
                        "Random Forest model with cross-validated performance metrics",
                        "SHAP explanation values identifying top drivers of attrition",
                        "Precision-Recall trade-off graph for HR intervention targeting"
                    ]
                }
            ]

    elif "unsupervised" in topic_lower or "clustering" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "Customer Segmentation using K-Means",
                    "description": "Perform unsupervised learning to cluster retail buyers based on purchasing frequency, recency, and monetary value.",
                    "difficulty": "Beginner",
                    "tech_stack": ["Scikit-Learn", "Pandas", "PCA", "Yellowbrick"],
                    "key_deliverables": [
                        "RFM feature calculation script from raw transactional rows",
                        "Elbow plot and Silhouette analysis determining optimal cluster count",
                        "PCA projection 3D scatter plot of customer clusters"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Document Topic Modeling (LDA)",
                    "description": "Extract semantic topics from millions of customer support tickets using Latent Dirichlet Allocation.",
                    "difficulty": "Advanced",
                    "tech_stack": ["Gensim", "spaCy", "pyLDAvis", "NLTK"],
                    "key_deliverables": [
                        "Coherence score optimization curve to choose optimal topic count",
                        "pyLDAvis interactive HTML visualization of topic overlap",
                        "Automated ticket routing categorization engine using topic weights"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Anomaly Detection in Server Logs",
                    "description": "Detect malicious requests and potential cyber attacks from network access logs using clustering.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["Scikit-Learn", "DBSCAN", "Isolation Forest", "Pandas"],
                    "key_deliverables": [
                        "DBSCAN clustering identifying dense traffic profiles and core outliers",
                        "Validation metrics for detected anomalies vs. true label samples",
                        "Log parser utility extracting client IP features, status codes, and bytes"
                    ]
                }
            ]

    elif "deep learning" in topic_lower or "neural" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "MNIST Digit Classifier",
                    "description": "Create a basic multilayer perceptron (MLP) to recognize hand-drawn digits.",
                    "difficulty": "Beginner",
                    "tech_stack": ["PyTorch", "Torchvision", "Matplotlib"],
                    "key_deliverables": [
                        "PyTorch training loop tracking cross-entropy loss and accuracy",
                        "Model checkpoint file storing weights",
                        "Inference pipeline visualizing test predictions with confidence levels"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Medical X-Ray Segmentation (U-Net)",
                    "description": "Fine-tune a deep encoder-decoder network to identify and segment lung anomalies in chest radiographs.",
                    "difficulty": "Advanced",
                    "tech_stack": ["TensorFlow", "Keras", "Segmentation Models", "OpenCV"],
                    "key_deliverables": [
                        "Custom Dice coefficient loss and Intersection-over-Union (IoU) metrics",
                        "Data augmentation pipeline handling limited medical datasets",
                        "Before/After mask segmentation overlays on test images"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Cats vs. Dogs CNN Classifier",
                    "description": "Build a convolutional neural network (CNN) from scratch and use transfer learning to classify images.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["PyTorch", "TIMM", "Albumentations", "Matplotlib"],
                    "key_deliverables": [
                        "ResNet transfer learning model achieving >95% validation accuracy",
                        "Training vs validation loss curve plots",
                        "Albumentations data augmentation script boosting model robustness"
                    ]
                }
            ]

    elif "nlp" in topic_lower or "natural language" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "IMDb Movie Sentiment Classifier",
                    "description": "Build a baseline NLP model to classify movie reviews as positive or negative.",
                    "difficulty": "Beginner",
                    "tech_stack": ["NLTK", "Scikit-Learn", "Pandas", "Naive Bayes"],
                    "key_deliverables": [
                        "Text cleaning pipeline removing HTML tags, stopwords, and punctuation",
                        "TF-IDF or CountVectorizer representation of the textual features",
                        "Classification report showing precision, recall, and F1-score"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Semantic Q&A Retrieval Engine",
                    "description": "Develop a semantic search engine indexing knowledge guides and answering queries using vector embeddings.",
                    "difficulty": "Advanced",
                    "tech_stack": ["Sentence-Transformers", "FAISS", "FastAPI", "Uvicorn"],
                    "key_deliverables": [
                        "Sentence embedding generator pipeline indexing target documents",
                        "FAISS vector store index performing similarity queries under 10ms",
                        "FastAPI endpoint returning the top K context references"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Spam SMS Classifier API Service",
                    "description": "Train an LSTM or SpaCy classifier to detect spam texts and serve it through a REST endpoint.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["SpaCy", "FastAPI", "Pandas", "Scikit-Learn"],
                    "key_deliverables": [
                        "Custom SpaCy text categorizer model pipeline",
                        "FastAPI prediction endpoint with input validation (Pydantic)",
                        "Deployment-ready requirements.txt and run scripts"
                    ]
                }
            ]

    elif "vision" in topic_lower or "cv" in topic_lower or "image" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "OpenCV Receipt Scanner",
                    "description": "Use classic computer vision edge detection to extract and align receipt images.",
                    "difficulty": "Beginner",
                    "tech_stack": ["OpenCV", "NumPy", "Matplotlib"],
                    "key_deliverables": [
                        "Canny edge detection and contour mapping highlighting receipt borders",
                        "Four-point perspective warping creating flat document outputs",
                        "Image thresholding converting receipts to sharp black and white text"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Real-Time Lane Line Segmentation",
                    "description": "Develop a video segmentation pipeline to track lane markers for self-driving cars.",
                    "difficulty": "Advanced",
                    "tech_stack": ["PyTorch", "OpenCV", "U-Net", "CUDA"],
                    "key_deliverables": [
                        "U-Net lane segmentation model trained on driving video frames",
                        "Real-time inference script processing video frames at >30 FPS",
                        "Lane tracking overlay project showing steering angle predictions"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Object Detection with YOLO",
                    "description": "Train a YOLO model on custom classes to identify specific items in real-time camera feeds.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["Ultralytics", "YOLOv8", "OpenCV", "Roboflow"],
                    "key_deliverables": [
                        "Fine-tuned YOLOv8 weights detecting user-defined custom categories",
                        "Webcam ingestion script rendering bounding boxes in real-time",
                        "Mean Average Precision (mAP) evaluation score report"
                    ]
                }
            ]

    elif "time series" in topic_lower or "forecasting" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "Weather Trend Forecasting",
                    "description": "Analyze historic weather data and perform simple moving average and autoregressive forecasting.",
                    "difficulty": "Beginner",
                    "tech_stack": ["Statsmodels", "Pandas", "Matplotlib", "Seaborn"],
                    "key_deliverables": [
                        "Seasonal decomposition separating trend, seasonality, and residuals",
                        "Moving average baseline forecast comparison chart",
                        "Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Cryptocurrency High-Frequency Price Predictor",
                    "description": "Predict short-term price movements of cryptocurrencies using multi-variable LSTM / GRU networks.",
                    "difficulty": "Advanced",
                    "tech_stack": ["PyTorch", "Pandas", "Binance API", "TensorBoard"],
                    "key_deliverables": [
                        "Order book depth feature engineering script generating high-frequency indicators",
                        "LSTM recurrent network predicting 1-minute forward price directions",
                        "Backtesting simulator comparing model returns against a buy-and-hold strategy"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Store Demand Forecasting (Prophet)",
                    "description": "Utilize Prophet to predict future sales demands across multiple stores and departments.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["Prophet", "Pandas", "Scikit-Learn", "Matplotlib"],
                    "key_deliverables": [
                        "Prophet forecasting model integrating holiday effects and weekly cycles",
                        "Cross-validation report calculating Mean Absolute Percentage Error (MAPE)",
                        "Interactive forecasting components charts showing long-term trends"
                    ]
                }
            ]

    elif "fastapi" in topic_lower or "microservice" in topic_lower or "deploy" in topic_lower or "service" in topic_lower:
        if is_beg:
            return [
                {
                    "title": "Simple Iris Classifier API",
                    "description": "Build a FastAPI microservice that serves predictions from a trained Iris classifier.",
                    "difficulty": "Beginner",
                    "tech_stack": ["FastAPI", "Scikit-Learn", "Joblib", "Uvicorn"],
                    "key_deliverables": [
                        "FastAPI post endpoint taking four numerical inputs and returning predictions",
                        "Robust input validation model built with Pydantic",
                        "Self-documenting Swagger UI interface"
                    ]
                }
            ]
        elif is_adv:
            return [
                {
                    "title": "Distributed ML Service on Kubernetes",
                    "description": "Deploy a high-throughput recommendation model using Docker, Redis, and Kubernetes.",
                    "difficulty": "Advanced",
                    "tech_stack": ["FastAPI", "Docker", "Redis", "Kubernetes", "Prometheus"],
                    "key_deliverables": [
                        "Multi-stage build Dockerfile minimizing container size",
                        "Redis layer caching model inference keys to minimize Latency",
                        "Kubernetes manifest files outlining deployment, service, and Horizontal Pod Autoscaler configs"
                    ]
                }
            ]
        else: # Intermediate
            return [
                {
                    "title": "Asynchronous Batch Inference Service",
                    "description": "Build a batch processing pipeline handling multiple inference requests via Celery and Redis.",
                    "difficulty": "Intermediate",
                    "tech_stack": ["FastAPI", "Celery", "Redis", "Scikit-Learn"],
                    "key_deliverables": [
                        "Asynchronous endpoints returning task IDs for long-running inference jobs",
                        "Celery background workers processing tasks off the main thread",
                        "Task status dashboard monitoring active queues"
                    ]
                }
            ]

    # Fallback default recommendations
    if is_beg:
        return [
            {
                "title": "Exploratory Data Analysis (EDA) on House Sales",
                "description": "Clean, visualize, and analyze a dataset of house prices to identify key drivers of home values.",
                "difficulty": "Beginner",
                "tech_stack": ["Python", "Pandas", "Matplotlib", "Seaborn"],
                "key_deliverables": [
                    "Data cleaning report handling missing prices and values",
                    "Correlation heatmap identifying variables related to house prices",
                    "Bar plots showing average price per zip code"
                ]
            },
            {
                "title": "Titanic Survival Classifier",
                "description": "Build a baseline classification model to predict passenger survival based on age, gender, and ticket class.",
                "difficulty": "Beginner",
                "tech_stack": ["Scikit-Learn", "Pandas", "Logistic Regression"],
                "key_deliverables": [
                    "Categorical encoding for passenger gender and port of embarkation",
                    "Trained Logistic Regression model with >75% validation accuracy",
                    "Evaluation confusion matrix"
                ]
            }
        ]
    elif is_adv:
        return [
            {
                "title": "Customer Churn Prediction API Service",
                "description": "Train a high-performance ensemble classifier to predict churn, and wrap it in a microservice using FastAPI.",
                "difficulty": "Advanced",
                "tech_stack": ["XGBoost", "FastAPI", "Docker", "Scikit-Learn", "MLflow"],
                "key_deliverables": [
                    "SMOTE feature balance handling churn label skewness",
                    "Hyperparameter tuning pipeline tracking metrics via MLflow",
                    "FastAPI REST endpoint accepting features and returning churn probabilities",
                    "Docker container for deployment"
                ]
            }
        ]
    else: # Intermediate
        return [
            {
                "title": "Customer Segmentation using K-Means",
                "description": "Perform unsupervised learning to cluster retail buyers based on purchasing frequency, recency, and monetary value.",
                "difficulty": "Intermediate",
                "tech_stack": ["Scikit-Learn", "Pandas", "PCA", "Yellowbrick"],
                "key_deliverables": [
                    "RFM feature calculation script from raw transactional rows",
                    "Elbow plot and Silhouette analysis determining optimal cluster count",
                    "PCA projection 3D scatter plot of customer clusters"
                ]
            }
        ]

@app.post("/api/v1/quiz")
def generate_quiz(req: QuizRequest):
    topic = req.topic
    
    # Try Ollama Llama 3
    system_prompt = (
        "You are an expert AI Data Science Quiz Generator. Generate a multiple-choice quiz with exactly 5 questions "
        "on the requested topic. You MUST respond ONLY with a raw JSON array matching this structure. "
        "Do not wrap in markdown code blocks or add any other text outside the JSON:\n"
        "[\n"
        "  {\n"
        '    "id": 1,\n'
        '    "question": "...",\n'
        '    "options": ["option1", "option2", "option3", "option4"],\n'
        '    "correct_answer": "option1",\n'
        '    "explanation": "..."\n'
        "  }, ...\n"
        "]"
    )
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": f"Topic: {topic}",
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if isinstance(parsed, list) and len(parsed) > 0:
                filtered = _filter_quiz_questions_for_topic(topic, parsed)
                if len(filtered) > 0:
                    return filtered
    except Exception as e:
        print(f"Ollama quiz generation failed ({e}). Using mock fallback.")
        
    return mock_quiz_response(topic)

@app.post("/api/v1/projects")
def get_project_recommendations(req: ProjectRequest):
    difficulty = req.difficulty
    topic = req.topic
    
    # Try Ollama Llama 3
    system_prompt = (
        "You are an expert AI Data Science Project Advisor. Generate exactly 2 custom project recommendations matching the requested topic "
        "and difficulty level. You MUST respond ONLY with a raw JSON array matching this structure. "
        "Do not wrap in markdown code blocks or add any other text outside the JSON:\n"
        "[\n"
        "  {\n"
        '    "title": "...",\n'
        '    "description": "...",\n'
        '    "difficulty": "...",\n'
        '    "tech_stack": ["...", "..."],\n'
        '    "key_deliverables": ["...", "..."]\n'
        "  }, ...\n"
        "]"
    )
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": f"Difficulty: {difficulty}, Topic: {topic}",
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
    except Exception as e:
        print(f"Ollama project advising failed ({e}). Using mock fallback.")
        
    return mock_projects_response(difficulty, topic)

class ChartDescriptionRequest(BaseModel):
    chart_type: str
    x_column: str
    y_column: str
    x_stats: Optional[Dict[str, Any]] = None
    y_stats: Optional[Dict[str, Any]] = None

@app.post("/api/v1/describe-chart")
def describe_chart(req: ChartDescriptionRequest):
    """Generate a rich plain-English explanation of the visualization chart patterns."""
    chart_type = req.chart_type
    x_col = req.x_column
    y_col = req.y_column
    
    prompt = (
        f"You are an expert Data Analyst and AI Data Science Mentor. Your task is to provide a brief, professional, "
        f"and insightful 3-4 sentence explanation/description of a {chart_type} that plots '{y_col}' against '{x_col}'. "
        f"Here are the summary statistics of the data:\n"
        f"X-Axis Column '{x_col}': {req.x_stats}\n"
        f"Y-Axis Column '{y_col}': {req.y_stats}\n\n"
        f"Identify key trends, correlations, or properties that a student of data science should notice "
        f"when analyzing this chart. Keep the response encouraging, structured, and informative. Do not wrap in extra commentary."
    )
    
    # Try calling Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            description = response.json().get("response", "").strip()
            return {"description": description}
    except Exception as e:
        print(f"Ollama chart description failed ({e}). Using rules-based fallback.")
        
    # Rules-based fallback if Ollama is not configured/running
    description = (
        f"This **{chart_type}** visualizes the distribution and mapping between the independent variable "
        f"**{x_col}** on the X-axis and the dependent variable **{y_col}** on the Y-axis.\n\n"
    )
    
    if req.x_stats and req.y_stats:
        x_mean = req.x_stats.get("mean", None)
        y_mean = req.y_stats.get("mean", None)
        if x_mean is not None and y_mean is not None:
            description += (
                f"Statistical insights show that **{x_col}** has a mean of `{x_mean:.2f}` and "
                f"**{y_col}** has a mean of `{y_mean:.2f}`. "
            )
        else:
            x_uniq = req.x_stats.get("unique_count", None)
            y_uniq = req.y_stats.get("unique_count", None)
            if x_uniq is not None:
                description += f"**{x_col}** contains `{x_uniq}` unique categories. "
            if y_uniq is not None:
                description += f"**{y_col}** contains `{y_uniq}` unique categories. "
                
    description += (
        f"This visualization helps students inspect how **{y_col}** correlates or trends with "
        f"**{x_col}**. Analyzing such charts is a vital Exploratory Data Analysis (EDA) step to identify patterns, "
        f"skewed features, and variance distribution before feeding variables into downstream machine learning models."
    )
    
    return {"description": description}

class VoiceMentorRequest(BaseModel):
    message: str

@app.post("/api/v1/voice-mentor")
def voice_mentor(req: VoiceMentorRequest):
    """Answers spoken questions using beginner-friendly markdown text and a concise spoken audio text response."""
    message = req.message
    
    system_prompt = (
        "You are an AI Voice Mentor for data science students. Listen to the student's spoken question, "
        "convert speech to text, understand the query, and provide a clear, beginner-friendly explanation. "
        "Respond both in text and as a concise spoken-style answer. If the student asks about coding, "
        "explain the concept first and then provide an example. Keep answers supportive, accurate, and easy to understand."
    )
    
    prompt = (
        f"System: {system_prompt}\n"
        f"Student Spoken Query: {message}\n\n"
        f"Provide your response. You MUST respond ONLY with a raw JSON object matching the following structure. "
        f"Do not wrap the output in markdown code blocks or add any other text outside the JSON:\n"
        f"{{\n"
        f"  \"reply\": \"Full explanation with code block examples using markdown formatting if applicable...\",\n"
        f"  \"spoken_reply\": \"Concise spoken-style 2-3 sentence answer suitable for reading aloud via text-to-speech...\"\n"
        f"}}"
    )
    
    # Try calling Ollama Llama 3
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            import json
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if "reply" in parsed and "spoken_reply" in parsed:
                return parsed
    except Exception as e:
        print(f"Ollama voice mentor failed ({e}). Using rule-based fallback.")
        
    # Rule-based fallback if Ollama is not configured
    msg_lower = message.lower()
    
    if "python" in msg_lower:
        reply = (
            "Python is a high-level, general-purpose programming language. For data science, it is the industry standard "
            "because of its readability and robust libraries like Pandas and NumPy.\n\n"
            "**Code Example:**\n"
            "```python\n"
            "# Defining a simple list comprehension in Python\n"
            "squares = [x**2 for x in range(5)]\n"
            "print(squares) # Output: [0, 1, 4, 9, 16]\n"
            "```"
        )
        spoken = "Python is a simple, readable programming language that is the standard in data science. An example is using list comprehensions to write concise loops, like creating a list of squared numbers."
    elif "sql" in msg_lower:
        reply = (
            "SQL (Structured Query Language) is used to communicate with databases. It allows you to query, filter, "
            "and aggregate records.\n\n"
            "**Code Example:**\n"
            "```sql\n"
            "SELECT department, AVG(salary) \n"
            "FROM employees \n"
            "GROUP BY department;\n"
            "```"
        )
        spoken = "SQL is the standard language for querying and managing relational databases. You can use it to select data, filter rows, or group records, such as calculating the average salary per department."
    else:
        reply = (
            f"I heard you ask about: '{message}'. Data science involves analyzing data, building predictive models, "
            f"and communicating insights. Let me know if you want to write some Python, query SQL, or discuss statistical assumptions!"
        )
        spoken = f"I received your question about {message}. Data science combines programming, statistics, and business insight to solve complex problems. Let me know what specific topic you would like to explore next."
        
    return {"reply": reply, "spoken_reply": spoken}

# Interview fallback questions
INTERVIEW_FALLBACK_QUESTIONS = {
    "python programming": [
        {"question": "Write a function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome. Consider edge cases like whitespace and casing. Time limit: 5 minutes.", "time_limit": 300},
        {"question": "How does a generator differ from a normal list in Python? Provide an example generator function. Time limit: 5 minutes.", "time_limit": 300},
        {"question": "Explain the GIL (Global Interpreter Lock) in Python and how it affects multi-threaded programs. Time limit: 5 minutes.", "time_limit": 300}
    ],
    "sql databases": [
        {"question": "Write a SQL query to find the second highest salary from an `Employee` table. If there is no second highest salary, return NULL. Time limit: 5 minutes.", "time_limit": 300},
        {"question": "Explain the difference between WHERE and HAVING clauses. Give an example query utilizing both. Time limit: 5 minutes.", "time_limit": 300},
        {"question": "What are database indexes, and how do B-Trees optimize SELECT queries? Are there any disadvantages to indexes? Time limit: 5 minutes.", "time_limit": 300}
    ],
    "machine learning": [
        {"question": "Explain the bias-variance tradeoff. How do you identify if a model has high bias or high variance? Time limit: 5 minutes.", "time_limit": 300},
        {"question": "Explain how you would compute precision and recall from a confusion matrix. Explain the difference between them. Time limit: 5 minutes.", "time_limit": 300},
        {"question": "How does Random Forest handle feature importance? How is it calculated? Time limit: 5 minutes.", "time_limit": 300}
    ],
    "statistics & probability": [
        {"question": "Explain p-value. How does it relate to the null hypothesis? If a p-value is 0.03 under alpha=0.05, what is your decision? Time limit: 5 minutes.", "time_limit": 300},
        {"question": "What is Bayes' Theorem? Provide the formula and describe a real-world scenario where it is applied. Time limit: 5 minutes.", "time_limit": 300},
        {"question": "What is the Central Limit Theorem? Why is it fundamental to statistical inference? Time limit: 5 minutes.", "time_limit": 300}
    ]
}

@app.post("/api/v1/interview/start")
def interview_start(req: InterviewStartRequest):
    topic = req.topic.strip().lower()
    
    # 1. Build prompt for Ollama Llama 3
    system_prompt = (
        "You are an elite Data Science Interviewer. Your task is to generate 3 realistic, sequential interview questions "
        "tailored to the chosen topic. Each question must have a coding or technical focus, and include a time limit (e.g. 5 minutes).\n"
        "Return ONLY a raw JSON array matching this schema: "
        "[{\"question\": \"Question text...\", \"time_limit\": 300}, ...]. "
        "Do not wrap in markdown code blocks or add explanations."
    )
    
    prompt = (
        f"Generate 3 realistic interview questions for the topic: {req.topic}. "
        "Return only the JSON array."
    )
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            import json
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if isinstance(parsed, list) and len(parsed) > 0 and "question" in parsed[0]:
                return {"questions": parsed}
    except Exception as e:
        print(f"Ollama interview start failed ({e}). Using rules-based fallback.")
        
    # Fallback to rules-based questions
    questions = INTERVIEW_FALLBACK_QUESTIONS.get(topic, INTERVIEW_FALLBACK_QUESTIONS["python programming"])
    return {"questions": questions}

@app.post("/api/v1/interview/evaluate")
def interview_evaluate(req: InterviewEvaluateRequest):
    topic = req.topic.strip().lower()
    question = req.question
    user_answer = req.user_answer
    
    # 1. Build prompt for Ollama Llama 3
    system_prompt = (
        "You are a Data Science Interviewer. Evaluate the student's answer to the coding/technical interview question.\n"
        "Grade the response based on four criteria: Correctness, Clarity, Problem-Solving Approach, and Code Quality.\n"
        "Provide a score out of 10 and detailed feedback.\n"
        "You MUST respond ONLY with a raw JSON object matching the following structure. Do not wrap in markdown code blocks or add any other text outside the JSON:\n"
        "{\n"
        "  \"score\": 8,\n"
        "  \"feedback\": \"Your explanation...\",\n"
        "  \"correctness\": \"Evaluation of correctness...\",\n"
        "  \"clarity\": \"Evaluation of clarity...\",\n"
        "  \"approach\": \"Evaluation of problem-solving approach...\",\n"
        "  \"code_quality\": \"Evaluation of code quality...\"\n"
        "}"
    )
    
    prompt = (
        f"Topic: {req.topic}\n"
        f"Question: {question}\n"
        f"Student Answer: {user_answer}\n"
    )
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            import json
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if "score" in parsed and "feedback" in parsed:
                return parsed
    except Exception as e:
        print(f"Ollama interview evaluate failed ({e}). Using rules-based fallback.")
        
    # Rule-based fallback
    ans_lower = user_answer.lower()
    score = 7  # default
    feedback = "Good attempt. Your answer covers some base requirements, but could be made more precise."
    correctness = "Mostly correct, but misses some key edge cases or optimizations."
    clarity = "The explanation is reasonably clear but could structure the logic better."
    approach = "Logical approach, though there are more optimal ways to handle the complexity."
    code_quality = "Basic code format is okay; try using cleaner variable naming or descriptive comments."
    
    # Topic specific evaluations
    if "python programming" in topic:
        if "is_palindrome" in question:
            if "replace(" in ans_lower or "lower()" in ans_lower or "[::-1]" in ans_lower:
                score = 9
                feedback = "Excellent! You correctly used string slicing `[::-1]` for reversing and handled casing/whitespace via `.lower()` or `.replace()`."
                correctness = "Fully correct and handles case/whitespace differences."
                clarity = "Clear and easy to understand logic."
                approach = "Optimal O(N) time and O(N) space. For O(1) space, a two-pointer approach could be used."
                code_quality = "High quality code."
            elif "def " not in ans_lower:
                score = 4
                feedback = "You should define the function `is_palindrome` as requested by the prompt."
                correctness = "Function definition is missing."
        elif "generator" in question:
            if "yield" in ans_lower:
                score = 9
                feedback = "Great explanation. Generators use the `yield` keyword and evaluate lazily (saving memory), whereas lists are fully computed in memory."
                correctness = "Correct distinction between lazy generation and eager evaluation."
                clarity = "Excellent clear conceptual differentiation."
                approach = "Correct code example showing a standard generator."
                code_quality = "Good Python syntax."
    elif "sql databases" in topic:
        if "second highest" in question:
            if "limit" in ans_lower or "offset" in ans_lower or "subquery" in ans_lower or "max(" in ans_lower:
                score = 9
                feedback = "Very good. Using `LIMIT 1 OFFSET 1` with a subquery (or sorting with distinct) is a standard approach to find the second highest salary."
                correctness = "Correct query syntax."
                clarity = "Logic is well explained."
                approach = "Optimal SQL query approach."
                code_quality = "Clean SQL capitalization."
    elif "machine learning" in topic:
        if "bias-variance" in question:
            if "overfit" in ans_lower or "underfit" in ans_lower:
                score = 9
                feedback = "Excellent definition. High bias causes underfitting (simplistic model), whereas high variance causes overfitting (model fits noise)."
                correctness = "Accurate conceptual understanding of the tradeoff."
                clarity = "Clear and well-structured explanation."
                approach = "Good identification guidelines (evaluating train vs validation error)."
                code_quality = "Concept explanation (no code required)."
    elif "statistics & probability" in topic:
        if "p-value" in question:
            if "reject" in ans_lower or "null" in ans_lower:
                score = 9
                feedback = "Correct! Since the p-value (0.03) is less than the significance level alpha (0.05), you reject the null hypothesis."
                correctness = "Correct decision rule and decision."
                clarity = "Clear, direct explanation."
                approach = "Excellent scientific approach."
                code_quality = "Concept explanation."

    return {
        "score": score,
        "feedback": feedback,
        "correctness": correctness,
        "clarity": clarity,
        "approach": approach,
        "code_quality": code_quality
    }

@app.post("/api/v1/interview/assess")
def interview_assess(req: InterviewAssessRequest):
    # 1. Build prompt for Ollama Llama 3
    system_prompt = (
        "You are a Data Science Interview Trainer. Provide a final overall assessment of the student's mock interview.\n"
        "Compile a detailed summary containing:\n"
        "1. Overall Score out of 10\n"
        "2. Strengths\n"
        "3. Weaknesses\n"
        "4. Specific Improvement Suggestions\n"
        "You MUST respond ONLY with a raw JSON object matching the following structure. Do not wrap in markdown code blocks or add any other text outside the JSON:\n"
        "{\n"
        "  \"overall_score\": 7.5,\n"
        "  \"strengths\": \"Bulleted markdown list of strengths...\",\n"
        "  \"weaknesses\": \"Bulleted markdown list of weaknesses...\",\n"
        "  \"suggestions\": \"Bulleted markdown list of suggestions...\"\n"
        "}"
    )
    
    history_str = ""
    for idx, (q, a, e) in enumerate(zip(req.questions, req.answers, req.evaluations)):
        history_str += f"Q{idx+1}: {q}\nStudent Answer: {a}\nScore: {e.get('score')}/10\nFeedback: {e.get('feedback')}\n\n"
        
    prompt = (
        f"Topic: {req.topic}\n"
        f"Interview History:\n{history_str}"
        f"Generate the overall assessment."
    )
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            import json
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if "overall_score" in parsed and "strengths" in parsed:
                return parsed
    except Exception as e:
        print(f"Ollama interview assess failed ({e}). Using rules-based fallback.")
        
    # Rule-based fallback
    total_score = sum(e.get("score", 7) for e in req.evaluations)
    avg_score = round(total_score / max(len(req.evaluations), 1), 1)
    
    strengths = (
        "- Solid conceptual grasp of the basic subject terminology.\n"
        "- Capable of writing syntactically correct snippets for basic problems.\n"
        "- Good effort in addressing coding problems directly without leaving blank."
    )
    
    weaknesses = (
        "- May miss edge cases (e.g. casing/special characters in string processing, NULL values in database records).\n"
        "- Could detail explanations more thoroughly to show deeper architectural knowledge.\n"
        "- Variable naming and inline comments could be improved for production readiness."
    )
    
    suggestions = (
        "- **Practice Edge Cases**: When solving coding problems, explicitly declare and test constraints and null values.\n"
        "- **Study Vectorization and Optimization**: Focus on performance and memory usage (e.g., using generators or vectorized queries).\n"
        "- **Simulate Timed Environments**: Solve Leetcode/HackerRank questions under strict time limits to build confidence."
    )
    
    return {
        "overall_score": avg_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions
    }

# Challenge fallback templates
CHALLENGE_FALLBACKS = {
    "python": {
        "beginner": {
            "title": "Reverse Words in a String",
            "description": "Write a function `reverse_words(s: str) -> str` that reverses the order of words in a given string. A word is defined as a sequence of non-space characters. The words in the input string may be separated by multiple spaces, but the output should only have a single space separating words, with no leading or trailing spaces.",
            "template": "def reverse_words(s: str) -> str:\n    # Write your solution code here\n    pass\n",
            "sample_input": "'  hello world  '",
            "sample_output": "'world hello'",
            "constraints": "- 1 <= len(s) <= 10^4\n- The string contains English letters, digits, and spaces.",
            "hints": "- Split the string into words using `.split()` which automatically handles multiple spaces.\n- Reverse the list of words.\n- Join them back with a single space using `' '.join()`.",
            "learning_objective": "Master basic string operations, list manipulation, and whitespace handling in Python."
        },
        "intermediate": {
            "title": "Find All Anagrams in a String",
            "description": "Write a function `find_anagrams(s: str, p: str) -> list` that takes two strings `s` and `p`, and returns an array of all the start indices of `p`'s anagrams in `s`. The anagram indices can be returned in any order. An anagram is a word formed by rearranging the letters of another, such as *cinema* and *iceman*.",
            "template": "def find_anagrams(s: str, p: str) -> list:\n    # Write your solution code here\n    pass\n",
            "sample_input": "s = 'cbaebabacd', p = 'abc'",
            "sample_output": "[0, 6]",
            "constraints": "- 1 <= len(s), len(p) <= 2 * 10^4\n- s and p consist of lowercase English letters.",
            "hints": "- Use the sliding window technique with two pointers.\n- Keep track of character counts in the window using a frequency array or hash map.",
            "learning_objective": "Implement sliding window algorithms, hash mapping, and count optimization in Python."
        },
        "advanced": {
            "title": "Merge k Sorted Lists",
            "description": "Write a function `merge_k_lists(lists: list) -> list` that takes an array of `k` sorted lists, and merges them all into one sorted list. Return the merged sorted list.",
            "template": "def merge_k_lists(lists: list) -> list:\n    # Write your solution code here\n    pass\n",
            "sample_input": "[[1, 4, 5], [1, 3, 4], [2, 6]]",
            "sample_output": "[1, 1, 2, 3, 4, 4, 5, 6]",
            "constraints": "- k == len(lists)\n- 0 <= k <= 10^4\n- 0 <= len(lists[i]) <= 500\n- lists[i] is sorted in ascending order.",
            "hints": "- You can use a min-heap (Priority Queue) to efficiently fetch the smallest element across all lists at each step.\n- Alternatively, divide and conquer by repeatedly merging pairs of lists.",
            "learning_objective": "Utilize advanced data structures (Heaps / Priority Queues) and divide-and-conquer optimization techniques."
        }
    },
    "sql": {
        "beginner": {
            "title": "Find Customers Who Never Order",
            "description": "Given a `Customers` table (with columns `id`, `name`) and an `Orders` table (with columns `id`, `customerId`), write a SQL query to find all customers who have never ordered anything.",
            "template": "SELECT name AS Customers \nFROM Customers\n-- Write your SQL query here\n",
            "sample_input": "Customers: {id: 1, name: 'Joe'}, Orders: {id: 1, customerId: 3}",
            "sample_output": "'Joe'",
            "constraints": "- id is the primary key of Customers table.\n- customerId is a foreign key referring to Customers.",
            "hints": "- Use a `LEFT JOIN` on customerId and look for rows where the joined `Orders.id` IS NULL.\n- Alternatively, use a subquery with `NOT IN` or `NOT EXISTS`.",
            "learning_objective": "Understand basic relational filtering using joins, null comparisons, and filtering criteria."
        },
        "intermediate": {
            "title": "Highest Grade For Each Student",
            "description": "Given an `Enrollments` table (with columns `student_id`, `course_id`, `grade`), write a SQL query to find the highest grade for each student. If there is a tie, return the course with the lowest `course_id` first. Order the result by `student_id` in ascending order.",
            "template": "SELECT student_id, course_id, grade\nFROM Enrollments\n-- Write your SQL query here\n",
            "sample_input": "Enrollments: {student_id: 1, course_id: 2, grade: 95}, {student_id: 1, course_id: 3, grade: 95}",
            "sample_output": "student_id: 1, course_id: 2, grade: 95",
            "constraints": "- (student_id, course_id) is the primary key of Enrollments.",
            "hints": "- Use SQL Window Functions, specifically `ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY grade DESC, course_id ASC)`.",
            "learning_objective": "Master SQL window functions (ROW_NUMBER, PARTITION BY) and multi-level sorting constraints."
        },
        "advanced": {
            "title": "Active Users Session Analytics",
            "description": "Given a `UserLogins` table (with columns `user_id`, `login_date`), write a SQL query to find the active users who logged in for 3 or more consecutive days. Return the result table ordered by `user_id`.",
            "template": "SELECT DISTINCT user_id\nFROM UserLogins\n-- Write your SQL query here\n",
            "sample_input": "UserLogins: {user_id: 1, login_date: '2026-06-01'}, {user_id: 1, login_date: '2026-06-02'}, {user_id: 1, login_date: '2026-06-03'}",
            "sample_output": "user_id: 1",
            "constraints": "- There are no duplicate logins for a user on the same date.",
            "hints": "- Use `LEAD` or `LAG` window functions to check dates of consecutive rows.\n- Alternatively, subtract `ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date)` from the `login_date` to group consecutive logins.",
            "learning_objective": "Implement complex SQL consecutive data clustering and advanced temporal analysis."
        }
    },
    "data analysis": {
        "beginner": {
            "title": "Calculate Missing Weights",
            "description": "Write a python Pandas function `impute_missing_weights(df: pd.DataFrame) -> pd.DataFrame` that replaces all missing values (NaN) in a column named `weight` with the median weight of the dataset, and returns the modified DataFrame.",
            "template": "import pandas as pd\n\ndef impute_missing_weights(df: pd.DataFrame) -> pd.DataFrame:\n    # Write your solution code here\n    pass\n",
            "sample_input": "pd.DataFrame({'weight': [10.0, None, 15.0, 20.0]})",
            "sample_output": "pd.DataFrame({'weight': [10.0, 15.0, 15.0, 20.0]})",
            "constraints": "- df contains a column named 'weight'.\n- 1 <= len(df) <= 10^5",
            "hints": "- Calculate the median using `df['weight'].median()`.\n- Fill missing records in the Series using `df['weight'].fillna()`.",
            "learning_objective": "Perform basic missing data profiling and median imputation using Pandas."
        },
        "intermediate": {
            "title": "Calculate Correlation Metrics",
            "description": "Write a python Pandas function `find_highly_correlated(df: pd.DataFrame, threshold: float) -> list` that calculates the Pearson correlation matrix for numerical features in a DataFrame. Return a list of tuples `(feature_a, feature_b)` for pairs of features that have an absolute correlation value strictly greater than the given `threshold`. Do not include duplicate pairs (e.g. if `(A, B)` is in the list, `(B, A)` should not be, nor should self-correlations `(A, A)`).",
            "template": "import pandas as pd\n\ndef find_highly_correlated(df: pd.DataFrame, threshold: float) -> list:\n    # Write your solution code here\n    pass\n",
            "sample_input": "df = pd.DataFrame({'A': [1,2,3], 'B': [2,4,5.9], 'C': [10, 1, 5]}), threshold = 0.95",
            "sample_output": "[('A', 'B')]",
            "constraints": "- threshold is between 0 and 1.\n- df has at least 2 numerical columns.",
            "hints": "- Compute the Pearson correlation matrix using `df.corr()`.\n- Use `np.triu()` or index comparisons to only inspect the upper-triangle of the matrix.",
            "learning_objective": "Perform Exploratory Data Analysis (EDA) correlation computations and filter collinear variables."
        },
        "advanced": {
            "title": "Calculate Cohort Retention Rates",
            "description": "Write a Pandas function `calculate_cohort_retention(df: pd.DataFrame) -> pd.DataFrame` that takes an transaction ledger DataFrame containing `user_id` and `transaction_date`. Return a pivot table showing the user retention rate percentage over subsequent months, where index represents cohort month and columns represent months since purchase (Month 0, Month 1, etc.).",
            "template": "import pandas as pd\n\ndef calculate_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:\n    # Write your solution code here\n    pass\n",
            "sample_input": "df with transaction rows",
            "sample_output": "Pivot DataFrame with percentage retention",
            "constraints": "- transaction_date is parsed as DateTime.",
            "hints": "- First determine the cohort group (first purchase month) for each user.\n- Group by cohort month and transaction month, count unique users, and pivot.",
            "learning_objective": "Build advanced customer lifecycle segmentation metrics using pivot tables and complex Pandas grouping."
        }
    }
}

@app.post("/api/v1/challenge/generate")
def challenge_generate(req: ChallengeGenerateRequest):
    language = req.language.strip().lower()
    level = req.level.strip().lower()
    
    # 1. Build prompt for Ollama Llama 3
    system_prompt = (
        "You are an elite Data Science Mentor. Generate a new, unique, and practical coding challenge for a data science student.\n"
        "The response MUST be a raw JSON object matching the following structure, with no markdown code blocks:\n"
        "{\n"
        "  \"title\": \"Challenge Title\",\n"
        "  \"description\": \"Detailed problem statement...\",\n"
        "  \"template\": \"Starting code template...\",\n"
        "  \"sample_input\": \"Sample input format...\",\n"
        "  \"sample_output\": \"Sample output format...\",\n"
        "  \"constraints\": \"Bulleted markdown constraints...\",\n"
        "  \"hints\": \"Bulleted markdown hints...\",\n"
        "  \"learning_objective\": \"Expected learning objective...\"\n"
        "}"
    )
    
    prompt = (
        f"Generate a {level} coding challenge focusing on {language}. "
        "Ensure the task is practical and relevant for a data scientist. "
        "Return ONLY the raw JSON object."
    )
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            import json
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if "title" in parsed and "description" in parsed:
                return parsed
    except Exception as e:
        print(f"Ollama challenge generate failed ({e}). Using rules-based fallback.")
        
    # Fallback logic
    lang_data = CHALLENGE_FALLBACKS.get(language, CHALLENGE_FALLBACKS["python"])
    challenge = lang_data.get(level, lang_data["beginner"])
    return challenge

@app.post("/api/v1/challenge/evaluate")
def challenge_evaluate(req: ChallengeEvaluateRequest):
    language = req.language.strip().lower()
    level = req.level.strip().lower()
    title = req.title
    description = req.description
    user_code = req.user_code
    
    # 1. Build prompt for Ollama Llama 3
    system_prompt = (
        "You are an elite Data Science Code Evaluator. Evaluate the student's solution code for the coding challenge.\n"
        "Evaluate correctness, performance, and best practices. Rate out of 100.\n"
        "The response MUST be a raw JSON object matching the following structure, with no markdown code blocks:\n"
        "{\n"
        "  \"status\": \"Success\" | \"Failed\" | \"Needs Optimization\",\n"
        "  \"score\": 90,\n"
        "  \"feedback\": \"Detailed grading feedback explaining correct parts and suggestions...\",\n"
        "  \"optimal_solution\": \"Annotated optimal solution code block...\"\n"
        "}"
    )
    
    prompt = (
        f"Language: {req.language}\n"
        f"Level: {req.level}\n"
        f"Challenge: {title}\n"
        f"Description: {description}\n"
        f"Student Solution:\n{user_code}\n"
    )
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        }, timeout=8)
        if response.status_code == 200:
            import json
            text_result = response.json().get("response", "").strip()
            if text_result.startswith("```"):
                text_result = re.sub(r"^```(?:json)?\n", "", text_result)
                text_result = re.sub(r"\n```$", "", text_result)
                text_result = text_result.strip()
            parsed = json.loads(text_result)
            if "status" in parsed and "feedback" in parsed:
                return parsed
    except Exception as e:
        print(f"Ollama challenge evaluate failed ({e}). Using rules-based fallback.")
        
    # Rule-based fallback
    code_lower = user_code.lower()
    status = "Success"
    score = 95
    feedback = "Fantastic solution! Your code matches the challenge guidelines and compiles cleanly."
    optimal = ""
    
    if "python" in language:
        if "reverse_words" in title.lower():
            if "split(" in code_lower and "join(" in code_lower:
                optimal = "```python\ndef reverse_words(s: str) -> str:\n    # Split automatically segments by any whitespace run and cleans leading/trailing spaces\n    words = s.split()\n    # Reverse and join back with single space\n    return ' '.join(words[::-1])\n```"
            else:
                status = "Needs Optimization"
                score = 70
                feedback = "Your solution works but could be made simpler by leveraging built-in python list slicing ([::-1]) and ' '.join()."
                optimal = "```python\ndef reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])\n```"
        elif "anagrams" in title.lower():
            optimal = "```python\ndef find_anagrams(s: str, p: str) -> list:\n    # Optimized sliding window using character counts\n    from collections import Counter\n    p_count = Counter(p)\n    s_count = Counter()\n    res = []\n    ns, np = len(s), len(p)\n    for i in range(ns):\n        s_count[s[i]] += 1\n        if i >= np:\n            if s_count[s[i-np]] == 1:\n                del s_count[s[i-np]]\n            else:\n                s_count[s[i-np]] -= 1\n        if s_count == p_count:\n            res.append(i - np + 1)\n    return res\n```"
    elif "sql" in language:
        if "never order" in title.lower():
            optimal = "```sql\n-- Optimal solution using LEFT JOIN\nSELECT c.name AS Customers\nFROM Customers c\nLEFT JOIN Orders o ON c.id = o.customerId\nWHERE o.id IS NULL;\n```"
        elif "highest grade" in title.lower():
            optimal = "```sql\n-- Optimal solution using ROW_NUMBER window function\nWITH RankedEnrollments AS (\n    SELECT student_id, course_id, grade,\n           ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY grade DESC, course_id ASC) as rn\n    FROM Enrollments\n)\nSELECT student_id, course_id, grade\nFROM RankedEnrollments\nWHERE rn = 1\nORDER BY student_id;\n```"

    return {
        "status": status,
        "score": score,
        "feedback": feedback,
        "optimal_solution": optimal
    }

class ExplainRequest(BaseModel):


    csv_data: str                          # base64-encoded CSV string
    features: List[str]
    target: str
    model_name: str
    test_size: Optional[float] = 0.2
    explain_row_index: Optional[int] = None  # index into test set for individual prediction

def _build_explainer_narrative(
    model_name: str,
    task_type: str,
    train_score: float,
    test_score: float,
    top_features: List[dict],
    metric_name: str,
) -> str:
    """Generate a rich plain-English explanation of the trained model."""
    overfit_gap = train_score - test_score
    if overfit_gap > 0.15:
        overfit_msg = (
            f"⚠️ **Overfitting Detected**: There is a notable gap between train {metric_name} "
            f"({train_score:.1%}) and test {metric_name} ({test_score:.1%}). "
            "The model has memorized training patterns rather than learning general rules. "
            "Consider adding regularization, reducing model complexity, or increasing dataset size."
        )
        health_emoji = "🔴"
        health_label = "Overfitting Risk"
    elif overfit_gap > 0.08:
        overfit_msg = (
            f"🟡 **Mild Generalization Gap**: Train {metric_name} ({train_score:.1%}) is somewhat higher "
            f"than test {metric_name} ({test_score:.1%}). "
            "The model generalizes reasonably but may benefit from cross-validation tuning."
        )
        health_emoji = "🟡"
        health_label = "Mild Generalization Gap"
    else:
        overfit_msg = (
            f"✅ **Healthy Generalization**: Train {metric_name} ({train_score:.1%}) is close to "
            f"test {metric_name} ({test_score:.1%}). "
            "The model generalizes well to unseen data."
        )
        health_emoji = "🟢"
        health_label = "Healthy"

    top_feat_names = [f['feature'] for f in top_features[:3]]
    feat_list = ", ".join([f"**{f}**" for f in top_feat_names])

    narrative = (
        f"## 🤖 AI Model Explainer Report — {model_name}\n\n"
        f"### 🧠 What Did the Model Learn?\n\n"
        f"Your **{model_name}** was trained on a **{task_type}** task. "
        f"The most influential input signals driving predictions were {feat_list}. "
        f"This means the model found strong patterns in these features that predict the target variable.\n\n"
        f"### 📊 Performance Summary\n\n"
        f"- **Train {metric_name}**: `{train_score:.1%}`\n"
        f"- **Test {metric_name}**: `{test_score:.1%}`\n"
        f"- **Model Health**: {health_emoji} {health_label}\n\n"
        f"{overfit_msg}\n\n"
        f"### 🔍 Feature Impact Explanation\n\n"
    )

    for i, feat in enumerate(top_features[:5], 1):
        pct = feat['importance']
        direction = "↑ strongly positive" if i == 1 else ("↑ positive" if pct > 0.1 else "→ moderate")
        narrative += (
            f"{i}. **{feat['feature']}** — contributes `{pct:.1%}` of the model's decision weight. "
            f"This feature has a {direction} influence.\n"
        )

    narrative += (
        f"\n### 💡 Actionable Insights\n\n"
        f"- Focus on **{top_feat_names[0]}** as the primary driver of the outcome — "
        f"small changes here will have the largest impact on predictions.\n"
        f"- Consider running correlation analysis between **{top_feat_names[0]}** and the target to confirm causality.\n"
        f"- If you want to improve {metric_name}, consider engineering new features derived from "
        f"{', '.join(top_feat_names[:2])}.\n"
    )

    return narrative

@app.post("/api/v1/explain")
def explain_model(req: ExplainRequest):
    """Train a model on a CSV dataset and return SHAP-based AI explanations."""
    import base64
    import io
    import traceback
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as mticker

    try:
        # 1. Decode and load CSV
        csv_bytes = base64.b64decode(req.csv_data)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        X = df[req.features].copy()
        y = df[req.target].copy()

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        # 2. Determine task type
        is_numeric = pd.api.types.is_numeric_dtype(y)
        task_type = "regression" if (is_numeric and y.nunique() > 10) else "classification"

        # 3. Build model
        model_map_reg = {
            "Linear Regression": ("sklearn.linear_model", "LinearRegression", {}),
            "Ridge Regression": ("sklearn.linear_model", "Ridge", {}),
            "Lasso Regression": ("sklearn.linear_model", "Lasso", {}),
            "Decision Tree Regressor": ("sklearn.tree", "DecisionTreeRegressor", {"max_depth": 5, "random_state": 42}),
            "Random Forest Regressor": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 100, "max_depth": 6, "random_state": 42}),
            "Gradient Boosting Regressor": ("sklearn.ensemble", "GradientBoostingRegressor", {"n_estimators": 100, "random_state": 42}),
        }
        model_map_cls = {
            "Logistic Regression": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000}),
            "K-Nearest Neighbors (KNN)": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
            "Support Vector Classifier (SVC)": ("sklearn.svm", "SVC", {"probability": True, "random_state": 42}),
            "Decision Tree Classifier": ("sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5, "random_state": 42}),
            "Random Forest Classifier": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100, "max_depth": 6, "random_state": 42}),
            "Gradient Boosting Classifier": ("sklearn.ensemble", "GradientBoostingClassifier", {"n_estimators": 100, "random_state": 42}),
            "Naive Bayes": ("sklearn.naive_bayes", "GaussianNB", {}),
        }

        model_map = model_map_reg if task_type == "regression" else model_map_cls
        # Default fallback model
        if req.model_name not in model_map:
            default_name = "Random Forest Regressor" if task_type == "regression" else "Random Forest Classifier"
        else:
            default_name = req.model_name

        mod_module, mod_class, mod_kwargs = model_map[default_name]
        import importlib
        mod = importlib.import_module(mod_module)
        model_obj = getattr(mod, mod_class)(**mod_kwargs)

        clf = Pipeline([("preprocessor", preprocessor), ("model", model_obj)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=42)

        if task_type == "classification":
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # 4. Compute metrics
        if task_type == "regression":
            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)
            metric_name = "R²"
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            extra_metrics = {"rmse": rmse}
        else:
            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)
            metric_name = "Accuracy"
            f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
            extra_metrics = {"f1": f1}

        # 5. Extract feature importances
        try:
            feature_names_out = list(clf.named_steps["preprocessor"].get_feature_names_out())
            clean_names = [
                n[5:] if (n.startswith("num__") or n.startswith("cat__")) else n
                for n in feature_names_out
            ]
        except Exception:
            clean_names = req.features

        trained_model = clf.named_steps["model"]
        importances = None
        importance_type = "Importance"

        if hasattr(trained_model, "feature_importances_"):
            importances = trained_model.feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(trained_model, "coef_"):
            coef = trained_model.coef_
            importances = np.mean(np.abs(coef), axis=0) if len(coef.shape) > 1 else np.abs(coef)
            importance_type = "Coefficient Magnitude"

        top_features = []
        if importances is not None and len(importances) == len(clean_names):
            imp_pairs = sorted(
                zip(clean_names, importances.tolist()), key=lambda x: x[1], reverse=True
            )
            total_imp = sum(v for _, v in imp_pairs) or 1.0
            top_features = [
                {"feature": name, "importance": round(val / total_imp, 4)}
                for name, val in imp_pairs[:10]
            ]

        # 6. Generate feature importance chart (base64 PNG)
        feat_chart_b64 = None
        if top_features:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor("#0f1322")
            ax.set_facecolor("#0f1322")

            feat_df = pd.DataFrame(top_features[:8])
            colors = [
                f"#{int(99 + (i/8)*156):02x}{int(102 + (i/8)*(85-102)):02x}{int(241 + (i/8)*(247-241)):02x}"
                for i in range(len(feat_df))
            ]
            bars = ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#6366f1", alpha=0.85, edgecolor="none")
            # gradient-ish effect
            for idx, bar in enumerate(bars):
                alpha = 0.6 + 0.4 * (idx / max(len(bars), 1))
                bar.set_alpha(alpha)

            ax.set_xlabel(importance_type, color="#9ca3af")
            ax.set_title(f"🔑 Top Feature Importances — {req.model_name}", color="#f3f4f6", fontsize=12, fontweight="bold")
            ax.tick_params(colors="#e5e7eb")
            ax.xaxis.label.set_color("#9ca3af")
            for spine in ax.spines.values():
                spine.set_color("#374151")
            ax.grid(axis="x", color="#1e293b", linestyle="--", alpha=0.5)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
            buf.seek(0)
            feat_chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        # 7. SHAP analysis (optional, graceful fallback)
        shap_values_list = []
        shap_chart_b64 = None
        shap_available = False
        individual_explanation = None

        try:
            import importlib

            shap = importlib.import_module("shap")  # pyright: ignore[reportMissingImports]
            X_test_transformed = clf.named_steps["preprocessor"].transform(X_test)
            shap_model = clf.named_steps["model"]

            if hasattr(shap_model, "feature_importances_"):
                explainer = shap.TreeExplainer(shap_model)
                shap_vals = explainer.shap_values(X_test_transformed)
                shap_available = True

                # For classification, use first class or average
                if isinstance(shap_vals, list):
                    sv_matrix = np.array(shap_vals[1]) if len(shap_vals) > 1 else np.array(shap_vals[0])
                else:
                    sv_matrix = np.array(shap_vals)

                mean_shap = np.abs(sv_matrix).mean(axis=0)
                if len(mean_shap) == len(clean_names):
                    shap_pairs = sorted(
                        zip(clean_names, mean_shap.tolist()), key=lambda x: x[1], reverse=True
                    )
                    shap_values_list = [
                        {"feature": n, "mean_abs_shap": round(v, 5)} for n, v in shap_pairs[:8]
                    ]

                # Individual prediction explanation
                row_idx = req.explain_row_index if req.explain_row_index is not None else 0
                row_idx = min(row_idx, len(X_test) - 1)
                row_shap = sv_matrix[row_idx]
                row_pred = y_pred_test[row_idx] if hasattr(y_pred_test, "__getitem__") else str(y_pred_test)

                if len(row_shap) == len(clean_names):
                    row_pairs = sorted(
                        zip(clean_names, row_shap.tolist()), key=lambda x: abs(x[1]), reverse=True
                    )
                    individual_explanation = {
                        "row_index": row_idx,
                        "prediction": str(row_pred),
                        "actual": str(list(y_test)[row_idx]),
                        "top_drivers": [
                            {
                                "feature": n,
                                "shap_value": round(v, 5),
                                "direction": "⬆ increases prediction" if v > 0 else "⬇ decreases prediction",
                            }
                            for n, v in row_pairs[:6]
                        ],
                    }

                # SHAP bar chart
                if shap_values_list:
                    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                    fig2.patch.set_facecolor("#0f1322")
                    ax2.set_facecolor("#0f1322")
                    sdf = pd.DataFrame(shap_values_list)
                    colors_shap = ["#10b981" if v > 0 else "#ef4444" for v in sdf["mean_abs_shap"]]
                    ax2.barh(sdf["feature"][::-1], sdf["mean_abs_shap"][::-1], color="#a855f7", alpha=0.82, edgecolor="none")
                    ax2.set_xlabel("Mean |SHAP Value|", color="#9ca3af")
                    ax2.set_title("🔮 SHAP Feature Impact (Mean Absolute Values)", color="#f3f4f6", fontsize=12, fontweight="bold")
                    ax2.tick_params(colors="#e5e7eb")
                    for spine in ax2.spines.values():
                        spine.set_color("#374151")
                    ax2.grid(axis="x", color="#1e293b", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="png", bbox_inches="tight", facecolor=fig2.get_facecolor())
                    buf2.seek(0)
                    shap_chart_b64 = base64.b64encode(buf2.read()).decode("utf-8")
                    plt.close(fig2)

        except ImportError:
            print("SHAP not installed — skipping SHAP analysis. Install with: pip install shap")
        except Exception as shap_err:
            print(f"SHAP analysis failed ({shap_err}). Continuing without SHAP.")

        # 8. Generate narrative
        narrative = _build_explainer_narrative(
            model_name=req.model_name,
            task_type=task_type,
            train_score=train_score,
            test_score=test_score,
            top_features=top_features,
            metric_name=metric_name,
        )

        # 9. Try enriching narrative via Ollama
        try:
            ollama_prompt = (
                f"You are an expert AI/ML explainer. The model '{req.model_name}' was trained for {task_type}. "
                f"Train {metric_name}: {train_score:.1%}, Test {metric_name}: {test_score:.1%}. "
                f"Top features: {[f['feature'] for f in top_features[:5]]}. "
                f"Write 3–4 concise plain-English sentences explaining what the model learned, "
                f"which features matter most, and whether it might be overfitting. "
                f"Be specific, practical, and encouraging for a learner."
            )
            ollama_resp = requests.post(OLLAMA_URL, json={
                "model": "llama3",
                "prompt": ollama_prompt,
                "stream": False,
            }, timeout=8)
            if ollama_resp.status_code == 200:
                llm_text = ollama_resp.json().get("response", "").strip()
                if llm_text:
                    narrative += f"\n\n### 🤖 LLM-Enhanced Insight\n\n{llm_text}"
        except Exception:
            pass  # Ollama not running — narrative still fully generated above

        return {
            "success": True,
            "task_type": task_type,
            "model_name": req.model_name,
            "metric_name": metric_name,
            "train_score": round(train_score, 4),
            "test_score": round(test_score, 4),
            "extra_metrics": extra_metrics,
            "top_features": top_features,
            "feat_chart_b64": feat_chart_b64,
            "shap_available": shap_available,
            "shap_values": shap_values_list,
            "shap_chart_b64": shap_chart_b64,
            "individual_explanation": individual_explanation,
            "narrative": narrative,
        }

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Model explanation failed: {str(e)}\n{tb}")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            frontend_app = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
            runpy.run_path(str(frontend_app), run_name="__main__")
        else:
            import uvicorn

            uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)

