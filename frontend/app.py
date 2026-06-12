import streamlit as st
import requests
import os
import re
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent GUI thread crashes on Windows
from dotenv import load_dotenv

# Load configurations
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


@st.cache_resource
def _load_local_backend():
    import sys
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    backend_path = Path(backend_dir) / "main_grounded.py"
    spec = importlib.util.spec_from_file_location("local_main_grounded", backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module




def _get_course_sources():
    try:
        docs_response = requests.get(f"{BACKEND_URL}/api/rag/documents", timeout=8)
        if docs_response.status_code == 200:
            return [doc.get("source") for doc in docs_response.json().get("documents", []) if doc.get("source")]
    except Exception:
        pass

    try:
        local_backend = _load_local_backend()
        return [doc.get("source") for doc in local_backend.rag_service.list_documents() if doc.get("source")]
    except Exception:
        return []


def _generate_learning_path_payload(payload: dict):
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/learning-path", json=payload, timeout=12)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    local_backend = _load_local_backend()
    skill_input = local_backend.SkillInput(**payload)
    return local_backend.generate_learning_path(skill_input)

# Set Page Config
st.set_page_config(
    page_title="AI Data Science Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Cyberpunk Dark Mode)
st.markdown("""
<style>
    /* Main Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #090c15;
        background-image: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05), transparent 40%),
                          radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.05), transparent 40%);
        color: #f3f4f6;
    }
    
    /* Force high contrast text inside sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] small {
        color: #f3f4f6 !important;
    }
    
    /* Fix text inputs/select box labels in sidebar */
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
    }
    
    /* Style Streamlit Buttons globally to be premium and readable */
    div.stButton > button {
        background-color: #161c2d !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Glassmorphic custom cards */
    .glass-card {
        background: rgba(22, 28, 45, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .glow-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    /* Custom Chat Bubbles */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 85%;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble.user {
        background: #6366f1;
        color: white;
        border-bottom-right-radius: 4px;
        float: right;
        clear: both;
    }
    
    .chat-bubble.mentor {
        background: rgba(22, 28, 45, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f3f4f6;
        border-bottom-left-radius: 4px;
        float: left;
        clear: both;
        backdrop-filter: blur(8px);
    }
    
    .chat-container {
        width: 100%;
        margin-bottom: 20px;
        display: flow-root;
    }
    
    .bubble-meta {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 4px;
        display: block;
    }
    
    /* Suggestion Chips */
    .chip {
        display: inline-block;
        padding: 8px 14px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chip:hover {
        border-color: #6366f1;
        background: rgba(99, 102, 241, 0.08);
        color: white;
    }

    /* Learning Path screen */
    .lp-hero {
        background: linear-gradient(135deg, #0f766e 0%, #065f46 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
    }

    .lp-hero-inner {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .lp-hero-icon {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.14);
        font-size: 1.55rem;
        flex-shrink: 0;
    }

    .lp-hero-title {
        margin: 0;
        font-size: 2.05rem;
        line-height: 1.1;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        color: #f8fafc;
    }

    .lp-hero-subtitle {
        margin: 6px 0 0 0;
        color: rgba(248, 250, 252, 0.9);
        font-size: 0.98rem;
    }

    .lp-panel {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
    }

    .lp-summary-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-top: 6px;
    }

    .lp-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e5e7eb;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .lp-progress-shell {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin-top: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .lp-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #f43f5e 0%, #fb7185 55%, #fda4af 100%);
        border-radius: 999px;
    }

    .lp-week-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 2px 10px;
        margin-bottom: 12px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
    }

    .lp-week-header {
        font-weight: 700;
        color: #e5e7eb;
        font-size: 0.98rem;
    }

    .lp-week-details {
        color: #9ca3af;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 8px;
    }

    .lp-meta-title {
        color: #d1fae5;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)

# Main Page Structure
st.title("🎓 AI Data Science Mentor")
st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: -10px;'>Your personalized machine learning tutor and guided study path planner.</p>", unsafe_allow_html=True)

# Sidebar Configuration removed - Study Panel moved to Main Page Tabs


# Session state initialization for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "mentor", "content": "Hello! I am your AI Data Science Mentor. Ask me any questions about machine learning, statistical modeling, data preprocessing, or python programming! You can also upload custom study documents on the sidebar to search them during our conversation."}
    ]

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: white; font-family: \"Outfit\", sans-serif; font-weight: 800; margin-top: 0;'>🎓 AI Mentor</h2>", unsafe_allow_html=True)

nav_options = [
    "🏠 Home",
    "📅 Learning Path",
    "💻 Code Evaluator",
    "🧠 Quiz Sandbox",
    "🚀 Project Recommendations",
    "💬 Ask Mentor",
    "📚 Knowledge Base",
    "📊 Interactive Visualization",
    "🎙️ Voice Mentor",
    "🤝 Interview Trainer",
    "💻 Challenge Generator"
]

default_idx = 0
if "target_nav" in st.session_state:
    if st.session_state.target_nav in nav_options:
        default_idx = nav_options.index(st.session_state.target_nav)
        # Clear the state to avoid locking the sidebar choice
        del st.session_state.target_nav

nav_selection = st.sidebar.radio(
    "AI Mentor Navigation",
    options=nav_options,
    index=default_idx,
    label_visibility="collapsed"
)



# Sidebar About Box
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='glass-card' style='padding: 15px; border-radius: 8px;'>
    <h4 style='color: #a855f7; margin-top: 0; font-family: "Outfit", sans-serif;'>📘 About</h4>
    <p style='font-size: 0.85rem; color: #9ca3af; line-height: 1.4; margin-bottom: 0;'>
        AI Data Science Mentor helps you learn, evaluate code, generate quizzes, and get project recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

if nav_selection == "🏠 Home":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>🏠</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>AI Data Science Mentor Home</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Your centralized intelligent learning portal, code evaluator, and study planner.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ Discover Key Features")
    st.write("Browse through all available specialized mentor agents and tools:")

    carousel_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;700&display=swap');
            body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: 'Inter', sans-serif;
                color: #e5e7eb;
                overflow: hidden;
            }
            .carousel-container {
                position: relative;
                width: 100%;
                margin: 0 auto;
                overflow: hidden;
            }
            .carousel-track-container {
                width: 100%;
                overflow: hidden;
                border-radius: 16px;
            }
            .carousel-track {
                display: flex;
                transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                width: 100%;
            }
            .slide {
                min-width: 100%;
                box-sizing: border-box;
                padding: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .card {
                background: rgba(22, 28, 45, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 24px;
                width: 100%;
                max-width: 650px;
                min-height: 220px;
                text-align: center;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
                backdrop-filter: blur(12px);
                position: relative;
                overflow: hidden;
                box-sizing: border-box;
                cursor: pointer;
                transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
            }
            .card:hover {
                transform: translateY(-4px) scale(1.01);
                border-color: #6366f1;
                box-shadow: 0 15px 45px rgba(99, 102, 241, 0.25);
            }
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: linear-gradient(90deg, var(--start-color), var(--end-color));
            }
            .card-icon {
                font-size: 3.2rem;
                margin-bottom: 12px;
                display: inline-block;
            }
            .card-title {
                font-family: 'Outfit', sans-serif;
                font-size: 1.6rem;
                font-weight: 800;
                margin: 0 0 10px 0;
                background: linear-gradient(135deg, #ffffff 40%, var(--end-color) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .card-desc {
                font-size: 0.92rem;
                color: #9ca3af;
                line-height: 1.6;
                margin: 0 auto 15px auto;
                max-width: 500px;
            }
            .card-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                background: rgba(255, 255, 255, 0.06);
                color: var(--start-color);
                border: 1px solid rgba(255, 255, 255, 0.04);
            }
            .carousel-nav {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 15px;
                gap: 15px;
            }
            .nav-btn {
                background: rgba(31, 41, 55, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.08);
                color: #f3f4f6;
                font-size: 1.1rem;
                cursor: pointer;
                border-radius: 50%;
                width: 38px;
                height: 38px;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: all 0.3s ease;
                outline: none;
            }
            .nav-btn:hover {
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                border-color: transparent;
                transform: scale(1.08);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            }
            .indicators {
                display: flex;
                gap: 8px;
            }
            .indicator {
                width: 8px;
                height: 8px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .indicator.active {
                background: #6366f1;
                width: 20px;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="carousel-container">
            <div class="carousel-track-container">
                <div class="carousel-track" id="track">
                    <!-- Slide 1 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('📅 Learning Path')" style="--start-color: #10b981; --end-color: #34d399;">
                            <div class="card-icon">📅</div>
                            <div class="card-title">Learning Path Generator</div>
                            <div class="card-desc">Enter your current skills and timeline to generate a customized week-by-week learning syllabus mapped out from dataset study notes.</div>
                            <div class="card-badge">Custom Planning</div>
                        </div>
                    </div>
                    <!-- Slide 2 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('💻 Code Evaluator')" style="--start-color: #6366f1; --end-color: #818cf8;">
                            <div class="card-icon">💻</div>
                            <div class="card-title">Code Evaluator</div>
                            <div class="card-desc">Submit Python/Pandas operations to receive instant constructive memory-efficiency critiques and production-grade alternatives.</div>
                            <div class="card-badge">Code Review</div>
                        </div>
                    </div>
                    <!-- Slide 3 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('🧠 Quiz Sandbox')" style="--start-color: #8b5cf6; --end-color: #a78bfa;">
                            <div class="card-icon">🧠</div>
                            <div class="card-title">Interactive Quiz Sandbox</div>
                            <div class="card-desc">Test your core concepts in Python, ML, SQL, and Statistics using multiple choice questions with detailed explanation logs.</div>
                            <div class="card-badge">Assessment</div>
                        </div>
                    </div>
                    <!-- Slide 4 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('🚀 Project Recommendations')" style="--start-color: #ea580c; --end-color: #f97316;">
                            <div class="card-icon">🚀</div>
                            <div class="card-title">Project Recommendations</div>
                            <div class="card-desc">Get customized intermediate and advanced machine learning project blueprints matching your level, or train models in our sandbox.</div>
                            <div class="card-badge">Hands-on Sandbox</div>
                        </div>
                    </div>
                    <!-- Slide 5 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('💬 Ask Mentor')" style="--start-color: #ec4899; --end-color: #f472b6;">
                            <div class="card-icon">💬</div>
                            <div class="card-title">Ask Mentor (RAG Chat)</div>
                            <div class="card-desc">Ask queries and get precise answers grounded in your custom uploaded course notes, bypassing general knowledge hallucination.</div>
                            <div class="card-badge">Grounded Chat</div>
                        </div>
                    </div>
                    <!-- Slide 6 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('📚 Knowledge Base')" style="--start-color: #3b82f6; --end-color: #60a5fa;">
                            <div class="card-icon">📚</div>
                            <div class="card-title">Knowledge Base</div>
                            <div class="card-desc">Upload study guide notes, PDFs, and manage your vector database store. Monitor sizes, chunks, and verification reports.</div>
                            <div class="card-badge">Storage Manager</div>
                        </div>
                    </div>
                    <!-- Slide 7 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('📊 Interactive Visualization')" style="--start-color: #14b8a6; --end-color: #2dd4bf;">
                            <div class="card-icon">📊</div>
                            <div class="card-title">Interactive Visualization</div>
                            <div class="card-desc">Profile uploaded data with automated correlation heatmaps, distribution histograms, outlier boxplots, and interactive scatter matrices.</div>
                            <div class="card-badge">Exploratory EDA</div>
                        </div>
                    </div>

                    <!-- Slide 9 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('🎙️ Voice Mentor')" style="--start-color: #f43f5e; --end-color: #fb7185;">
                            <div class="card-icon">🎙️</div>
                            <div class="card-title">Voice Mentor</div>
                            <div class="card-desc">Practice technical voice interviews. Record responses directly in the UI and receive custom spoken audio feedback.</div>
                            <div class="card-badge">Audio Interviews</div>
                        </div>
                    </div>
                    <!-- Slide 10 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('🤝 Interview Trainer')" style="--start-color: #eab308; --end-color: #facc15;">
                            <div class="card-icon">🤝</div>
                            <div class="card-title">Interview Trainer</div>
                            <div class="card-desc">Engage in simulated technical interview chats covering specific concept topics, complete with grading and scorecard reports.</div>
                            <div class="card-badge">Technical Prep</div>
                        </div>
                    </div>
                    <!-- Slide 11 -->
                    <div class="slide">
                        <div class="card" onclick="selectNav('💻 Challenge Generator')" style="--start-color: #a855f7; --end-color: #c084fc;">
                            <div class="card-icon">💻</div>
                            <div class="card-title">Challenge Generator</div>
                            <div class="card-desc">Practice coding problems in a live interactive Python sandbox, executing code and verifying scripts against tests.</div>
                            <div class="card-badge">Coding Sandbox</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="carousel-nav">
                <button class="nav-btn" id="prev">◀</button>
                <div class="indicators" id="indicators"></div>
                <button class="nav-btn" id="next">▶</button>
            </div>
        </div>

        <script>
            function selectNav(navName) {
                try {
                    const topUrl = new URL(window.top.location.href);
                    topUrl.searchParams.set('nav', navName);
                    window.top.location.href = topUrl.toString();
                } catch (e) {
                    try {
                        const parentUrl = new URL(window.parent.location.href);
                        parentUrl.searchParams.set('nav', navName);
                        window.parent.location.href = parentUrl.toString();
                    } catch (e2) {
                        window.location.search = "?nav=" + encodeURIComponent(navName);
                    }
                }
            }

            const track = document.getElementById('track');
            const prev = document.getElementById('prev');
            const next = document.getElementById('next');
            const indicatorsContainer = document.getElementById('indicators');
            const slides = Array.from(track.children);
            let currentIndex = 0;
            let timer;

            // Generate dot indicators
            slides.forEach((_, idx) => {
                const ind = document.createElement('div');
                ind.classList.add('indicator');
                if (idx === 0) ind.classList.add('active');
                ind.addEventListener('click', () => {
                    moveToSlide(idx);
                    resetAutoPlay();
                });
                indicatorsContainer.appendChild(ind);
            });

            const indicators = Array.from(indicatorsContainer.children);

            function moveToSlide(index) {
                currentIndex = index;
                track.style.transform = `translateX(-${currentIndex * 100}%)`;
                indicators.forEach((ind, idx) => {
                    if (idx === currentIndex) {
                        ind.classList.add('active');
                    } else {
                        ind.classList.remove('active');
                    }
                });
            }

            function goNext() {
                currentIndex = (currentIndex + 1) % slides.length;
                moveToSlide(currentIndex);
            }

            function goPrev() {
                currentIndex = (currentIndex - 1 + slides.length) % slides.length;
                moveToSlide(currentIndex);
            }

            prev.addEventListener('click', () => {
                goPrev();
                resetAutoPlay();
            });

            next.addEventListener('click', () => {
                goNext();
                resetAutoPlay();
            });

            function startAutoPlay() {
                timer = setInterval(goNext, 3500);
            }

            function resetAutoPlay() {
                clearInterval(timer);
                startAutoPlay();
            }

            // Start
            startAutoPlay();
        </script>
    </body>
    </html>
    """
    
    st.components.v1.html(carousel_code, height=310)
    
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    st.markdown("### 🚀 Launch Features")
    st.write("Click any tool below to launch it directly:")
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        if st.button("📅 Learning Path", use_container_width=True):
            st.session_state.target_nav = "📅 Learning Path"
            st.rerun()
        if st.button("💻 Code Evaluator", use_container_width=True):
            st.session_state.target_nav = "💻 Code Evaluator"
            st.rerun()
        if st.button("🧠 Quiz Sandbox", use_container_width=True):
            st.session_state.target_nav = "🧠 Quiz Sandbox"
            st.rerun()
        if st.button("🚀 Projects & Models", use_container_width=True):
            st.session_state.target_nav = "🚀 Project Recommendations"
            st.rerun()
            
    with col_l2:
        if st.button("💬 Ask Mentor Chat", use_container_width=True):
            st.session_state.target_nav = "💬 Ask Mentor"
            st.rerun()
        if st.button("📚 Knowledge Base", use_container_width=True):
            st.session_state.target_nav = "📚 Knowledge Base"
            st.rerun()
        if st.button("📊 Exploratory EDA", use_container_width=True):
            st.session_state.target_nav = "📊 Interactive Visualization"
            st.rerun()

            
    with col_l3:
        if st.button("🎙️ Voice Mentor", use_container_width=True):
            st.session_state.target_nav = "🎙️ Voice Mentor"
            st.rerun()
        if st.button("🤝 Interview Trainer", use_container_width=True):
            st.session_state.target_nav = "🤝 Interview Trainer"
            st.rerun()
        if st.button("💻 Challenge Generator", use_container_width=True):
            st.session_state.target_nav = "💻 Challenge Generator"
            st.rerun()

elif nav_selection == "📅 Learning Path":
    if st.button("⬅️ Back to Home", key="back_home_lp"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div class='lp-hero'>
        <div class='lp-hero-inner'>
            <div class='lp-hero-icon'>📚</div>
            <div>
                <h1 class='lp-hero-title'>Learning Path Generator</h1>
                <p class='lp-hero-subtitle'>Design a course-specific week-by-week study plan from your selected material and current skill level.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if "course_sources" not in st.session_state:
        st.session_state.course_sources = _get_course_sources()

    if not st.session_state.course_sources:
        st.warning("Upload course material first so the learning path can be generated from that specific course content.")
    
    with st.container():
        st.markdown("<div class='lp-panel'>", unsafe_allow_html=True)
        col_input, col_slider = st.columns([2.2, 1])
        with col_input:
            skills_input_str = st.text_input(
                "Enter your current skills (comma-separated):",
                value="Java",
                placeholder="e.g. Python, SQL",
                key="skills_input_str"
            )
        with col_slider:
            weeks_count = st.slider("How many weeks do you have?", min_value=1, max_value=24, value=4, key="weeks_count")
            st.markdown(f"<div style='margin-top: 6px; color: #cbd5e1; font-size: 0.85rem;'>Slider set to <strong>{weeks_count}</strong> weeks</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    selected_course_source = st.session_state.course_sources[0] if st.session_state.course_sources else None
        
    if st.button("Generate Learning Path ➔", width="stretch", key="study_gen_btn"):
        selected_skills = [s.strip() for s in skills_input_str.split(",") if s.strip()]
        if not selected_skills:
            st.warning("Please enter at least one skill.")
        else:
            is_python_learning_path = any("python" in skill.lower() or "sql" in skill.lower() for skill in selected_skills)
            is_java_learning_path = any("java" in skill.lower() for skill in selected_skills)
            if is_java_learning_path or is_python_learning_path:
                weeks_count = 4
            with st.spinner("Analyzing curriculum..."):
                try:
                    payload = {"skills": selected_skills}
                    if selected_course_source:
                        payload["course_source"] = selected_course_source

                    response_data = _generate_learning_path_payload(payload)
                    raw_path = response_data.get("generated_path", [])
                    if is_java_learning_path:
                        raw_path = [
                            {
                                "week": 1,
                                "topic": "Java Fundamentals & Core Programming Concepts",
                                "details": "Learn Java syntax, variables, data types, operators, control flow, methods, and input/output basics.",
                            },
                            {
                                "week": 2,
                                "topic": "Object-Oriented Programming (OOP) & Class Design",
                                "details": "Practice classes, objects, constructors, inheritance, encapsulation, polymorphism, abstraction, and interfaces.",
                            },
                            {
                                "week": 3,
                                "topic": "Collections Framework & Exception Handling",
                                "details": "Study ArrayList, LinkedList, HashMap, HashSet, generics, iterators, try-catch blocks, and custom exceptions.",
                            },
                            {
                                "week": 4,
                                "topic": "Advanced Java Development & Project Implementation",
                                "details": "Work with file handling, streams, lambda expressions, and a small Java project that combines everything learned.",
                            },
                        ]
                    # Adapt dynamically to week slider selection
                    st.session_state.learning_path = raw_path[:weeks_count]
                    st.session_state.mapped_path = response_data.get("mapped", False)
                    st.session_state.selected_course_source = response_data.get("course_source")
                    if is_java_learning_path:
                        st.session_state.learning_path_mode = "java"
                    elif is_python_learning_path:
                        st.session_state.learning_path_mode = "python"
                    else:
                        st.session_state.learning_path_mode = "standard"
                    # Clear old checkbox states
                    for k in list(st.session_state.keys()):
                        if k.startswith("week_"):
                            del st.session_state[k]
                    st.success("Study plan created successfully!")
                except Exception as e:
                    st.error(f"Could not connect to backend server: {e}")
                    
    if "learning_path" in st.session_state:
        total_weeks = len(st.session_state.learning_path)
        completed_weeks = sum(1 for plan in st.session_state.learning_path if st.session_state.get(f"week_{plan.get('week')}", False))
        progress_percentage = (completed_weeks / total_weeks) if total_weeks > 0 else 0.0

        st.markdown("<div class='lp-panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='lp-meta-title'>{total_weeks} weeks planned</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='lp-summary-row'>"
            f"<span class='lp-pill'>✅ {completed_weeks} completed</span>"
            f"<span class='lp-pill'>📅 {total_weeks} total weeks</span>"
            f"<span class='lp-pill'>{'✨ Verified curriculum mapping' if st.session_state.mapped_path else '🤖 AI generated curriculum'}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.get("selected_course_source"):
            st.markdown(f"<div style='margin-top:10px; color:#d1d5db; font-size:0.9rem;'>📘 Source course: <strong>{st.session_state.selected_course_source}</strong></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='lp-progress-shell'><div class='lp-progress-fill' style='width: {int(progress_percentage * 100)}%;'></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:8px; color:#fca5a5; font-size:0.84rem; font-weight:700;'>{completed_weeks} of {total_weeks} weeks completed ({int(progress_percentage * 100)}%)</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        completed_weeks = sum(1 for plan in st.session_state.learning_path if st.session_state.get(f"week_{plan.get('week')}", False))
        learning_path_mode = st.session_state.get("learning_path_mode", "standard")
        for plan in st.session_state.learning_path:
            week_num = plan.get("week")
            topic = plan.get("topic")
            details = plan.get("details")
            subtopics = plan.get("subtopics", [])
            
            key = f"week_{week_num}"
            is_done = st.session_state.get(key, False)
            with st.expander(f"Week {week_num}: {topic}", expanded=(week_num == 1)):
                st.markdown("<div class='lp-week-card'>", unsafe_allow_html=True)
                st.checkbox("Mark this week complete", key=key)
                st.markdown(
                    f"<div class='lp-week-header'>{'✅ Completed' if is_done else '📌 In Progress'}</div>",
                    unsafe_allow_html=True,
                )
                if subtopics:
                    st.markdown("<div style='margin-top: 8px; color:#d1d5db; font-size:0.9rem; font-weight:600;'>Subtopics:</div>", unsafe_allow_html=True)
                    st.markdown("\n".join([f"- {item}" for item in subtopics]), unsafe_allow_html=True)
                elif learning_path_mode == "java":
                    st.markdown(
                        f"<div class='lp-week-details'><strong>Java Week {week_num}:</strong> {details}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"<div class='lp-week-details'>{details}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


elif nav_selection == "💻 Code Evaluator":
    if st.button("⬅️ Back to Home", key="back_home_ce"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>💻</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Code Assignment Evaluator</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Receive detailed code reviews, memory efficiency critiques, and production-grade optimizations.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.subheader("💻 Submit Code")
        submitted_code = st.text_area(
            "Paste your Python or Pandas code here:",
            height=200,
            placeholder="e.g. df.dropna()",
            key="eval_submitted_code"
        )
        assignment_context = st.text_input(
            "Assignment Context or Objective (Optional):",
            placeholder="e.g. Remove rows with null target variable",
            key="eval_assignment_context"
        )
        if st.button("Evaluate Code ➔", width="stretch", key="eval_submit_btn"):
            if not submitted_code.strip():
                st.warning("Please submit some code before evaluating.")
            else:
                with st.spinner("Analyzing code for correctness and memory efficiency..."):
                    try:
                        payload = {
                            "submitted_code": submitted_code,
                            "assignment_context": assignment_context if assignment_context.strip() else None
                        }
                        res = requests.post(f"{BACKEND_URL}/api/v1/evaluate", json=payload)
                        if res.status_code == 200:
                            st.session_state.eval_result = res.json()
                            st.success("Assignment evaluated successfully!")
                        else:
                            st.error(f"Evaluation API returned status code {res.status_code}.")
                    except Exception as eval_err:
                        st.error(f"Could not connect to backend evaluation server: {eval_err}")

        
    with col_eval2:
        if "eval_result" in st.session_state and st.session_state.eval_result:
            result = st.session_state.eval_result
            status = result.get("status", "Needs Improvement")
            
            if status == "Correct":
                st.success(f"#### Status: {status} ✅")
            elif status == "Incorrect":
                st.error(f"#### Status: {status} ❌")
            else:
                st.warning(f"#### Status: {status} ⚠️")
                
            st.markdown("##### 📋 Review Summary")
            st.write(result.get("review_summary", ""))
            
            st.markdown("##### 🔍 Line-by-Line Feedback")
            st.markdown(result.get("line_by_line_feedback", ""))
            
            st.markdown("##### 💡 Optimized Production-Grade Alternative")
            st.markdown(result.get("optimized_alternative_code", ""))
        else:
            st.info("💡 Submit your code on the left to see the review and optimization suggestions here.")

elif nav_selection == "🧠 Quiz Sandbox":
    if st.button("⬅️ Back to Home", key="back_home_qs"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <style>
        /* Target the button immediately following a quiz-option-marker */
        div:has(> div.stMarkdown > div.quiz-option-marker) + div.element-container div.stButton > button {
            background-color: #2563eb !important;
            background-image: none !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 14px 20px !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            text-align: center !important;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100% !important;
        }
        div:has(> div.stMarkdown > div.quiz-option-marker) + div.element-container div.stButton > button:hover {
            background-color: #1d4ed8 !important;
            background-image: none !important;
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.35) !important;
            transform: translateY(-1px) !important;
        }
        div:has(> div.stMarkdown > div.quiz-option-marker) + div.element-container div.stButton > button:active {
            transform: translateY(1px) !important;
        }
        
        /* Target the Generate Quiz button specifically */
        div:has(> div.stMarkdown > div.quiz-gen-marker) + div.element-container div.stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 20px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            text-align: center !important;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100% !important;
        }
        div:has(> div.stMarkdown > div.quiz-gen-marker) + div.element-container div.stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
            transform: translateY(-1px) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>🧠</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Quiz Sandbox</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Challenge yourself on core data science and machine learning topics.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state variables for Quiz
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = "setup"  # setup, active, results
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_current_idx" not in st.session_state:
        st.session_state.quiz_current_idx = 0
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = {}

    q_state = st.session_state.quiz_state

    # 1. SETUP STATE
    if q_state == "setup":
        st.subheader("⚙️ Select Quiz Topic")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            quiz_topic = st.selectbox(
                "Choose a domain topic:",
                options=[
                    "Python Basics",
                    "Pandas & DataFrames",
                    "Machine Learning",
                    "SQL & Databases",
                    "Statistics & Probability",
                    "Deep Learning & Neural Networks",
                    "Natural Language Processing (NLP)",
                    "Data Visualization",
                    "Big Data & PySpark"
                ],
                key="quiz_topic"
            )
            st.markdown("<div class='quiz-gen-marker'></div>", unsafe_allow_html=True)
            if st.button("Generate Quiz ⚡", key="quiz_gen_btn", use_container_width=True):
                with st.spinner("Building interactive quiz..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/api/v1/quiz", json={"topic": quiz_topic})
                        if res.status_code == 200:
                            st.session_state.quiz_questions = res.json()
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = {}
                            st.session_state.quiz_current_idx = 0
                            st.session_state.quiz_state = "active"
                            st.success("Quiz loaded!")
                            st.rerun()
                        else:
                            st.error("Quiz API returned an error.")
                    except Exception as e:
                        st.error(f"Could not contact quiz server: {e}")
        with col_q2:
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px;'>
                <h4 style='color:#a78bfa; margin-top:0; font-family:"Outfit",sans-serif;'>Sandbox Policy:</h4>
                <ul style='color:#9ca3af; font-size:0.88rem; padding-left:20px; line-height:1.5; margin-bottom:0;'>
                    <li>Multiple choice questions are generated dynamically.</li>
                    <li>Questions display one-by-one to help focus.</li>
                    <li>Receive explanations instantly after submitting each answer.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # 2. ACTIVE STATE
    elif q_state == "active":
        questions = st.session_state.quiz_questions
        idx = st.session_state.quiz_current_idx

        if not questions:
            st.session_state.quiz_state = "setup"
            st.rerun()

        # Quiz navigation header
        c_nav1, c_nav2 = st.columns([4, 1])
        with c_nav2:
            if st.button("Quit Quiz 🛑", key="quit_quiz_btn"):
                st.session_state.quiz_state = "setup"
                st.rerun()
        
        q = questions[idx]
        question_text = q.get("question")
        options = q.get("options", [])
        correct_ans = q.get("correct_answer")
        
        st.markdown(f"<span style='color:#9ca3af; font-size:0.9rem;'>Question {idx+1} of {len(questions)}</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background-color:#faf8f0; border-left:6px solid #2563eb; border-top:1px solid rgba(0,0,0,0.05); border-right:1px solid rgba(0,0,0,0.05); border-bottom:1px solid rgba(0,0,0,0.05); border-radius:8px; padding:25px; margin-top:10px; margin-bottom:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.08), 0 2px 4px -1px rgba(0,0,0,0.04);'>
            <h3 style='color:#1e293b; font-family:"Outfit",sans-serif; margin:0; line-height:1.45;'>📌 {question_text} (Q{idx+1})</h3>
        </div>
        """, unsafe_allow_html=True)

        is_answered = st.session_state.quiz_submitted.get(idx, False)

        prefixes = ["A) ", "B) ", "C) ", "D) ", "E) "]
        
        if not is_answered:
            st.write("Choose your answer:")
            for i, opt in enumerate(options):
                prefix = prefixes[i] if i < len(prefixes) else ""
                opt_label = f"{prefix}{opt}"
                
                # Styled buttons (vertical full width)
                st.markdown("<div class='quiz-option-marker'></div>", unsafe_allow_html=True)
                if st.button(opt_label, key=f"quiz_opt_{idx}_{i}", use_container_width=True):
                    st.session_state.quiz_answers[idx] = opt
                    st.session_state.quiz_submitted[idx] = True
                    st.rerun()
        else:
            user_ans = st.session_state.quiz_answers.get(idx)
            
            # Show correctness result
            if user_ans == correct_ans:
                st.success(f"✅ Correct! Your answer: {user_ans}")
            else:
                st.error(f"❌ Incorrect! Your answer: {user_ans} | Correct answer: {correct_ans}")
            
            # Display disabled options showing correct/incorrect
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            for i, opt in enumerate(options):
                prefix = prefixes[i] if i < len(prefixes) else ""
                opt_label = f"{prefix}{opt}"
                
                if opt == correct_ans:
                    st.markdown(f"""
                    <div style='background:rgba(16,185,129,0.15); border:1px solid #10b981; border-radius:8px; padding:12px 18px; margin-bottom:8px; font-weight:600; color:#10b981;'>
                        {opt_label} (Correct Answer)
                    </div>
                    """, unsafe_allow_html=True)
                elif opt == user_ans:
                    st.markdown(f"""
                    <div style='background:rgba(239,68,68,0.15); border:1px solid #ef4444; border-radius:8px; padding:12px 18px; margin-bottom:8px; font-weight:600; color:#ef4444;'>
                        {opt_label} (Your Answer - Incorrect)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px 18px; margin-bottom:8px; color:#9ca3af;'>
                        {opt_label}
                    </div>
                    """, unsafe_allow_html=True)

            # Display explanation card
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:20px; margin-top:20px; margin-bottom:20px;'>
                <strong style='color:#a78bfa;'>💡 Explanation:</strong>
                <p style='color:#d1d5db; font-size:0.92rem; margin:6px 0 0 0; line-height:1.55;'>{q.get("explanation")}</p>
            </div>
            """, unsafe_allow_html=True)

            # Navigation buttons
            is_last = (idx + 1) >= len(questions)
            btn_label = "View Final Results ➔" if is_last else "Next Question ➔"
            
            if st.button(btn_label, key="quiz_next_btn", use_container_width=True):
                if is_last:
                    st.session_state.quiz_state = "results"
                else:
                    st.session_state.quiz_current_idx += 1
                st.rerun()

    # 3. RESULTS STATE
    elif q_state == "results":
        questions = st.session_state.quiz_questions
        answers = st.session_state.quiz_answers
        
        correct_count = 0
        for idx, q in enumerate(questions):
            if answers.get(idx) == q.get("correct_answer"):
                correct_count += 1
                
        accuracy = int(correct_count / len(questions) * 100) if questions else 0
        
        st.subheader("🏁 Quiz Complete!")
        
        # Results gauge
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #1e1b4b, #311042); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:35px; text-align:center; margin-bottom:25px;'>
            <div style='font-size:0.9rem; color:#a78bfa; text-transform:uppercase; font-weight:700;'>Final Accuracy Score</div>
            <div style='font-size:4.5rem; font-weight:800; color:#10b981; margin-top:10px;'>{correct_count} / {len(questions)}</div>
            <div style='font-size:1.2rem; font-weight:600; color:#e5e7eb; margin-top:5px;'>{accuracy}% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Review Questions:")
        
        # Display list of questions for review
        for idx, q in enumerate(questions):
            user_ans = answers.get(idx)
            correct_ans = q.get("correct_answer")
            is_correct = user_ans == correct_ans
            
            status_badge = "<span style='color:#10b981; font-weight:700;'>✅ Correct</span>" if is_correct else f"<span style='color:#ef4444; font-weight:700;'>❌ Incorrect (Correct: {correct_ans})</span>"
            
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:18px; margin-bottom:12px;'>
                <div style='font-weight:700; color:#e5e7eb; font-size:0.95rem; margin-bottom:6px;'>Q{idx+1}: {q.get("question")}</div>
                <div style='font-size:0.87rem; color:#9ca3af;'>Your Answer: {user_ans if user_ans else 'Unanswered'} | {status_badge}</div>
                <div style='font-size:0.82rem; color:#6b7280; font-style:italic; margin-top:6px;'>{q.get("explanation")}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("Start New Quiz ⚡", key="restart_quiz_btn", use_container_width=True):
            st.session_state.quiz_state = "setup"
            st.session_state.quiz_questions = []
            st.session_state.quiz_current_idx = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = {}
            st.rerun()

elif nav_selection == "🚀 Project Recommendations":
    if st.button("⬅️ Back to Home", key="back_home_pr"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div style='background: linear-gradient(135deg, #7c2d12 0%, #c2410c 50%, #ea580c 100%); padding: 40px; border-radius: 16px; margin-bottom: 30px; color: white;'>
        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 3.5rem; margin-right: 25px;'>🚀</span>
            <div>
                <h1 style='margin: 0; font-size: 2.4rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Project Recommendations</h1>
                <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Discover custom, high-quality project blueprints tailored to your skills and goals.</p>
            </div>
        </div>
        <div style='display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;'>
            <div style='background: rgba(255,255,255,0.12); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.4rem; font-weight: 800;'>50+</span><br/>
                <span style='font-size: 0.8rem; opacity: 0.85;'>Project Templates</span>
            </div>
            <div style='background: rgba(255,255,255,0.12); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.4rem; font-weight: 800;'>8</span><br/>
                <span style='font-size: 0.8rem; opacity: 0.85;'>Topic Domains</span>
            </div>
            <div style='background: rgba(255,255,255,0.12); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.4rem; font-weight: 800;'>3</span><br/>
                <span style='font-size: 0.8rem; opacity: 0.85;'>Skill Levels</span>
            </div>
            <div style='background: rgba(255,255,255,0.12); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.4rem; font-weight: 800;'>AI</span><br/>
                <span style='font-size: 0.8rem; opacity: 0.85;'>Powered Advisor</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Featured / Trending Projects Showcase ──────────────────────────────
    st.markdown("### 🔥 Featured & Trending Projects")
    st.markdown("<p style='color:#9ca3af; font-size:0.9rem; margin-top:-10px; margin-bottom:16px;'>Hand-picked high-impact projects loved by the DS community</p>", unsafe_allow_html=True)

    featured = [
        {
            "emoji": "📈",
            "title": "Stock Price Predictor",
            "tags": ["Time Series", "LSTM", "Finance"],
            "level": "Intermediate",
            "level_color": "#f59e0b",
            "duration": "3–4 weeks",
            "tech": ["Python", "TensorFlow", "Yahoo Finance API", "Streamlit"],
            "desc": "Build an LSTM-based neural network that forecasts stock closing prices using historical OHLCV data. Deploy as an interactive Streamlit dashboard.",
            "stars": "⭐ 4.8"
        },
        {
            "emoji": "🛒",
            "title": "E-Commerce Recommender",
            "tags": ["Recommender Systems", "Collaborative Filtering"],
            "level": "Advanced",
            "level_color": "#ef4444",
            "duration": "4–6 weeks",
            "tech": ["Python", "Surprise", "FastAPI", "React"],
            "desc": "Design a collaborative-filtering recommendation engine with matrix factorization (SVD). Serve via FastAPI and measure precision@K and recall@K.",
            "stars": "⭐ 4.9"
        },
        {
            "emoji": "🌿",
            "title": "Plant Disease Classifier",
            "tags": ["Computer Vision", "CNN", "Transfer Learning"],
            "level": "Intermediate",
            "level_color": "#f59e0b",
            "duration": "2–3 weeks",
            "tech": ["Python", "PyTorch", "EfficientNet", "Gradio"],
            "desc": "Fine-tune EfficientNet on PlantVillage dataset to classify 38 plant diseases from leaf images. Deploy via Gradio for demo-ready web inference.",
            "stars": "⭐ 4.7"
        },
        {
            "emoji": "💬",
            "title": "Sentiment Analysis API",
            "tags": ["NLP", "Transformers", "REST API"],
            "level": "Beginner",
            "level_color": "#10b981",
            "duration": "1–2 weeks",
            "tech": ["Python", "HuggingFace", "FastAPI", "Docker"],
            "desc": "Fine-tune a BERT-based model on movie reviews for sentiment classification. Wrap it in a FastAPI microservice and containerize with Docker.",
            "stars": "⭐ 4.6"
        },
        {
            "emoji": "🏠",
            "title": "House Price Predictor",
            "tags": ["Regression", "EDA", "Feature Engineering"],
            "level": "Beginner",
            "level_color": "#10b981",
            "duration": "1–2 weeks",
            "tech": ["Python", "scikit-learn", "XGBoost", "Pandas"],
            "desc": "Explore the Ames Housing dataset, engineer powerful features, and build an XGBoost regressor. Analyze feature importances and interpret predictions with SHAP.",
            "stars": "⭐ 4.5"
        },
        {
            "emoji": "🔍",
            "title": "Customer Churn Detector",
            "tags": ["Classification", "Imbalanced Data", "Business ML"],
            "level": "Beginner",
            "level_color": "#10b981",
            "duration": "1–2 weeks",
            "tech": ["Python", "scikit-learn", "SMOTE", "Matplotlib"],
            "desc": "Predict telecom customer churn with Random Forest + SMOTE oversampling. Generate an executive-level report with ROC curves and business impact analysis.",
            "stars": "⭐ 4.4"
        },
    ]

    fc1, fc2, fc3 = st.columns(3)
    feat_cols = [fc1, fc2, fc3]
    for i, proj in enumerate(featured):
        level_badge_color = proj["level_color"]
        tags_html = " ".join([
            f"<span style='background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:2px 10px; font-size:0.72rem; color:#d1d5db; margin-right:4px;'>{t}</span>"
            for t in proj["tags"]
        ])
        tech_html = " ".join([
            f"<span style='background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); border-radius:6px; padding:2px 8px; font-size:0.72rem; color:#a5b4fc; margin-right:3px; display:inline-block; margin-bottom:3px;'>{t}</span>"
            for t in proj["tech"]
        ])
        card_html = f"""
        <div style='background: linear-gradient(145deg, #111827, #161c2d); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 22px; height: 100%; position: relative; overflow: hidden;'>
            <div style='position:absolute; top:0; right:0; width:80px; height:80px; background:radial-gradient(circle, rgba(234,88,12,0.15), transparent 70%); border-radius: 0 14px 0 0;'></div>
            <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;'>
                <span style='font-size:2rem;'>{proj["emoji"]}</span>
                <span style='font-size:0.75rem; color:{level_badge_color}; background:rgba(0,0,0,0.3); border:1px solid {level_badge_color}; border-radius:20px; padding:3px 10px; font-weight:700;'>{proj["level"]}</span>
            </div>
            <h4 style='color:#f3f4f6; margin:0 0 6px 0; font-family:"Outfit",sans-serif; font-size:1.05rem; font-weight:700;'>{proj["title"]}</h4>
            <div style='margin-bottom:8px;'>{tags_html}</div>
            <p style='color:#9ca3af; font-size:0.82rem; line-height:1.5; margin-bottom:10px;'>{proj["desc"]}</p>
            <div style='margin-bottom:10px; flex-wrap:wrap;'>{tech_html}</div>
            <div style='display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; margin-top:auto;'>
                <span style='font-size:0.75rem; color:#6b7280;'>⏱ {proj["duration"]}</span>
                <span style='font-size:0.75rem; color:#fbbf24;'>{proj["stars"]}</span>
            </div>
        </div>
        """
        with feat_cols[i % 3]:
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Skills Profiler ─────────────────────────────────────────────────────
    st.markdown("### 🧑‍💻 Your Skills Profile")
    st.markdown("<p style='color:#9ca3af; font-size:0.9rem; margin-top:-10px; margin-bottom:16px;'>Tell us about your background so we can tailor recommendations perfectly</p>", unsafe_allow_html=True)

    prof_col1, prof_col2, prof_col3 = st.columns(3)
    with prof_col1:
        user_skills = st.multiselect(
            "🛠 Your Current Skills:",
            options=["Python", "SQL", "R", "Pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch",
                     "Keras", "Spark", "Tableau", "Power BI", "Excel", "Git", "Docker", "FastAPI", "Flask",
                     "HuggingFace", "OpenCV", "NLTK", "SpaCy", "XGBoost", "LightGBM", "Matplotlib", "Seaborn"],
            default=["Python", "Pandas"],
            key="proj_user_skills"
        )
    with prof_col2:
        user_goal = st.selectbox(
            "🎯 Your Primary Goal:",
            options=[
                "Build a portfolio for jobs",
                "Win a Kaggle competition",
                "Learn a specific algorithm",
                "Build a deployable product",
                "Academic / Research project",
                "Freelance / Client project"
            ],
            key="proj_user_goal"
        )
    with prof_col3:
        user_time = st.selectbox(
            "⏰ Time Available Per Week:",
            options=["< 5 hours", "5–10 hours", "10–20 hours", "20+ hours"],
            key="proj_user_time"
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Generator Controls ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Generate Custom Project Blueprint")

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        diff_level = st.selectbox(
            "Skill Level:",
            options=["Beginner", "Intermediate", "Advanced"],
            key="project_diff"
        )
    with ctrl2:
        proj_topic = st.selectbox(
            "Domain / Topic:",
            options=[
                "Exploratory Data Analysis",
                "Supervised Learning",
                "Unsupervised Learning & Clustering",
                "Deep Learning",
                "Natural Language Processing (NLP)",
                "Computer Vision",
                "Time Series Analysis",
                "FastAPI / Microservices",
                "Reinforcement Learning",
                "MLOps & Model Deployment"
            ],
            key="project_topic"
        )
    with ctrl3:
        proj_type = st.selectbox(
            "Project Type:",
            options=["End-to-End Pipeline", "Research / Experiment", "API / Microservice", "Dashboard / Visualization", "Kaggle Competition Style"],
            key="project_type"
        )
    with ctrl4:
        num_projects = st.slider("# of Projects:", min_value=1, max_value=5, value=3, key="num_proj")

    skills_str = ", ".join(user_skills) if user_skills else "Python"
    enriched_prompt = f"Skills: {skills_str}. Goal: {user_goal}. Time: {user_time}. Level: {diff_level}. Topic: {proj_topic}. Type: {proj_type}. Count: {num_projects}."

    if st.button("🚀 Generate Project Blueprints", width="stretch", key="project_gen_btn"):
        with st.spinner("🤖 AI Advisor is crafting your personalized blueprints..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/api/v1/projects",
                    json={"difficulty": diff_level, "topic": proj_topic, "context": enriched_prompt}
                )
                if res.status_code == 200:
                    st.session_state.project_blueprints = res.json()
                    st.session_state.proj_meta = {
                        "level": diff_level, "topic": proj_topic,
                        "skills": user_skills, "goal": user_goal
                    }
                    st.success(f"✅ Generated {len(res.json())} custom project blueprints!")
                else:
                    st.error(f"Project API returned status {res.status_code}.")
            except Exception as e:
                st.error(f"Could not connect to project adviser: {e}")

    # ── Blueprint Results ────────────────────────────────────────────────────
    if "project_blueprints" in st.session_state and st.session_state.project_blueprints:
        blueprints = st.session_state.project_blueprints
        meta = st.session_state.get("proj_meta", {})

        st.markdown("---")
        st.markdown(f"### 📋 Your Personalized Blueprints — {meta.get('level','')}: {meta.get('topic','')}")

        # Stats bar
        total_bp = len(blueprints)
        avg_deliverables = round(sum(len(p.get("key_deliverables", [])) for p in blueprints) / max(total_bp, 1), 1)
        total_tech = len(set(t for p in blueprints for t in p.get("tech_stack", [])))
        stat1, stat2, stat3, stat4 = st.columns(4)
        for col, label, val, icon in [
            (stat1, "Blueprints", total_bp, "🗂"),
            (stat2, "Avg. Deliverables", avg_deliverables, "✅"),
            (stat3, "Unique Tech", total_tech, "🛠"),
            (stat4, "Skill Level", meta.get("level","–"), "📊"),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#111827,#1f2937); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:16px; text-align:center;'>
                    <div style='font-size:1.8rem;'>{icon}</div>
                    <div style='font-size:1.4rem; font-weight:800; color:#f3f4f6; font-family:"Outfit",sans-serif;'>{val}</div>
                    <div style='font-size:0.75rem; color:#6b7280;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        difficulty_colors = {"Beginner": "#10b981", "Intermediate": "#f59e0b", "Advanced": "#ef4444"}

        for idx, p in enumerate(blueprints):
            title = p.get("title", f"Project {idx+1}")
            description = p.get("description", "")
            difficulty = p.get("difficulty", meta.get("level", "Intermediate"))
            tech_stack = p.get("tech_stack", [])
            deliverables = p.get("key_deliverables", [])
            d_color = difficulty_colors.get(difficulty, "#6366f1")

            tech_pills = " ".join([
                f"<span style='background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); border-radius:6px; padding:3px 10px; font-size:0.75rem; color:#a5b4fc; margin-right:4px; display:inline-block; margin-bottom:4px;'>{t}</span>"
                for t in tech_stack
            ])
            deliverables_items = "".join([
                f"<li style='color:#d1d5db; font-size:0.85rem; margin-bottom:4px; list-style:none; padding-left:0;'>✅ {d}</li>"
                for d in deliverables
            ])
            num_badge = f"<span style='background:rgba(234,88,12,0.2); color:#fb923c; border:1px solid rgba(234,88,12,0.4); border-radius:20px; padding:2px 10px; font-size:0.7rem; font-weight:700;'>#{idx+1}</span>"
            level_badge = f"<span style='background:rgba(0,0,0,0.3); color:{d_color}; border:1px solid {d_color}; border-radius:20px; padding:3px 12px; font-size:0.72rem; font-weight:700;'>{difficulty}</span>"

            bp_html = f"""
            <div style='background:linear-gradient(145deg,#0f172a,#1a2035); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:28px; margin-bottom:18px; position:relative; overflow:hidden;'>
                <div style='position:absolute; top:0; right:0; width:160px; height:160px; background:radial-gradient(circle, rgba(234,88,12,0.07), transparent 70%); pointer-events:none;'></div>
                <div style='display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:14px;'>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        {num_badge}
                        <h3 style='margin:0; font-size:1.2rem; font-weight:800; color:#f3f4f6; font-family:"Outfit",sans-serif;'>🏆 {title}</h3>
                    </div>
                    {level_badge}
                </div>
                <p style='color:#9ca3af; font-size:0.9rem; line-height:1.6; margin-bottom:16px;'>{description}</p>
                <div style='margin-bottom:14px;'>
                    <span style='font-size:0.78rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Tech Stack</span><br/>
                    <div style='margin-top:6px;'>{tech_pills}</div>
                </div>
                <div>
                    <span style='font-size:0.78rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Key Deliverables</span>
                    <ul style='margin-top:8px; padding-left:0;'>{deliverables_items}</ul>
                </div>
            </div>
            """
            st.markdown(bp_html, unsafe_allow_html=True)

        # ── Next Steps CTA ────────────────────────────────────────────────
        st.markdown("""
        <div style='background:linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.12)); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:24px; text-align:center; margin-top:10px;'>
            <h3 style='color:#a5b4fc; margin:0 0 8px 0; font-family:"Outfit",sans-serif;'>🎯 Ready to Build?</h3>
            <p style='color:#9ca3af; font-size:0.9rem; margin-bottom:0;'>
                Head to the <strong style='color:#e5e7eb;'>💬 Ask Mentor</strong> page to get step-by-step code guidance for any blueprint above, 
                or the <strong style='color:#e5e7eb;'>📊 Interactive Visualization</strong> to prototype your charts right inside this app.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Empty state ─────────────────────────────────────────────────────
        st.markdown("""
        <div style='background:linear-gradient(145deg,#111827,#1a2035); border:1px dashed rgba(234,88,12,0.3); border-radius:16px; padding:40px; text-align:center; margin-top:10px;'>
            <div style='font-size:3.5rem; margin-bottom:12px;'>🗺️</div>
            <h3 style='color:#f3f4f6; font-family:"Outfit",sans-serif; margin-bottom:8px;'>No Blueprints Yet</h3>
            <p style='color:#6b7280; font-size:0.9rem; max-width:460px; margin:0 auto;'>
                Fill in your skills profile above, choose a domain and difficulty, then click 
                <strong style='color:#ea580c;'>Generate Project Blueprints</strong> to get AI-crafted, 
                portfolio-ready project ideas customized just for you.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif nav_selection == "💬 Ask Mentor":
    if st.button("⬅️ Back to Home", key="back_home_am"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>💬</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Ask AI Mentor</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Ask me any questions about data science, coding, statistics, and machine learning models.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    chat_placeholder = st.container()

    with chat_placeholder:
        for msg in st.session_state.messages:
            align_class = "user" if msg["role"] == "user" else "mentor"
            icon = "👤" if msg["role"] == "user" else "🤖"
            
            content_html = msg['content'].replace(chr(10), '<br/>')
            # Format status lines as HTML badges
            content_html = content_html.replace(
                "❌ Not Found in Knowledge Base",
                "<span style='background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.4); border-radius:20px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; margin:6px 0;'>❌ Not Found in Knowledge Base</span>"
            )
            content_html = re.sub(
                r"✅ Knowledge Base Answer(?: \(Source: (.*?)\))?",
                lambda m: f"<span style='background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.4); border-radius:20px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; margin:6px 0;'>✅ Document-Grounded {f'(Source: {m.group(1)})' if m.group(1) else ''}</span>",
                content_html
            )
            
            bubble_html = f"""
            <div class="chat-container">
                <div class="chat-bubble {align_class}">
                    <strong>{icon} {align_class.capitalize()}:</strong><br/>
                    {content_html}
                </div>
            </div>
            """
            st.markdown(bubble_html, unsafe_allow_html=True)
            
            if msg["role"] == "mentor" and msg.get("context"):
                with st.container():
                    source_names = list(set([c['source'] for c in msg['context']]))
                    st.markdown(f"<div style='margin-left: 20px; margin-bottom: 8px;'>", unsafe_allow_html=True)
                    with st.expander("💡 Grounded in Course Material", expanded=False):
                        for chunk in msg["context"]:
                            st.markdown(f"**Document**: `{chunk['source']}`")
                            st.markdown(f"_{chunk['text']}_")
                    st.markdown(f"</div>", unsafe_allow_html=True)

    st.markdown("##### Quick Mentor Prompts:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Explain Overfitting", width="stretch"):
            st.session_state.temp_prompt = "Explain overfitting and regularization techniques."
    with col2:
        if st.button("📈 Linear Regression Assumptions", width="stretch"):
            st.session_state.temp_prompt = "What are the assumptions of Linear Regression?"
    with col3:
        if st.button("🧠 Explain Neural Networks", width="stretch"):
            st.session_state.temp_prompt = "What is backpropagation in neural networks?"

    # Prompt input inside the Ask Mentor screen
    user_query = st.chat_input("Ask your mentor a question...")

    if "temp_prompt" in st.session_state and st.session_state.temp_prompt:
        user_query = st.session_state.temp_prompt
        st.session_state.temp_prompt = ""

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()

    # Processing the last user query if exists
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_query = st.session_state.messages[-1]["content"]
        
        with st.spinner("AI Mentor is thinking..."):
            try:
                response = requests.post(f"{BACKEND_URL}/api/chat", json={"message": last_query})
                if response.status_code == 200:
                    reply = response.json().get("reply", "")
                    context = response.json().get("context", [])
                    st.session_state.messages.append({
                        "role": "mentor", 
                        "content": reply,
                        "context": context
                    })
                else:
                    st.session_state.messages.append({
                        "role": "mentor", 
                        "content": "I couldn't contact the mentor server. Please verify the FastAPI backend is running on port 8000."
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "mentor",
                    "content": f"Connection error: Could not reach the API backend on port 8000. Error: {e}"
                })
        st.rerun()

elif nav_selection == "📚 Knowledge Base":
    if st.button("⬅️ Back to Home", key="back_home_kb"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1040 0%, #312e81 50%, #4c1d95 100%); padding: 40px; border-radius: 16px; margin-bottom: 28px; color: white;'>
        <div style='display: flex; align-items: center; margin-bottom: 12px;'>
            <span style='font-size: 3.5rem; margin-right: 25px;'>📚</span>
            <div>
                <h1 style='margin: 0; font-size: 2.4rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Knowledge Base (RAG)</h1>
                <p style='margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem;'>Upload PDFs, notes, or textbooks — then ask questions grounded in your own documents.</p>
            </div>
        </div>
        <div style='display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px;'>
            <div style='background: rgba(255,255,255,0.1); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.3rem; font-weight: 800;'>RAG</span><br/>
                <span style='font-size: 0.78rem; opacity: 0.85;'>Retrieval-Augmented</span>
            </div>
            <div style='background: rgba(255,255,255,0.1); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.3rem; font-weight: 800;'>PDF</span><br/>
                <span style='font-size: 0.78rem; opacity: 0.85;'>Supported Format</span>
            </div>
            <div style='background: rgba(255,255,255,0.1); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.3rem; font-weight: 800;'>🔍</span><br/>
                <span style='font-size: 0.78rem; opacity: 0.85;'>Vector Search</span>
            </div>
            <div style='background: rgba(255,255,255,0.1); border-radius: 8px; padding: 10px 18px; backdrop-filter: blur(8px);'>
                <span style='font-size: 1.3rem; font-weight: 800;'>📎</span><br/>
                <span style='font-size: 0.78rem; opacity: 0.85;'>Source Citations</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kb_col_left, kb_col_right = st.columns([1, 1.6], gap="large")

    with kb_col_left:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #0f172a, #1a1040); border: 1px solid rgba(139,92,246,0.3); border-radius: 14px; padding: 22px; margin-bottom: 18px;'>
            <h3 style='color: #a78bfa; margin: 0 0 6px 0; font-family: "Outfit", sans-serif; font-size: 1.05rem;'>📤 Upload Documents</h3>
            <p style='color: #6b7280; font-size: 0.8rem; margin: 0;'>Supported: PDF, TXT, MD, CSV</p>
        </div>
        """, unsafe_allow_html=True)

        # Track ingestion state to avoid re-sending on every Streamlit rerun
        if "kb_last_ingested" not in st.session_state:
            st.session_state["kb_last_ingested"] = None
        if "kb_upload_status" not in st.session_state:
            st.session_state["kb_upload_status"] = None

        uploaded_kb_file = st.file_uploader(
            "Drop a file here (PDF, TXT, MD, CSV):",
            type=["pdf", "txt", "md", "csv"],
            key="kb_file_uploader",
            label_visibility="collapsed",
            help="Select a file — it will be ingested automatically into the knowledge base."
        )

        # Auto-ingest: fires immediately when a NEW file is detected
        # Fingerprint = filename + size to detect truly new uploads
        if uploaded_kb_file is not None:
            file_fingerprint = f"{uploaded_kb_file.name}_{uploaded_kb_file.size}"
            if st.session_state["kb_last_ingested"] != file_fingerprint:
                # Read all bytes NOW before any rerun can clear the widget
                file_bytes = uploaded_kb_file.getvalue()
                file_name  = uploaded_kb_file.name
                file_type  = uploaded_kb_file.type or "application/octet-stream"

                with st.spinner(f"⏳ Ingesting **{file_name}** into knowledge base..."):
                    try:
                        files = {"file": (file_name, file_bytes, file_type)}
                        resp = requests.post(f"{BACKEND_URL}/api/upload", files=files)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["kb_last_ingested"] = file_fingerprint
                            st.session_state["kb_upload_status"] = {
                                "ok": True,
                                "msg": f"✅ **{data['filename']}** ingested successfully! Knowledge base now has **{data.get('chunks', '?')} total chunks**."
                            }
                            if "kb_documents" in st.session_state:
                                del st.session_state["kb_documents"]
                        else:
                            err_detail = ""
                            try:
                                err_detail = resp.json().get("detail", resp.text)
                            except Exception:
                                err_detail = resp.text
                            st.session_state["kb_upload_status"] = {"ok": False, "msg": f"❌ Upload failed: {err_detail}"}
                    except Exception as upload_err:
                        st.session_state["kb_upload_status"] = {"ok": False, "msg": f"❌ Cannot reach backend: {upload_err}"}
                st.rerun()

        # Persistent upload status banner
        if st.session_state.get("kb_upload_status"):
            status = st.session_state["kb_upload_status"]
            if status["ok"]:
                st.success(status["msg"])
            else:
                st.error(status["msg"])
            if st.button("✖ Dismiss", key="kb_dismiss_status"):
                st.session_state["kb_upload_status"] = None
                st.rerun()



        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        st.markdown("""<h3 style='color: #a78bfa; font-family: "Outfit", sans-serif; font-size: 1rem; margin-bottom: 10px;'>🗂️ Knowledge Base Contents</h3>""", unsafe_allow_html=True)

        if st.button("🔄 Refresh Document List", key="kb_refresh_btn"):
            # Sources stats line removed per user request.
            if "kb_documents" in st.session_state:
                del st.session_state["kb_documents"]

        if "kb_documents" not in st.session_state:
            try:
                doc_resp = requests.get(f"{BACKEND_URL}/api/rag/documents")
                if doc_resp.status_code == 200:
                    st.session_state["kb_documents"] = doc_resp.json()
                else:
                    st.session_state["kb_documents"] = {"documents": [], "total_chunks": 0}
            except Exception:
                st.session_state["kb_documents"] = {"documents": [], "total_chunks": 0}

        kb_data = st.session_state.get("kb_documents", {})
        docs_list = kb_data.get("documents", [])
        total_chunks = kb_data.get("total_chunks", 0)

        if docs_list:
            st.markdown(f"<p style='color:#6b7280; font-size:0.78rem; margin-bottom:10px;'>📊 {len(docs_list)} sources &bull; {total_chunks} total chunks indexed</p>", unsafe_allow_html=True)
            for doc in docs_list:
                doc_name = doc.get("source", "Unknown")
                doc_chunks = doc.get("chunks", "?")
                if doc_name.endswith(".pdf"):
                    icon, color = "📄", "#f87171"
                elif doc_name.endswith(".txt") or doc_name.endswith(".md"):
                    icon, color = "📝", "#34d399"
                elif "Guide" in doc_name:
                    icon, color = "📘", "#818cf8"
                else:
                    icon, color = "📁", "#9ca3af"
                st.markdown(f"""
                <div style='background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.07); border-left:3px solid {color}; border-radius:8px; padding:9px 14px; margin-bottom:7px; display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#e5e7eb; font-size:0.82rem; word-break:break-all;'>{icon} {doc_name}</span>
                    <span style='color:#6b7280; font-size:0.72rem; white-space:nowrap; margin-left:8px;'>{doc_chunks} chunks</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(15,23,42,0.5); border:1px dashed rgba(139,92,246,0.3); border-radius:10px; padding:20px; text-align:center;'>
                <div style='font-size:2rem; margin-bottom:6px;'>📭</div>
                <p style='color:#6b7280; font-size:0.83rem; margin:0;'>No documents yet. Upload a file or refresh.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px;'>
            <h4 style='color:#818cf8; margin:0 0 10px 0; font-family:"Outfit",sans-serif; font-size:0.92rem;'>⚙️ How RAG Works</h4>
            <ol style='color:#9ca3af; font-size:0.78rem; line-height:1.75; padding-left:18px; margin:0;'>
                <li>Document split into overlapping <strong style='color:#c4b5fd;'>text chunks</strong></li>
                <li>Each chunk → <strong style='color:#c4b5fd;'>vector embedding</strong></li>
                <li>Query embedded and <strong style='color:#c4b5fd;'>similarity-searched</strong></li>
                <li>Top chunks injected as <strong style='color:#c4b5fd;'>AI context</strong></li>
                <li>Answer <strong style='color:#c4b5fd;'>grounded</strong> with source citations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with kb_col_right:
        st.markdown("""
        <div style='background:linear-gradient(145deg,#0f172a,#1e1040); border:1px solid rgba(139,92,246,0.3); border-radius:14px; padding:22px; margin-bottom:16px;'>
            <h3 style='color:#a78bfa; margin:0 0 6px 0; font-family:"Outfit",sans-serif; font-size:1.05rem;'>🔍 Ask About Your Documents</h3>
            <p style='color:#6b7280; font-size:0.8rem; margin:0;'>The AI searches your knowledge base and returns a grounded, cited answer.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color:#9ca3af; font-size:0.82rem; margin-bottom:8px;'>💡 <strong>Quick Prompts</strong></p>", unsafe_allow_html=True)
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("📚 Study Notes", key="rag_preset_notes"):
                st.session_state["kb_query_text"] = ""
                st.rerun()
        with preset_col2:
            if st.button("🔁 Overfitting & Reg.", key="rag_preset_reg"):
                st.session_state["kb_query_text"] = "What are the main techniques to prevent overfitting according to the documents?"
                st.rerun()
        with preset_col3:
            if st.button("🧩 Feature Engineering", key="rag_preset_feat"):
                st.session_state["kb_query_text"] = "Explain feature engineering and selection methods from the knowledge base."
                st.rerun()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        default_query_val = st.session_state.get("kb_query_text", "Based on the uploaded documents, ask your question here.")
        kb_query = st.text_area(
            "Your question:",
            value=default_query_val,
            height=88,
            key="kb_query_area",
            placeholder="Based on the uploaded documents, ask your question here.",
            label_visibility="collapsed"
        )
        kb_top_k = st.slider("Number of document chunks to retrieve (top-K):", min_value=1, max_value=5, value=3, key="kb_top_k")

        if st.button("🔍 Search & Answer", key="kb_query_btn"):
            if not kb_query.strip():
                st.warning("Please enter a question before querying.")
            else:
                with st.spinner("🔍 Searching knowledge base and synthesizing answer..."):
                    try:
                        rag_resp = requests.post(
                            f"{BACKEND_URL}/api/rag/query",
                            json={"query": kb_query.strip(), "top_k": kb_top_k}
                        )
                        if rag_resp.status_code == 200:
                            st.session_state["kb_result"] = rag_resp.json()
                            st.session_state["kb_query_used"] = kb_query.strip()
                        else:
                            st.error(f"RAG query failed (status {rag_resp.status_code}): {rag_resp.text}")
                    except Exception as rag_err:
                        st.error(f"Could not reach backend RAG endpoint: {rag_err}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if "kb_result" in st.session_state and st.session_state.kb_result:
            result = st.session_state.kb_result
            reply = result.get("reply", "")
            grounded = result.get("grounded", False)
            context_sources = result.get("context", [])
            query_used = st.session_state.get("kb_query_used", "")

            grounded_badge = (
                "<span style='background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.4); border-radius:20px; padding:3px 12px; font-size:0.75rem; font-weight:700;'>✅ Document-Grounded</span>"
                if grounded else
                "<span style='background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.4); border-radius:20px; padding:3px 12px; font-size:0.75rem; font-weight:700;'>❌ Not Found in Knowledge Base</span>"
            )

            st.markdown(f"""
            <div style='background:linear-gradient(145deg,#0c1428,#111827); border:1px solid rgba(99,102,241,0.3); border-radius:14px; padding:24px; margin-bottom:16px;'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; flex-wrap:wrap; gap:8px;'>
                    <div>
                        <span style='color:#818cf8; font-size:0.73rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;'>📖 Query</span><br/>
                        <span style='color:#e5e7eb; font-size:0.88rem; font-style:italic;'>"{query_used}"</span>
                    </div>
                    {grounded_badge}
                </div>
                <div style='border-top:1px solid rgba(255,255,255,0.06); padding-top:14px; color:#e5e7eb; font-size:0.92rem; line-height:1.75;'>{reply.replace(chr(10), "<br/>")}</div>
            </div>
            """, unsafe_allow_html=True)

            if context_sources:
                unique_src_names = list(dict.fromkeys([c["source"] for c in context_sources]))
                src_pills = " &bull; ".join([
                    f'<code style="color:#a5b4fc; background:rgba(99,102,241,0.12); padding:1px 7px; border-radius:4px;">{s}</code>'
                    for s in unique_src_names
                ])
                st.markdown(f"""<p style='color:#6b7280; font-size:0.78rem; margin-bottom:8px;'>📎 <strong style='color:#9ca3af;'>Sources ({len(context_sources)} chunks):</strong> {src_pills}</p>""", unsafe_allow_html=True)
                with st.expander(f"📂 View Retrieved Document Excerpts ({len(context_sources)} chunks)", expanded=False):
                    for i, chunk in enumerate(context_sources):
                        st.markdown(f"""
                        <div style='background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.15); border-left:3px solid #6366f1; border-radius:8px; padding:12px 16px; margin-bottom:10px;'>
                            <p style='color:#818cf8; font-size:0.73rem; font-weight:700; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.06em;'>Chunk {i+1} &bull; {chunk["source"]}</p>
                            <p style='color:#d1d5db; font-size:0.83rem; line-height:1.6; margin:0;'>{chunk["text"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25); border-radius:10px; padding:14px 18px;'>
                    <p style='color:#fbbf24; font-size:0.84rem; margin:0;'>⚠️ No matching document chunks found. Try uploading a relevant PDF or refine your query.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("<p style='color:#6b7280; font-size:0.8rem; margin-bottom:8px;'>🔁 Follow-up Questions:</p>", unsafe_allow_html=True)
            fu1, fu2, fu3 = st.columns(3)
            with fu1:
                if st.button("📈 Scikit-learn Example", key="rag_fu1"):
                    st.session_state["kb_query_text"] = "Give me a practical Python code example demonstrating cross-validation using scikit-learn."
                    st.rerun()
            with fu2:
                if st.button("⚡ When to use LOOCV?", key="rag_fu2"):
                    st.session_state["kb_query_text"] = "When should I use Leave-One-Out Cross-Validation instead of K-Fold?"
                    st.rerun()
            with fu3:
                if st.button("📊 Best K value?", key="rag_fu3"):
                    st.session_state["kb_query_text"] = "What is the recommended value of K in K-Fold cross-validation and why?"
                    st.rerun()
        else:
            st.markdown("""
            <div style='background:linear-gradient(145deg,#0c1428,#111827); border:1px dashed rgba(139,92,246,0.35); border-radius:14px; padding:48px 30px; text-align:center; margin-top:6px;'>
                <div style='font-size:3rem; margin-bottom:14px;'>🔍</div>
                <h3 style='color:#f3f4f6; font-family:"Outfit",sans-serif; margin-bottom:8px; font-size:1.2rem;'>Ready to Answer Your Questions</h3>
                <p style='color:#6b7280; font-size:0.86rem; max-width:380px; margin:0 auto; line-height:1.65;'>
                    Type your question in the box above, then click <strong style='color:#a78bfa;'>🔍 Search &amp; Answer</strong> for a document-grounded response.
                </p>
            </div>
            """, unsafe_allow_html=True)

elif nav_selection == "📊 Interactive Visualization":
    if st.button("⬅️ Back to Home", key="back_home_iv"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()


    st.markdown("""
    <div style='background: linear-gradient(135deg, #0d9488 0%, #115e59 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>📊</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Interactive Visualization</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Upload datasets, choose columns, generate interactive charts, and get structured AI-based descriptions instantly.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Interactive Data Visualizer")
    st.write("Plot and analyze trends on your dataset (CSV) inside the interactive visualization sandbox.")
    
    uploaded_csv = st.file_uploader("Upload CSV for visualization:", type=["csv"], key="sandbox_csv")
    default_csv_path = r"C:\Users\megha\Downloads\AI_DataScience_Mentor_Dataset.csv"
    
    df_sandbox = None
    if uploaded_csv is not None:
        try:
            df_sandbox = pd.read_csv(uploaded_csv)
            st.success(f"Loaded CSV successfully!")
        except Exception as csv_err:
            st.error(f"Error reading CSV: {csv_err}")
    elif os.path.exists(default_csv_path):
        try:
            df_sandbox = pd.read_csv(default_csv_path)
            st.info("💡 Loaded default `AI_DataScience_Mentor_Dataset.csv` dataset.")
        except Exception as default_err:
            st.error(f"Failed to read default dataset: {default_err}")
            
    if df_sandbox is not None:
        with st.expander("📊 Dataset Preview"):
            st.dataframe(df_sandbox.head(5), width="stretch")
        
        all_cols = list(df_sandbox.columns)
        
        col_x, col_y, col_type = st.columns(3)
        with col_x:
            x_col = st.selectbox("Select X-Axis Column:", options=all_cols, index=0, key="viz_x")
        with col_y:
            y_col = st.selectbox("Select Y-Axis Column:", options=all_cols, index=min(1, len(all_cols)-1), key="viz_y")
        with col_type:
            chart_type = st.selectbox("Select Chart Type:", options=["Bar Chart 📊", "Line Chart 📈", "Scatter Plot 🎯", "Area Chart 🗺️"], key="viz_type")
        
        if st.button("Generate Visualization & Description ➔", key="viz_gen_btn"):
            with st.spinner("Generating interactive chart and AI description..."):
                try:
                    # Render the chart
                    st.write("---")
                    st.subheader(f"📊 {chart_type.split()[0]} of {y_col} vs {x_col}")
                    
                    # Prepare data for plotting
                    plot_df = df_sandbox[[x_col, y_col]].dropna().copy()
                    
                    if "Bar Chart" in chart_type:
                        # Group by x and take mean of y (if numeric) or count
                        is_numeric_y = pd.api.types.is_numeric_dtype(plot_df[y_col])
                        if is_numeric_y:
                            grouped = plot_df.groupby(x_col)[y_col].mean().reset_index()
                            st.bar_chart(data=grouped, x=x_col, y=y_col)
                        else:
                            # Count occurrences
                            grouped = plot_df.groupby([x_col, y_col]).size().reset_index(name="Count")
                            st.bar_chart(data=grouped, x=x_col, y="Count", color=y_col)
                    elif "Line Chart" in chart_type:
                        is_numeric_y = pd.api.types.is_numeric_dtype(plot_df[y_col])
                        if is_numeric_y:
                            grouped = plot_df.groupby(x_col)[y_col].mean().reset_index()
                            st.line_chart(data=grouped, x=x_col, y=y_col)
                        else:
                            grouped = plot_df.groupby([x_col, y_col]).size().reset_index(name="Count")
                            st.line_chart(data=grouped, x=x_col, y="Count", color=y_col)
                    elif "Scatter Plot" in chart_type:
                        st.scatter_chart(data=plot_df, x=x_col, y=y_col)
                    elif "Area Chart" in chart_type:
                        is_numeric_y = pd.api.types.is_numeric_dtype(plot_df[y_col])
                        if is_numeric_y:
                            grouped = plot_df.groupby(x_col)[y_col].mean().reset_index()
                            st.area_chart(data=grouped, x=x_col, y=y_col)
                        else:
                            grouped = plot_df.groupby([x_col, y_col]).size().reset_index(name="Count")
                            st.area_chart(data=grouped, x=x_col, y="Count", color=y_col)

                    # Compute statistics
                    def get_stats(col_name):
                        col_data = df_sandbox[col_name]
                        if pd.api.types.is_numeric_dtype(col_data):
                            return {
                                "mean": float(col_data.mean()),
                                "min": float(col_data.min()),
                                "max": float(col_data.max()),
                                "std": float(col_data.std())
                            }
                        else:
                            vc = col_data.value_counts()
                            return {
                                "unique_count": int(col_data.nunique()),
                                "most_common": str(vc.index[0]) if len(vc) > 0 else "N/A",
                                "most_common_frequency": int(vc.values[0]) if len(vc) > 0 else 0
                            }
                    
                    x_stats = get_stats(x_col)
                    y_stats = get_stats(y_col)
                    
                    # Call API
                    resp = requests.post(
                        f"{BACKEND_URL}/api/v1/describe-chart",
                        json={
                            "chart_type": chart_type,
                            "x_column": x_col,
                            "y_column": y_col,
                            "x_stats": x_stats,
                            "y_stats": y_stats
                        }
                    )
                    
                    if resp.status_code == 200:
                        desc_text = resp.json().get("description", "")
                        st.markdown("---")
                        st.markdown(f"""
                        <div style='background: linear-gradient(145deg, #0f172a, #0d9488); border: 1px solid rgba(13,148,136,0.3); border-radius: 12px; padding: 24px; margin-top: 15px;'>
                            <h3 style='color: #2dd4bf; margin: 0 0 12px 0; font-family: "Outfit", sans-serif; display: flex; align-items: center;'>
                                📊 AI Graph Description &amp; Insights
                            </h3>
                            <p style='color: #e5e7eb; font-size: 0.95rem; line-height: 1.6; margin: 0;'>
                                {desc_text}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Failed to generate AI description (status {resp.status_code}): {resp.text}")
                except Exception as viz_err:
                    st.error(f"Visualization generation failed: {viz_err}")



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🤖  AI MODEL EXPLAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif False:  # 🤖 AI Explainer removed
    import base64
    import io as _io

    # ── Hero Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #312e81 100%);
                padding: 40px; border-radius: 16px; margin-bottom: 28px; color: white;
                border: 1px solid rgba(99,102,241,0.3);'>
        <div style='display:flex; align-items:center; margin-bottom: 14px;'>
            <span style='font-size:3.5rem; margin-right:22px; filter: drop-shadow(0 0 12px #6366f1);'>🤖</span>
            <div>
                <h1 style='margin:0; font-size:2.3rem; font-weight:800; color:white; font-family:"Outfit",sans-serif;'>
                    AI Model Explainer
                </h1>
                <p style='margin:6px 0 0 0; opacity:0.85; font-size:1.05rem;'>
                    Upload a dataset, train a model, and get a plain-English AI explanation of what it learned.
                </p>
            </div>
        </div>
        <div style='display:flex; gap:14px; flex-wrap:wrap;'>
            <div style='background:rgba(255,255,255,0.08); border-radius:8px; padding:10px 18px; backdrop-filter:blur(6px);'>
                <span style='font-size:1.3rem; font-weight:800;'>SHAP</span><br/>
                <span style='font-size:0.77rem; opacity:0.85;'>Value Analysis</span>
            </div>
            <div style='background:rgba(255,255,255,0.08); border-radius:8px; padding:10px 18px; backdrop-filter:blur(6px);'>
                <span style='font-size:1.3rem; font-weight:800;'>📝</span><br/>
                <span style='font-size:0.77rem; opacity:0.85;'>AI Narrative</span>
            </div>
            <div style='background:rgba(255,255,255,0.08); border-radius:8px; padding:10px 18px; backdrop-filter:blur(6px);'>
                <span style='font-size:1.3rem; font-weight:800;'>❤️</span><br/>
                <span style='font-size:0.77rem; opacity:0.85;'>Health Report</span>
            </div>
            <div style='background:rgba(255,255,255,0.08); border-radius:8px; padding:10px 18px; backdrop-filter:blur(6px);'>
                <span style='font-size:1.3rem; font-weight:800;'>🔍</span><br/>
                <span style='font-size:0.77rem; opacity:0.85;'>Row-Level Why</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Uploader ───────────────────────────────────────────────────────
    exp_col_left, exp_col_right = st.columns([1, 1.6], gap="large")

    with exp_col_left:
        st.markdown("""
        <div style='background:linear-gradient(145deg,#0f172a,#1a1040); border:1px solid rgba(99,102,241,0.3);
                    border-radius:14px; padding:20px; margin-bottom:16px;'>
            <h3 style='color:#818cf8; margin:0 0 6px 0; font-family:"Outfit",sans-serif; font-size:1.05rem;'>📂 Dataset & Model Config</h3>
            <p style='color:#6b7280; font-size:0.8rem; margin:0;'>Upload a CSV and configure your model below.</p>
        </div>
        """, unsafe_allow_html=True)

        exp_uploaded = st.file_uploader(
            "Upload CSV dataset:",
            type=["csv"],
            key="explainer_csv",
            label_visibility="collapsed",
        )

        default_csv_path_exp = r"C:\Users\megha\Downloads\AI_DataScience_Mentor_Dataset.csv"
        df_exp = None
        data_source_label = ""

        if exp_uploaded is not None:
            try:
                df_exp = pd.read_csv(exp_uploaded)
                data_source_label = f"✅ Loaded: **{exp_uploaded.name}** ({len(df_exp):,} rows × {len(df_exp.columns)} cols)"
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        elif os.path.exists(default_csv_path_exp):
            try:
                df_exp = pd.read_csv(default_csv_path_exp)
                data_source_label = f"💡 Using default dataset ({len(df_exp):,} rows × {len(df_exp.columns)} cols)"
            except Exception:
                pass

        if data_source_label:
            st.markdown(f"<p style='color:#34d399; font-size:0.83rem; margin-bottom:10px;'>{data_source_label}</p>",
                        unsafe_allow_html=True)

        if df_exp is not None:
            all_exp_cols = list(df_exp.columns)

            exp_target = st.selectbox(
                "🎯 Target Variable (Y):",
                options=all_exp_cols,
                index=all_exp_cols.index("Quiz_Score") if "Quiz_Score" in all_exp_cols else 0,
                key="exp_target",
            )
            exp_default_feats = [c for c in all_exp_cols if c != exp_target and c != "Student_ID"]
            exp_features = st.multiselect(
                "🛠 Feature Columns (X):",
                options=[c for c in all_exp_cols if c != exp_target],
                default=exp_default_feats[:8],
                key="exp_features",
            )

            # Determine task type for model options
            is_exp_reg = pd.api.types.is_numeric_dtype(df_exp[exp_target]) and df_exp[exp_target].nunique() > 10
            if is_exp_reg:
                exp_model_opts = [
                    "Random Forest Regressor",
                    "Gradient Boosting Regressor",
                    "Decision Tree Regressor",
                    "Ridge Regression",
                    "Lasso Regression",
                    "Linear Regression",
                ]
            else:
                exp_model_opts = [
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                    "Decision Tree Classifier",
                    "Logistic Regression",
                    "K-Nearest Neighbors (KNN)",
                    "Naive Bayes",
                ]

            exp_model = st.selectbox("🤖 Algorithm:", options=exp_model_opts, key="exp_model")
            exp_test_size = st.slider("Test Split Ratio:", 0.1, 0.4, 0.2, 0.05, key="exp_test_size")
            exp_row_idx = st.number_input(
                "🔍 Row Index for Individual Explanation (test set):",
                min_value=0, max_value=500, value=0, step=1, key="exp_row_idx"
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("🔍 Train & Explain ➔", key="exp_run_btn"):
                if not exp_features:
                    st.warning("Please select at least one feature column.")
                else:
                    with st.spinner("🤖 Training model and generating AI explanations..."):
                        try:
                            # Encode CSV to base64
                            csv_buffer = _io.BytesIO()
                            df_exp.to_csv(csv_buffer, index=False)
                            csv_b64 = base64.b64encode(csv_buffer.getvalue()).decode("utf-8")

                            payload = {
                                "csv_data": csv_b64,
                                "features": exp_features,
                                "target": exp_target,
                                "model_name": exp_model,
                                "test_size": exp_test_size,
                                "explain_row_index": int(exp_row_idx),
                            }
                            resp = requests.post(
                                f"{BACKEND_URL}/api/v1/explain", json=payload, timeout=120
                            )
                            if resp.status_code == 200:
                                st.session_state["exp_result"] = resp.json()
                                st.success("✅ Model trained and explained!")
                            else:
                                try:
                                    detail = resp.json().get("detail", resp.text)
                                except Exception:
                                    detail = resp.text
                                st.error(f"Explainer API error ({resp.status_code}): {detail}")
                        except Exception as exp_err:
                            st.error(f"Could not reach backend: {exp_err}")

            # Preview
            with st.expander("📋 Dataset Preview (first 5 rows)", expanded=False):
                st.dataframe(df_exp.head(5))

        else:
            st.markdown("""
            <div style='background:rgba(15,23,42,0.5); border:1px dashed rgba(99,102,241,0.4);
                        border-radius:12px; padding:32px; text-align:center;'>
                <div style='font-size:2.5rem; margin-bottom:10px;'>📂</div>
                <p style='color:#6b7280; font-size:0.85rem; margin:0;'>
                    Upload a CSV file to get started. A default dataset will be used if available.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Results Panel ──────────────────────────────────────────────────────────
    with exp_col_right:
        if "exp_result" in st.session_state and st.session_state.exp_result:
            res = st.session_state.exp_result
            model_nm = res.get("model_name", "Model")
            task_tp = res.get("task_type", "")
            metric_nm = res.get("metric_name", "Score")
            train_sc = res.get("train_score", 0)
            test_sc = res.get("test_score", 0)
            extra = res.get("extra_metrics", {})
            top_feats = res.get("top_features", [])
            narrative_txt = res.get("narrative", "")
            feat_img_b64 = res.get("feat_chart_b64")
            shap_img_b64 = res.get("shap_chart_b64")
            shap_avail = res.get("shap_available", False)
            shap_vals = res.get("shap_values", [])
            indiv_exp = res.get("individual_explanation")

            # Health badge
            gap = train_sc - test_sc
            if gap > 0.15:
                health_color, health_icon, health_txt = "#ef4444", "🔴", "Overfitting Risk"
            elif gap > 0.08:
                health_color, health_icon, health_txt = "#f59e0b", "🟡", "Mild Gap"
            else:
                health_color, health_icon, health_txt = "#10b981", "🟢", "Healthy"

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, label, val, icon in [
                (mc1, f"Train {metric_nm}", f"{train_sc:.1%}", "📈"),
                (mc2, f"Test {metric_nm}", f"{test_sc:.1%}", "📊"),
                (mc3, "Model Health", health_txt, health_icon),
                (mc4, list(extra.keys())[0].upper() if extra else "Metric",
                 f"{list(extra.values())[0]:.4f}" if extra else "—", "🎯"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#111827,#1f2937); border:1px solid rgba(255,255,255,0.07);
                                border-radius:10px; padding:14px; text-align:center;'>
                        <div style='font-size:1.5rem;'>{icon}</div>
                        <div style='font-size:1.1rem; font-weight:800; color:#f3f4f6; font-family:"Outfit",sans-serif;'>{val}</div>
                        <div style='font-size:0.71rem; color:#6b7280;'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Tabs
            tab_fi, tab_shap, tab_narr, tab_indiv = st.tabs([
                "📊 Feature Importance",
                "🔮 SHAP Analysis",
                "🧠 AI Narrative",
                "🔍 Individual Explanation",
            ])

            with tab_fi:
                if feat_img_b64:
                    img_data = base64.b64decode(feat_img_b64)
                    st.image(img_data, use_container_width=True)
                elif top_feats:
                    feat_df_disp = pd.DataFrame(top_feats)
                    st.bar_chart(feat_df_disp.set_index("feature")["importance"])
                else:
                    st.info("Feature importance not available for this model type.")

                if top_feats:
                    st.markdown("<p style='color:#6b7280; font-size:0.78rem; margin-top:10px;'>Top 10 Feature Importance Scores:</p>",
                                unsafe_allow_html=True)
                    feat_table = pd.DataFrame(top_feats)
                    feat_table["importance"] = feat_table["importance"].apply(lambda x: f"{x:.2%}")
                    feat_table.columns = ["Feature", "Importance (%)"] if len(feat_table.columns) == 2 else feat_table.columns
                    st.dataframe(feat_table, use_container_width=True, hide_index=True)

            with tab_shap:
                if shap_avail and shap_img_b64:
                    st.markdown("""
                    <p style='color:#a78bfa; font-size:0.85rem; margin-bottom:10px;'>
                    🔮 <strong>SHAP (SHapley Additive exPlanations)</strong> measures the contribution of each
                    feature to the model's predictions using game-theoretic principles.
                    Features with higher |SHAP| values have more impact.
                    </p>
                    """, unsafe_allow_html=True)
                    shap_img_data = base64.b64decode(shap_img_b64)
                    st.image(shap_img_data, use_container_width=True)

                    if shap_vals:
                        st.markdown("<p style='color:#6b7280; font-size:0.78rem; margin-top:10px;'>Mean |SHAP| per Feature:</p>",
                                    unsafe_allow_html=True)
                        shap_df = pd.DataFrame(shap_vals)
                        shap_df["mean_abs_shap"] = shap_df["mean_abs_shap"].apply(lambda x: f"{x:.5f}")
                        shap_df.columns = ["Feature", "Mean |SHAP|"] if len(shap_df.columns) == 2 else shap_df.columns
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)
                elif not shap_avail:
                    st.markdown("""
                    <div style='background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25);
                                border-radius:12px; padding:24px; text-align:center;'>
                        <div style='font-size:2rem; margin-bottom:8px;'>⚡</div>
                        <h4 style='color:#fbbf24; margin:0 0 6px 0; font-family:"Outfit",sans-serif;'>SHAP Not Available</h4>
                        <p style='color:#9ca3af; font-size:0.84rem; margin:0;'>
                            SHAP analysis requires tree-based models (Random Forest, Gradient Boosting, Decision Tree)
                            and the <code style='color:#fbbf24;'>shap</code> library.<br/><br/>
                            Install it with: <code style='background:rgba(0,0,0,0.3); color:#a5b4fc; padding:2px 8px; border-radius:4px;'>pip install shap</code>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("SHAP chart could not be generated for this configuration.")

            with tab_narr:
                if narrative_txt:
                    st.markdown("""
                    <div style='background:linear-gradient(145deg,#0c1428,#111827); border:1px solid rgba(99,102,241,0.25);
                                border-radius:14px; padding:24px; margin-bottom:10px;'>
                        <div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
                            <span style='font-size:1.5rem;'>🧠</span>
                            <h3 style='color:#818cf8; margin:0; font-family:"Outfit",sans-serif; font-size:1.05rem;'>AI-Generated Model Explanation</h3>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(narrative_txt)
                else:
                    st.info("Narrative not yet generated. Train a model first.")

            with tab_indiv:
                if indiv_exp:
                    row_i = indiv_exp.get("row_index", 0)
                    pred_val = indiv_exp.get("prediction", "—")
                    actual_val = indiv_exp.get("actual", "—")
                    drivers = indiv_exp.get("top_drivers", [])

                    correct_icon = "✅" if str(pred_val).strip() == str(actual_val).strip() else "❌"
                    st.markdown(f"""
                    <div style='background:linear-gradient(145deg,#0c1428,#1a1040); border:1px solid rgba(139,92,246,0.3);
                                border-radius:14px; padding:22px; margin-bottom:16px;'>
                        <h3 style='color:#a78bfa; margin:0 0 14px 0; font-family:"Outfit",sans-serif; font-size:1rem;'>
                            🔍 Why did the model predict this? — Test Row #{row_i}
                        </h3>
                        <div style='display:flex; gap:20px; flex-wrap:wrap; margin-bottom:14px;'>
                            <div style='background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25); border-radius:8px; padding:10px 16px;'>
                                <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase;'>Predicted</div>
                                <div style='font-size:1.2rem; font-weight:800; color:#818cf8;'>{pred_val}</div>
                            </div>
                            <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:10px 16px;'>
                                <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase;'>Actual</div>
                                <div style='font-size:1.2rem; font-weight:800; color:#34d399;'>{actual_val}</div>
                            </div>
                            <div style='background:rgba(0,0,0,0.2); border-radius:8px; padding:10px 16px;'>
                                <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase;'>Match</div>
                                <div style='font-size:1.4rem;'>{correct_icon}</div>
                            </div>
                        </div>
                        <p style='color:#6b7280; font-size:0.8rem; margin-bottom:10px;'>🔑 Top SHAP Drivers for this prediction:</p>
                    </div>
                    """, unsafe_allow_html=True)

                    for d in drivers:
                        is_pos = d["shap_value"] > 0
                        bar_color = "#10b981" if is_pos else "#ef4444"
                        arrow = "⬆" if is_pos else "⬇"
                        sv = d["shap_value"]
                        st.markdown(f"""
                        <div style='background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.07);
                                    border-left:3px solid {bar_color}; border-radius:8px;
                                    padding:10px 16px; margin-bottom:8px;
                                    display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <span style='color:#e5e7eb; font-weight:600; font-size:0.87rem;'>{d["feature"]}</span><br/>
                                <span style='color:#6b7280; font-size:0.75rem;'>{d["direction"]}</span>
                            </div>
                            <div style='text-align:right;'>
                                <span style='color:{bar_color}; font-weight:700; font-size:0.9rem;'>{arrow} {sv:+.4f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No individual explanation available.")
        else:
            st.markdown("""
            <div style='background:linear-gradient(145deg,#0f172a,#1a1040); border:1px dashed rgba(99,102,241,0.35);
                        border-radius:16px; padding:56px 30px; text-align:center; margin-top:10px;'>
                <div style='font-size:3.5rem; margin-bottom:16px; filter:drop-shadow(0 0 16px #6366f1);'>🤖</div>
                <h3 style='color:#f3f4f6; font-family:"Outfit",sans-serif; margin-bottom:8px; font-size:1.3rem;'>
                    Ready to Explain Your Model
                </h3>
                <p style='color:#6b7280; font-size:0.88rem; max-width:400px; margin:0 auto; line-height:1.65;'>
                    Upload a CSV dataset on the left, configure your target variable, select features,
                    choose an algorithm, then click
                    <strong style='color:#818cf8;'>🔍 Train &amp; Explain</strong> to generate:
                </p>
                <div style='display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:18px;'>
                    <span style='background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); color:#818cf8;
                                 border-radius:20px; padding:5px 14px; font-size:0.77rem;'>📊 Feature Importance</span>
                    <span style='background:rgba(168,85,247,0.1); border:1px solid rgba(168,85,247,0.3); color:#c084fc;
                                 border-radius:20px; padding:5px 14px; font-size:0.77rem;'>🔮 SHAP Analysis</span>
                    <span style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); color:#34d399;
                                border-radius:20px; padding:5px 14px; font-size:0.77rem;'>🧠 AI Narrative</span>
                    <span style='background:rgba(234,88,12,0.1); border:1px solid rgba(234,88,12,0.3); color:#fb923c;
                                border-radius:20px; padding:5px 14px; font-size:0.77rem;'>🔍 Row-Level Why</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif nav_selection == "🎙️ Voice Mentor":
    if st.button("⬅️ Back to Home", key="back_home_vm"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div style='background: linear-gradient(135deg, #ec4899 0%, #be185d 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>🎙️</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>AI Voice Mentor</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Click the microphone, speak your data science question, and the AI will listen, transcribe, and read the answer back to you!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎙️ Speak with your AI Data Science Mentor")
    
    html_code = f"""
    <div id="voice-container" style="background: rgba(22, 28, 45, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 30px; text-align: center; color: #f3f4f6; font-family: 'Plus Jakarta Sans', sans-serif;">
        <div style="margin-bottom: 20px;">
            <button id="mic-btn" style="background: linear-gradient(135deg, #ec4899 0%, #be185d 100%); border: none; border-radius: 50%; width: 90px; height: 90px; color: white; font-size: 2.5rem; cursor: pointer; transition: all 0.3s; box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4); outline: none;">
                🎙️
            </button>
        </div>
        <div id="status" style="font-size: 1.1rem; font-weight: 600; color: #f472b6; margin-bottom: 15px;">Click the microphone to start speaking...</div>
        
        <div style="text-align: left; margin-top: 20px;">
            <label style="font-weight: 700; color: #9ca3af; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Your Question (Transcription):</label>
            <div id="transcript" style="background: rgba(9, 12, 21, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; min-height: 50px; margin-top: 5px; font-size: 0.95rem; color: #e5e7eb; line-height: 1.5; font-style: italic;">...</div>
        </div>

        <div style="text-align: left; margin-top: 25px;">
            <label style="font-weight: 700; color: #9ca3af; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">AI Mentor Response:</label>
            <div id="response" style="background: rgba(9, 12, 21, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 20px; min-height: 100px; margin-top: 5px; font-size: 0.95rem; color: #e5e7eb; line-height: 1.6; max-height: 250px; overflow-y: auto;">...</div>
        </div>
        
        <div style="margin-top: 20px; text-align: right;">
            <button id="speak-again" style="background: transparent; color: #f472b6; border: 1px solid #f472b6; border-radius: 8px; padding: 6px 14px; font-size: 0.85rem; cursor: pointer; display: none; font-weight: 600;">🔊 Listen Again</button>
        </div>
    </div>

    <script>
        const micBtn = document.getElementById('mic-btn');
        const statusDiv = document.getElementById('status');
        const transcriptDiv = document.getElementById('transcript');
        const responseDiv = document.getElementById('response');
        const speakAgainBtn = document.getElementById('speak-again');
        
        let recognition;
        let lastSpokenText = "";
        let isRecording = false;

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.lang = 'en-US';
            recognition.interimResults = false;

            recognition.onstart = function() {{
                isRecording = true;
                micBtn.style.transform = "scale(1.1)";
                micBtn.style.boxShadow = "0 0 25px #ec4899";
                statusDiv.innerText = "🎙️ Listening... Speak now!";
                transcriptDiv.innerText = "...";
            }};

            recognition.onresult = function(event) {{
                const textResult = event.results[0][0].transcript;
                transcriptDiv.innerText = textResult;
                statusDiv.innerText = "⏳ Processing your question...";
                sendToMentorBackend(textResult);
            }};

            recognition.onerror = function(event) {{
                console.error("Speech recognition error", event.error);
                statusDiv.innerText = "❌ Speech Error: " + event.error + ". Try again.";
                resetMicButton();
            }};

            recognition.onend = function() {{
                isRecording = false;
                resetMicButton();
            }};
        }} else {{
            statusDiv.innerText = "❌ Speech Recognition is not supported in this browser. Please use Chrome, Edge, or Safari.";
            micBtn.disabled = true;
            micBtn.style.opacity = "0.5";
        }}

        function resetMicButton() {{
            micBtn.style.transform = "none";
            micBtn.style.boxShadow = "0 4px 15px rgba(236, 72, 153, 0.4)";
        }}

        micBtn.addEventListener('click', () => {{
            if (isRecording) {{
                recognition.stop();
            }} else {{
                window.speechSynthesis.cancel();
                recognition.start();
            }}
        }});

        speakAgainBtn.addEventListener('click', () => {{
            if (lastSpokenText) {{
                speakText(lastSpokenText);
            }}
        }});

        function sendToMentorBackend(query) {{
            fetch('{BACKEND_URL}/api/v1/voice-mentor', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify({{ message: query }})
            }})
            .then(res => res.json())
            .then(data => {{
                responseDiv.innerHTML = data.reply.replace(/\\n/g, '<br/>');
                statusDiv.innerText = "🔊 Speaking...";
                lastSpokenText = data.spoken_reply;
                speakAgainBtn.style.display = "inline-block";
                speakText(data.spoken_reply);
            }})
            .catch(err => {{
                console.error(err);
                statusDiv.innerText = "❌ Backend connection error. Verify port 8000.";
            }});
        }}

        function speakText(text) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            
            const voices = window.speechSynthesis.getVoices();
            const preferredVoice = voices.find(voice => voice.name.includes("Google US English") || voice.name.includes("Natural") || voice.name.includes("Zira"));
            if (preferredVoice) {{
                utterance.voice = preferredVoice;
            }}
            utterance.rate = 1.0;
            
            utterance.onend = function() {{
                statusDiv.innerText = "✅ Done speaking. Click mic to ask another question.";
            }};
            
            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """
    
    st.components.v1.html(html_code, height=620)

elif nav_selection == "🤝 Interview Trainer":
    if st.button("⬅️ Back to Home", key="back_home_it"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    # Hero Banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #4f46e5 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>🤝</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>AI Data Science Interview Trainer</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Practice realistic coding interviews under time pressure with instant detailed feedback.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "interview_state" not in st.session_state:
        st.session_state.interview_state = "start"  # start, question, evaluation, assessment
    if "interview_topic" not in st.session_state:
        st.session_state.interview_topic = ""
    if "interview_questions" not in st.session_state:
        st.session_state.interview_questions = []
    if "interview_current_idx" not in st.session_state:
        st.session_state.interview_current_idx = 0
    if "interview_answers" not in st.session_state:
        st.session_state.interview_answers = []
    if "interview_evaluations" not in st.session_state:
        st.session_state.interview_evaluations = []
    if "interview_overall_assessment" not in st.session_state:
        st.session_state.interview_overall_assessment = None

    state = st.session_state.interview_state

    # 1. START STATE
    if state == "start":
        st.subheader("⚙️ Configure Your Mock Interview")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            topic = st.selectbox(
                "Choose a core data science topic:",
                options=[
                    "Python Programming",
                    "SQL Databases",
                    "Machine Learning",
                    "Statistics & Probability"
                ],
                key="it_topic"
            )
            
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-top:20px;'>
                <h4 style='color:#a78bfa; margin-top:0; font-family:"Outfit",sans-serif;'>Guidelines:</h4>
                <ul style='color:#9ca3af; font-size:0.9rem; padding-left:20px; line-height:1.6;'>
                    <li>The interview consists of <strong>3 sequential questions</strong>.</li>
                    <li>Each question has a strict <strong>5-minute time limit</strong>.</li>
                    <li>Submit your answer (code or explanation) before time runs out.</li>
                    <li>Each answer is graded on correctness, clarity, approach, and code quality.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            if st.button("Start Mock Interview ⚡", key="start_interview_btn", use_container_width=True):
                with st.spinner("Generating interview questions..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/api/v1/interview/start", json={"topic": topic})
                        if res.status_code == 200:
                            st.session_state.interview_topic = topic
                            st.session_state.interview_questions = res.json().get("questions", [])
                            st.session_state.interview_current_idx = 0
                            st.session_state.interview_answers = []
                            st.session_state.interview_evaluations = []
                            st.session_state.interview_state = "question"
                            st.rerun()
                        else:
                            st.error("Failed to generate questions. Verify backend server is running.")
                    except Exception as e:
                        st.error(f"Error reaching server: {e}")

    # 2. QUESTION STATE
    elif state == "question":
        idx = st.session_state.interview_current_idx
        questions = st.session_state.interview_questions
        topic = st.session_state.interview_topic
        
        if idx >= len(questions):
            st.session_state.interview_state = "assessment"
            st.rerun()
            
        current_q = questions[idx]
        question_text = current_q.get("question")
        time_limit = current_q.get("time_limit", 300)
        
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:20px; margin-bottom:20px;'>
            <span style='background:#4f46e5; color:white; padding:4px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; text-transform:uppercase;'>Question {idx+1} of {len(questions)}</span>
            <h3 style='color:white; font-family:"Outfit",sans-serif; margin-top:12px; margin-bottom:8px;'>{question_text}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Grid layout for timer and workspace
        q_col_left, q_col_right = st.columns([1.8, 1])
        
        with q_col_right:
            # Client-side countdown timer in HTML
            timer_html = f"""
            <div id="timer-box" style="background:linear-gradient(135deg, #0f172a, #1e1b4b); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:24px; text-align:center; color:white; font-family:'Plus Jakarta Sans', sans-serif;">
                <div style="font-size:0.8rem; color:#a78bfa; text-transform:uppercase; letter-spacing:0.05em; font-weight:700; margin-bottom:8px;">Time Remaining</div>
                <div id="countdown" style="font-size:3.5rem; font-weight:800; font-family:'Courier New', monospace; color:#f43f5e; text-shadow:0 0 10px rgba(244,63,94,0.4);">05:00</div>
                <div id="timer-bar-container" style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; width:100%; margin-top:15px; overflow:hidden;">
                    <div id="timer-bar" style="background:#f43f5e; height:100%; width:100%; transition: width 1s linear;"></div>
                </div>
            </div>
            
            <script>
                let totalSeconds = {time_limit};
                const countdownDiv = document.getElementById('countdown');
                const timerBar = document.getElementById('timer-bar');
                
                const timer = setInterval(() => {{
                    if (totalSeconds <= 0) {{
                        clearInterval(timer);
                        countdownDiv.innerText = "00:00";
                        countdownDiv.style.color = "#ef4444";
                        timerBar.style.width = "0%";
                        // Trigger alert on timer completion
                        alert("Time limit exceeded for this question! Please formulate and submit your response now.");
                    }} else {{
                        totalSeconds--;
                        let minutes = Math.floor(totalSeconds / 60);
                        let seconds = totalSeconds % 60;
                        
                        minutes = minutes < 10 ? '0' + minutes : minutes;
                        seconds = seconds < 10 ? '0' + seconds : seconds;
                        
                        countdownDiv.innerText = minutes + ":" + seconds;
                        
                        // Change color based on remaining time
                        let percentage = (totalSeconds / {time_limit}) * 100;
                        timerBar.style.width = percentage + "%";
                        
                        if (percentage < 25) {{
                            countdownDiv.style.color = "#f43f5e";
                            timerBar.style.backgroundColor = "#f43f5e";
                        }} else if (percentage < 50) {{
                            countdownDiv.style.color = "#fb923c";
                            timerBar.style.backgroundColor = "#fb923c";
                        }} else {{
                            countdownDiv.style.color = "#34d399";
                            timerBar.style.backgroundColor = "#34d399";
                        }}
                    }}
                }}, 1000);
            </script>
            """
            st.components.v1.html(timer_html, height=220)
            
        with q_col_left:
            # Code / Text area
            st.markdown("<p style='color:#9ca3af; font-size:0.9rem; margin-bottom:8px;'>Write your solution below. Use clean formatting, write comments where appropriate, and explain your rationale:</p>", unsafe_allow_html=True)
            user_ans = st.text_area(
                "Write your code or answer here:",
                height=260,
                key=f"ans_input_{idx}",
                label_visibility="collapsed",
                placeholder="# Example:\n# def solve():\n#     pass\n"
            )
            
            if st.button("Submit Answer ➔", key=f"submit_ans_{idx}", use_container_width=True):
                if not user_ans.strip():
                    st.warning("Please provide an answer before submitting.")
                else:
                    with st.spinner("Interviewer is evaluating your response..."):
                        try:
                            payload = {
                                "topic": topic,
                                "question": question_text,
                                "user_answer": user_ans
                            }
                            res = requests.post(f"{BACKEND_URL}/api/v1/interview/evaluate", json=payload)
                            if res.status_code == 200:
                                st.session_state.interview_answers.append(user_ans)
                                st.session_state.interview_evaluations.append(res.json())
                                st.session_state.interview_state = "evaluation"
                                st.rerun()
                            else:
                                st.error("Feedback evaluation failed. Please check backend.")
                        except Exception as e:
                            st.error(f"Error evaluating: {e}")

    # 3. EVALUATION STATE
    elif state == "evaluation":
        idx = st.session_state.interview_current_idx
        questions = st.session_state.interview_questions
        topic = st.session_state.interview_topic
        ans = st.session_state.interview_answers[idx]
        eval_res = st.session_state.interview_evaluations[idx]
        
        score = eval_res.get("score", 0)
        feedback = eval_res.get("feedback", "")
        correctness = eval_res.get("correctness", "")
        clarity = eval_res.get("clarity", "")
        approach = eval_res.get("approach", "")
        code_quality = eval_res.get("code_quality", "")
        
        # Render score and feedback
        st.subheader(f"📊 Question {idx+1} Evaluation")
        
        # Color mapping for rating
        if score >= 8:
            score_color = "#10b981"
            badge = "🟢 Strong"
        elif score >= 5:
            score_color = "#fb923c"
            badge = "🟡 Average"
        else:
            score_color = "#ef4444"
            badge = "🔴 Needs Improvement"
            
        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, #111827, #1f2937); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:30px; text-align:center;'>
                <div style='font-size:0.8rem; color:#9ca3af; text-transform:uppercase;'>Score</div>
                <div style='font-size:3.5rem; font-weight:800; color:{score_color};'>{score}/10</div>
                <div style='font-size:0.9rem; font-weight:600; color:#e5e7eb; margin-top:8px;'>{badge}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:20px; height:100%;'>
                <h4 style='color:#a78bfa; margin-top:0; font-family:"Outfit",sans-serif;'>Trainer Feedback:</h4>
                <p style='color:#d1d5db; font-size:0.95rem; line-height:1.5; margin:0;'>{feedback}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Show criteria breakdown
        st.markdown("### 🔍 Criteria Breakdown")
        
        for name, value, icon in [
            ("Correctness", correctness, "✅"),
            ("Clarity", clarity, "💡"),
            ("Problem-Solving Approach", approach, "🔮"),
            ("Code Quality", code_quality, "💻")
        ]:
            st.markdown(f"""
            <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:12px 18px; margin-bottom:8px;'>
                <div style='font-weight:700; color:#f3f4f6; font-size:0.9rem; display:flex; align-items:center; gap:8px;'>
                    <span>{icon}</span> {name}
                </div>
                <div style='color:#9ca3af; font-size:0.82rem; margin-top:4px; line-height:1.4;'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Next buttons
        is_last = (idx + 1) >= len(questions)
        btn_label = "Proceed to Final Assessment ➔" if is_last else "Next Question ➔"
        
        if st.button(btn_label, key="next_q_btn", use_container_width=True):
            if is_last:
                # Generate final assessment
                with st.spinner("Generating overall mock interview assessment..."):
                    try:
                        payload = {
                            "topic": topic,
                            "questions": [q.get("question") for q in questions],
                            "answers": st.session_state.interview_answers,
                            "evaluations": st.session_state.interview_evaluations
                        }
                        res = requests.post(f"{BACKEND_URL}/api/v1/interview/assess", json=payload)
                        if res.status_code == 200:
                            st.session_state.interview_overall_assessment = res.json()
                            st.session_state.interview_state = "assessment"
                            st.rerun()
                        else:
                            st.error("Overall assessment compilation failed.")
                    except Exception as e:
                        st.error(f"Error reaching backend: {e}")
            else:
                st.session_state.interview_current_idx += 1
                st.session_state.interview_state = "question"
                st.rerun()

    # 4. ASSESSMENT STATE
    elif state == "assessment":
        assessment = st.session_state.interview_overall_assessment
        topic = st.session_state.interview_topic
        
        if not assessment:
            st.error("No assessment data found.")
            if st.button("Return to Start"):
                st.session_state.interview_state = "start"
                st.rerun()
        else:
            overall_score = assessment.get("overall_score", 0.0)
            strengths = assessment.get("strengths", "")
            weaknesses = assessment.get("weaknesses", "")
            suggestions = assessment.get("suggestions", "")
            
            st.subheader("🏁 Interview Training Assessment")
            
            # Score card
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, #065f46, #064e3b); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:30px; text-align:center; margin-bottom:25px;'>
                <div style='font-size:0.9rem; color:#a7f3d0; text-transform:uppercase; letter-spacing:0.05em; font-weight:700;'>Overall Interview Performance</div>
                <div style='font-size:4rem; font-weight:800; color:white; margin-top:8px;'>{overall_score} / 10</div>
            </div>
            """, unsafe_allow_html=True)
            
            def to_html_bullets(text):
                lines = []
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("- ") or line.startswith("* "):
                        lines.append(f"• {line[2:]}")
                    elif line:
                        lines.append(line)
                return "<br/>".join(lines)
            
            strengths_html = to_html_bullets(strengths)
            weaknesses_html = to_html_bullets(weaknesses)
            suggestions_html = to_html_bullets(suggestions)

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown(f"""
                <div style='background:rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:20px; height:100%;'>
                    <h4 style='color:#34d399; margin-top:0; font-family:"Outfit",sans-serif; display:flex; align-items:center; gap:8px;'>🟢 Strengths</h4>
                    <div style='color:#d1d5db; font-size:0.9rem; line-height:1.55;'>{strengths_html}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_a2:
                st.markdown(f"""
                <div style='background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:20px; height:100%;'>
                    <h4 style='color:#f87171; margin-top:0; font-family:"Outfit",sans-serif; display:flex; align-items:center; gap:8px;'>🔴 Areas for Improvement</h4>
                    <div style='color:#d1d5db; font-size:0.9rem; line-height:1.55;'>{weaknesses_html}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:24px; margin-bottom:25px;'>
                <h4 style='color:#818cf8; margin-top:0; font-family:"Outfit",sans-serif; display:flex; align-items:center; gap:8px;'>🔮 Improvement Strategy & Suggestions</h4>
                <div style='color:#d1d5db; font-size:0.92rem; line-height:1.55;'>{suggestions_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start New Mock Interview ⚡", key="restart_btn", use_container_width=True):
                st.session_state.interview_state = "start"
                st.session_state.interview_overall_assessment = None
                st.rerun()

elif nav_selection == "💻 Challenge Generator":
    if st.button("⬅️ Back to Home", key="back_home_cg"):
        st.session_state.target_nav = "🏠 Home"
        st.rerun()

    st.markdown("""
    <div style='background: linear-gradient(135deg, #065f46 0%, #047857 50%, #10b981 100%); padding: 35px; border-radius: 12px; margin-bottom: 25px; color: white; display: flex; align-items: center;'>
        <span style='font-size: 3rem; margin-right: 25px;'>💻</span>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; font-weight: 800; color: white; font-family: "Outfit", sans-serif;'>Coding Challenge Generator</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Auto-generates fresh Python/SQL/Pandas challenges at your level with detailed grading.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "challenge_state" not in st.session_state:
        st.session_state.challenge_state = "setup"  # setup, workspace, evaluation
    if "challenge_lang" not in st.session_state:
        st.session_state.challenge_lang = "Python"
    if "challenge_level" not in st.session_state:
        st.session_state.challenge_level = "Beginner"
    if "challenge_current" not in st.session_state:
        st.session_state.challenge_current = None
    if "challenge_user_code" not in st.session_state:
        st.session_state.challenge_user_code = ""
    if "challenge_eval" not in st.session_state:
        st.session_state.challenge_eval = None

    c_state = st.session_state.challenge_state

    # 1. SETUP STATE
    if c_state == "setup":
        st.subheader("⚙️ Challenge Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            lang = st.selectbox(
                "Choose Programming Language / Domain:",
                options=["Python", "SQL", "Data Analysis"],
                key="cg_lang"
            )
            level = st.selectbox(
                "Select Skill Level:",
                options=["Beginner", "Intermediate", "Advanced"],
                key="cg_level"
            )
        
        with col2:
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px;'>
                <h4 style='color:#34d399; margin-top:0; font-family:"Outfit",sans-serif;'>Learning Policy:</h4>
                <ul style='color:#9ca3af; font-size:0.88rem; padding-left:20px; line-height:1.5; margin-bottom:0;'>
                    <li>Challenges focus on real-world data science tasks.</li>
                    <li>Ollama auto-generates unique problem statements dynamically.</li>
                    <li>Detailed feedback validates correctness, speed, and optimization.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("Generate Fresh Challenge ⚡", key="gen_challenge_btn", use_container_width=True):
            with st.spinner("Generating unique coding challenge..."):
                try:
                    payload = {"language": lang, "level": level}
                    res = requests.post(f"{BACKEND_URL}/api/v1/challenge/generate", json=payload)
                    if res.status_code == 200:
                        st.session_state.challenge_lang = lang
                        st.session_state.challenge_level = level
                        st.session_state.challenge_current = res.json()
                        st.session_state.challenge_state = "workspace"
                        st.session_state.challenge_eval = None
                        st.rerun()
                    else:
                        st.error("Failed to generate challenge. Verify backend status.")
                except Exception as e:
                    st.error(f"Error connecting to server: {e}")

    # 2. WORKSPACE STATE
    elif c_state == "workspace":
        ch = st.session_state.challenge_current
        lang = st.session_state.challenge_lang
        level = st.session_state.challenge_level
        
        title = ch.get("title", "Coding Challenge")
        desc = ch.get("description", "")
        template = ch.get("template", "")
        sample_in = ch.get("sample_input", "")
        sample_out = ch.get("sample_output", "")
        constraints = ch.get("constraints", "")
        hints = ch.get("hints", "")
        objective = ch.get("learning_objective", "")
        
        st.markdown(f"""
        <div style='background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:22px; margin-bottom:20px;'>
            <span style='background:#10b981; color:white; padding:4px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; text-transform:uppercase;'>{level} • {lang}</span>
            <h3 style='color:white; font-family:"Outfit",sans-serif; margin-top:12px; margin-bottom:8px;'>{title}</h3>
            <p style='color:#d1d5db; font-size:0.95rem; line-height:1.55; margin-bottom:12px;'>{desc}</p>
            <div style='background:rgba(0,0,0,0.15); border-radius:8px; padding:12px 16px; font-size:0.84rem; color:#9ca3af;'>
                <strong>🎯 Objective:</strong> {objective}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_w1, col_w2 = st.columns([1.5, 1])
        
        with col_w2:
            st.markdown("### 📋 Guidelines")
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:18px; margin-bottom:15px;'>
                <div style='font-weight:700; color:#e5e7eb; font-size:0.85rem; text-transform:uppercase;'>📥 Sample Input:</div>
                <code style='color:#a5b4fc; font-size:0.85rem;'>{sample_in}</code>
                <div style='font-weight:700; color:#e5e7eb; font-size:0.85rem; text-transform:uppercase; margin-top:12px;'>📤 Sample Output:</div>
                <code style='color:#34d399; font-size:0.85rem;'>{sample_out}</code>
            </div>
            """, unsafe_allow_html=True)
            
            if constraints:
                with st.expander("⛓️ Constraints", expanded=True):
                    st.markdown(constraints)
            
            if hints:
                with st.expander("💡 Hints & Help", expanded=False):
                    st.markdown(hints)
                    
        with col_w1:
            st.markdown("### 💻 Solution Workspace")
            code_val = st.text_area(
                "Write your solution:",
                value=template,
                height=300,
                key="challenge_code_editor"
            )
            
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Evaluate Solution ➔", key="eval_chal_btn", use_container_width=True):
                    if not code_val.strip():
                        st.warning("Please write some code before submitting.")
                    else:
                        with st.spinner("Grading your solution..."):
                            try:
                                payload = {
                                    "language": lang,
                                    "level": level,
                                    "title": title,
                                    "description": desc,
                                    "user_code": code_val
                                }
                                res = requests.post(f"{BACKEND_URL}/api/v1/challenge/evaluate", json=payload)
                                if res.status_code == 200:
                                    st.session_state.challenge_user_code = code_val
                                    st.session_state.challenge_eval = res.json()
                                    st.session_state.challenge_state = "evaluation"
                                    st.rerun()
                                else:
                                    st.error("Evaluation request failed. Check backend server.")
                            except Exception as e:
                                st.error(f"Error reaching server: {e}")
            with c_btn2:
                if st.button("Cancel & Return 🎾", key="cancel_chal_btn", use_container_width=True):
                    st.session_state.challenge_state = "setup"
                    st.rerun()

    # 3. EVALUATION STATE
    elif c_state == "evaluation":
        ch = st.session_state.challenge_current
        lang = st.session_state.challenge_lang
        level = st.session_state.challenge_level
        user_c = st.session_state.challenge_user_code
        eval_res = st.session_state.challenge_eval
        
        status = eval_res.get("status", "Success")
        score = eval_res.get("score", 100)
        feedback = eval_res.get("feedback", "")
        optimal = eval_res.get("optimal_solution", "")
        
        st.subheader("🏁 Challenge Result & Evaluation")
        
        if status == "Success":
            status_color = "#10b981"
            badge = "✅ Solved"
        elif status == "Failed":
            status_color = "#ef4444"
            badge = "❌ Incorrect"
        else:
            status_color = "#fb923c"
            badge = "⚠️ Needs Optimization"
            
        col_e1, col_e2 = st.columns([1, 2.5])
        
        with col_e1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, #111827, #1f2937); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:30px; text-align:center;'>
                <div style='font-size:0.8rem; color:#9ca3af; text-transform:uppercase;'>Score</div>
                <div style='font-size:3.5rem; font-weight:800; color:{status_color};'>{score}/100</div>
                <div style='font-size:0.95rem; font-weight:700; color:#e5e7eb; margin-top:8px;'>{badge}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_e2:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:20px; height:100%;'>
                <h4 style='color:#34d399; margin-top:0; font-family:"Outfit",sans-serif;'>Grading Insights:</h4>
                <p style='color:#d1d5db; font-size:0.95rem; line-height:1.55; margin:0;'>{feedback}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        
        if optimal:
            with st.expander("💡 View Optimal Solution & Annotations", expanded=True):
                st.markdown(optimal)
                
        col_eb1, col_eb2 = st.columns(2)
        with col_eb1:
            if st.button("New Challenge ⚡", key="cg_new_btn", use_container_width=True):
                st.session_state.challenge_state = "setup"
                st.rerun()
        with col_eb2:
            if st.button("Try Question Again 🔄", key="cg_retry_btn", use_container_width=True):
                st.session_state.challenge_state = "workspace"
                st.rerun()



