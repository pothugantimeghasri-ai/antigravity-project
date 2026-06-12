import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv

# Load configurations
BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    try:
        if "BACKEND_URL" in st.secrets:
            BACKEND_URL = st.secrets["BACKEND_URL"]
    except Exception:
        pass
if not BACKEND_URL:
    BACKEND_URL = "http://127.0.0.1:8000"

BACKEND_URL = BACKEND_URL.rstrip("/")



# Set Page Config
st.set_page_config(
    page_title="AI Dataset & Document Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
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
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("🎓 AI Dataset & Document Mentor")
st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: -10px;'>Upload your files (PDF, CSV, TXT, MD) and ask questions. Get precise, document-grounded answers directly related to your file content.</p>", unsafe_allow_html=True)
st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)

# Main Split-pane Layout
kb_col_left, kb_col_right = st.columns([1, 1.6], gap="large")

with kb_col_left:
    st.markdown("""
    <div style='background: linear-gradient(145deg, #0f172a, #1a1040); border: 1px solid rgba(139,92,246,0.3); border-radius: 14px; padding: 22px; margin-bottom: 18px;'>
        <h3 style='color: #a78bfa; margin: 0 0 6px 0; font-family: "Outfit", sans-serif; font-size: 1.1rem;'>📤 Upload Document / Dataset</h3>
        <p style='color: #6b7280; font-size: 0.85rem; margin: 0;'>Supported formats: PDF, TXT, MD, CSV</p>
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
    if uploaded_kb_file is not None:
        file_fingerprint = f"{uploaded_kb_file.name}_{uploaded_kb_file.size}"
        if st.session_state["kb_last_ingested"] != file_fingerprint:
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
        if st.button("✖ Dismiss Status", key="kb_dismiss_status"):
            st.session_state["kb_upload_status"] = None
            st.rerun()

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    st.markdown("""<h3 style='color: #a78bfa; font-family: "Outfit", sans-serif; font-size: 1.1rem; margin-bottom: 10px;'>🗂️ Ingested Files</h3>""", unsafe_allow_html=True)

    if st.button("🔄 Refresh Document List", key="kb_refresh_btn"):
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
        st.markdown(f"<p style='color:#6b7280; font-size:0.85rem; margin-bottom:10px;'>📊 {len(docs_list)} sources &bull; {total_chunks} total chunks indexed</p>", unsafe_allow_html=True)
        for doc in docs_list:
            doc_name = doc.get("source", "Unknown")
            doc_chunks = doc.get("chunks", "?")
            if doc_name.endswith(".pdf"):
                icon, color = "📄", "#f87171"
            elif doc_name.endswith(".txt") or doc_name.endswith(".md"):
                icon, color = "📝", "#34d399"
            elif doc_name.endswith(".csv"):
                icon, color = "📊", "#60a5fa"
            else:
                icon, color = "📁", "#9ca3af"
            st.markdown(f"""
            <div style='background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.07); border-left:3px solid {color}; border-radius:8px; padding:9px 14px; margin-bottom:7px; display:flex; justify-content:space-between; align-items:center;'>
                <span style='color:#e5e7eb; font-size:0.85rem; word-break:break-all;'>{icon} {doc_name}</span>
                <span style='color:#6b7280; font-size:0.75rem; white-space:nowrap; margin-left:8px;'>{doc_chunks} chunks</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(15,23,42,0.5); border:1px dashed rgba(139,92,246,0.3); border-radius:10px; padding:20px; text-align:center;'>
            <div style='font-size:2rem; margin-bottom:6px;'>📭</div>
            <p style='color:#6b7280; font-size:0.85rem; margin:0;'>No documents uploaded yet. Drop a file above to begin.</p>
        </div>
        """, unsafe_allow_html=True)

with kb_col_right:
    st.markdown("""
    <div style='background:linear-gradient(145deg,#0f172a,#1e1040); border: 1px solid rgba(139,92,246,0.3); border-radius: 14px; padding: 22px; margin-bottom: 16px;'>
        <h3 style='color:#a78bfa; margin:0 0 6px 0; font-family:"Outfit",sans-serif; font-size:1.1rem;'>🔍 Ask Questions From Your File</h3>
        <p style='color:#6b7280; font-size:0.85rem; margin:0;'>The AI searches only your uploaded document context to formulate related answers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color:#9ca3af; font-size:0.85rem; margin-bottom:8px;'>💡 <strong>Quick Presets</strong></p>", unsafe_allow_html=True)
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("📊 Explain Main Findings", key="rag_preset_notes"):
            st.session_state["kb_query_text"] = "What are the main topics or findings discussed in the uploaded document?"
            st.rerun()
    with preset_col2:
        if st.button("🔁 Overfitting & Reg.", key="rag_preset_reg"):
            st.session_state["kb_query_text"] = "What does the document state about overfitting and regularization?"
            st.rerun()
    with preset_col3:
        if st.button("🧩 Cross-Validation", key="rag_preset_feat"):
            st.session_state["kb_query_text"] = "Explain k-fold cross-validation based on the uploaded file."
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if "kb_query_text" not in st.session_state:
        st.session_state["kb_query_text"] = ""

    kb_query = st.text_area(
        "Your question:",
        value=st.session_state["kb_query_text"],
        height=88,
        key="kb_query_area",
        placeholder="Type your question about the uploaded document/dataset here...",
        label_visibility="collapsed"
    )
    # Sync visual value with state
    st.session_state["kb_query_text"] = kb_query

    kb_top_k = st.slider("Document retrieval sensitivity (top-K chunks):", min_value=1, max_value=5, value=3, key="kb_top_k")

    if st.button("🔍 Search & Answer Document", key="kb_query_btn"):
        if not kb_query.strip():
            st.warning("Please enter a question before querying.")
        else:
            with st.spinner("🔍 Retrieving context and generating related answer..."):
                try:
                    rag_resp = requests.post(
                        f"{BACKEND_URL}/api/rag/query",
                        json={"query": kb_query.strip(), "top_k": kb_top_k}
                    )
                    if rag_resp.status_code == 200:
                        st.session_state["kb_result"] = rag_resp.json()
                        st.session_state["kb_query_used"] = kb_query.strip()
                    else:
                        st.error(f"Query failed: {rag_resp.text}")
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
                    <span style='color:#818cf8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;'>📖 Question</span><br/>
                    <span style='color:#e5e7eb; font-size:0.95rem; font-style:italic;'>"{query_used}"</span>
                </div>
                {grounded_badge}
            </div>
            <div style='border-top:1px solid rgba(255,255,255,0.06); padding-top:14px; color:#e5e7eb; font-size:0.95rem; line-height:1.75;'>{reply.replace(chr(10), "<br/>")}</div>
        </div>
        """, unsafe_allow_html=True)

        if context_sources and grounded:
            unique_src_names = list(dict.fromkeys([c["source"] for c in context_sources]))
            src_pills = " &bull; ".join([
                f'<code style="color:#a5b4fc; background:rgba(99,102,241,0.12); padding:1px 7px; border-radius:4px;">{s}</code>'
                for s in unique_src_names
            ])
            st.markdown(f"""<p style='color:#6b7280; font-size:0.85rem; margin-bottom:8px;'>📎 <strong style='color:#9ca3af;'>Sources ({len(context_sources)} chunks):</strong> {src_pills}</p>""", unsafe_allow_html=True)
            with st.expander(f"📂 View Retrieved Document Excerpts ({len(context_sources)} chunks)", expanded=False):
                for i, chunk in enumerate(context_sources):
                    st.markdown(f"""
                    <div style='background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.15); border-left:3px solid #6366f1; border-radius:8px; padding:12px 16px; margin-bottom:10px;'>
                        <p style='color:#818cf8; font-size:0.75rem; font-weight:700; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.06em;'>Chunk {i+1} &bull; {chunk["source"]}</p>
                        <p style='color:#d1d5db; font-size:0.85rem; line-height:1.6; margin:0;'>{chunk["text"]}</p>
                    </div>
                    """, unsafe_allow_html=True)


    else:
        st.markdown("""
        <div style='background:linear-gradient(145deg,#0c1428,#111827); border:1px dashed rgba(139,92,246,0.35); border-radius:14px; padding:48px 30px; text-align:center; margin-top:6px;'>
            <div style='font-size:3rem; margin-bottom:14px;'>🔍</div>
            <h3 style='color:#f3f4f6; font-family:"Outfit",sans-serif; margin-bottom:8px; font-size:1.2rem;'>Ready to Answer Your Questions</h3>
            <p style='color:#6b7280; font-size:0.9rem; max-width:380px; margin:0 auto; line-height:1.65;'>
                Upload a document on the left, type your query, and hit <strong style='color:#a78bfa;'>Search &amp; Answer Document</strong> to generate answers strictly related to your file.
            </p>
        </div>
        """, unsafe_allow_html=True)
