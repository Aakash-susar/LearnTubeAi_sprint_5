import streamlit as st
import streamlit.components.v1 as components
import re
import time
import sys
import io
import os
import html
import requests

from src.utils.pdf_exporter import generate_summary_pdf

# FastAPI Backend Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="LearnTube AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    section[data-testid="stMain"] {
        background: #0f0f0f;
        color: #f1f1f1;
    }

    .stApp {
        background: #0f0f0f;
        color: #f1f1f1;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    .stTextInput > div > div > input {
        background: #272727 !important;
        border: 2px solid #3a3a3a !important;
        color: #f1f1f1 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #cc0000 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #cc0000 0%, #aa0000 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #e60000 0%, #cc0000 100%) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1a1a1a;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #272727;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #888 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: #cc0000 !important;
        color: white !important;
        font-weight: 600 !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #272727 !important;
        color: #f1f1f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ========================================
# SESSION STATE
# ========================================

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_data" not in st.session_state:
    st.session_state.video_data = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "ingestion_result" not in st.session_state:
    st.session_state.ingestion_result = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_summary" not in st.session_state:
    st.session_state.video_summary = None
if "video_summaries" not in st.session_state:
    st.session_state.video_summaries = {}
if "video_quiz" not in st.session_state:
    st.session_state.video_quiz = None
if "video_quizzes" not in st.session_state:
    st.session_state.video_quizzes = {}
if "quiz_states" not in st.session_state:
    st.session_state.quiz_states = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "video_visualizations" not in st.session_state:
    st.session_state.video_visualizations = {}
if "current_timestamp" not in st.session_state:
    st.session_state.current_timestamp = 0
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ========================================
# FASTAPI BACKEND CLIENT CALLS
# ========================================

def _handle_api_error(response, default_msg="API request failed"):
    """Extract clean error message from FastAPI response."""
    try:
        data = response.json()
        return data.get("detail", default_msg)
    except Exception:
        return response.text or default_msg


def call_process_video(url):
    """Call FastAPI /video/process endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video/process",
            json={"url": url, "auto_ingest": False},
            timeout=35
        )
        if response.status_code == 200:
            payload = response.json()
            return payload.get("data", {}).get("video_data")
        error_msg = _handle_api_error(response, "Failed to process YouTube video")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_upload_video(file_bytes, filename):
    """Call FastAPI /video/upload endpoint with multipart/form-data."""
    try:
        files = {"file": (filename, file_bytes)}
        response = requests.post(
            f"{API_BASE_URL}/video/upload?auto_ingest=false",
            files=files,
            timeout=600
        )
        if response.status_code == 200:
            payload = response.json()
            return payload.get("data", {}).get("video_data")
        error_msg = _handle_api_error(response, "Failed to upload and transcribe media file")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_get_video(video_id):
    """Retrieve full video data directly by video_id from FastAPI /video/{video_id}."""
    try:
        response = requests.get(f"{API_BASE_URL}/video/{video_id}", timeout=15)
        if response.status_code == 200:
            return response.json().get("data", {}).get("video_data")
        return None
    except Exception:
        return None


def call_validate(metadata, transcript, video_id=None):
    """Call FastAPI /video/validate endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video/validate",
            json={"metadata": metadata, "transcript": transcript, "video_id": video_id},
            timeout=35
        )
        if response.status_code == 200:
            return response.json().get("data", {})
        error_msg = _handle_api_error(response, "Validation failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_ingest(transcript, video_id):
    """Call FastAPI /video/ingest endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video/ingest",
            json={"transcript": transcript, "video_id": video_id, "force_reindex": False},
            timeout=90
        )
        if response.status_code == 200:
            return response.json().get("data", {})
        error_msg = _handle_api_error(response, "Ingestion failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_answer(question, video_id, metadata=None):
    """Call FastAPI /chat endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question, "video_id": video_id, "k": 5, "metadata": metadata},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("data", {})
        error_msg = _handle_api_error(response, "Chat request failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_agent_answer(question, video_id, chat_history=None, metadata=None):
    """Call FastAPI /chat/agent endpoint for Multi-Agent LangGraph reasoning."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/agent",
            json={
                "question": question,
                "video_id": video_id,
                "chat_history": chat_history or [],
                "metadata": metadata
            },
            timeout=90
        )
        if response.status_code == 200:
            return response.json().get("data", {})
        error_msg = _handle_api_error(response, "Agent reasoning request failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_get_chat_history(video_id):
    """Retrieve persistent conversation history from MongoDB."""
    try:
        response = requests.get(f"{API_BASE_URL}/chat/{video_id}", timeout=15)
        if response.status_code == 200:
            return response.json().get("history", [])
        return []
    except Exception:
        return []


def call_get_cached_summary(video_id, style="study_notes"):
    """Retrieve persistent summary from MongoDB for a specific style if it exists."""
    try:
        response = requests.get(f"{API_BASE_URL}/video/summary/{video_id}?style={style}", timeout=15)
        if response.status_code == 200 and response.json().get("cached"):
            return response.json().get("summary")
        return None
    except Exception:
        return None


def call_get_cached_quiz(video_id, num_questions=None, style="quick_test"):
    """Retrieve persistent quiz from MongoDB if it exists."""
    try:
        url = f"{API_BASE_URL}/video/quiz/{video_id}?style={style}"
        if num_questions:
            url += f"&num_questions={num_questions}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200 and response.json().get("cached"):
            return response.json().get("quiz")
        return None
    except Exception:
        return None


def call_summary(transcript, metadata=None, video_id=None, style="study_notes", force_regenerate=False):
    """Call FastAPI /video/summary endpoint with multi-style MongoDB caching."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video/summary",
            json={
                "transcript": transcript,
                "metadata": metadata,
                "video_id": video_id,
                "style": style,
                "force_regenerate": force_regenerate
            },
            timeout=90
        )
        if response.status_code == 200:
            return response.json().get("summary", "")
        error_msg = _handle_api_error(response, "Summary generation failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_quiz(transcript, metadata=None, num_questions=5, video_id=None, style="quick_test", force_regenerate=False):
    """Call FastAPI /video/quiz endpoint with multi-style MongoDB caching."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video/quiz",
            json={
                "transcript": transcript,
                "metadata": metadata,
                "num_questions": num_questions,
                "video_id": video_id,
                "style": style,
                "force_regenerate": force_regenerate
            },
            timeout=90
        )
        if response.status_code == 200:
            return response.json().get("quiz", [])
        error_msg = _handle_api_error(response, "Quiz generation failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def call_get_all_videos(limit=50):
    """Retrieve list of all processed videos stored in MongoDB."""
    try:
        response = requests.get(f"{API_BASE_URL}/videos?limit={limit}", timeout=10)
        if response.status_code == 200:
            return response.json().get("videos", [])
        return []
    except Exception:
        return []


def call_delete_video(video_id):
    """Delete a video and all associated documents from MongoDB."""
    try:
        response = requests.delete(f"{API_BASE_URL}/video/{video_id}", timeout=10)
        return response.status_code == 200 and response.json().get("success", False)
    except Exception:
        return False


def call_get_cached_visualization(video_id, visual_type="mindmap"):
    """Retrieve persistent Mermaid diagram from MongoDB if available."""
    try:
        response = requests.get(f"{API_BASE_URL}/video/visualization/{video_id}?type={visual_type}", timeout=15)
        if response.status_code == 200 and response.json().get("cached"):
            return response.json().get("diagram")
        return None
    except Exception:
        return None


def call_visualization(transcript, metadata=None, video_id=None, visual_type="mindmap", force_regenerate=False):
    """Call FastAPI /video/visualization endpoint with MongoDB caching."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video/visualization",
            json={
                "transcript": transcript,
                "metadata": metadata,
                "video_id": video_id,
                "visual_type": visual_type,
                "force_regenerate": force_regenerate
            },
            timeout=90
        )
        if response.status_code == 200:
            return response.json().get("diagram", "")
        error_msg = _handle_api_error(response, "Visualization generation failed")
        raise RuntimeError(error_msg)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure it is running: `uvicorn backend.main:app --reload`")


def render_mermaid(diagram_code: str, height: int = 620):
    """
    Renders Mermaid code into a high-clarity, interactive vector SVG with:
    - High-contrast vibrant Mindmap & Flowchart palettes with pure white bold text (no text clipping).
    - Interactive Canvas with Click-to-Drag Panning and Mouse-Wheel Zooming.
    - Fullscreen Maximize Mode (fills entire monitor with escape key support).
    """
    if not diagram_code:
        return
        
    cleaned = diagram_code.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:mermaid)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Ensure diagram starts with a valid directive (e.g. graph TD or mindmap)
    first_line = cleaned.split("\n")[0].strip().lower()
    valid_headers = ("graph ", "graph\n", "flowchart ", "flowchart\n", "mindmap", "sequencediagram", "classdiagram", "statediagram", "erdiagram", "gantt", "pie", "gitgraph")
    if not any(first_line.startswith(vh) for vh in valid_headers):
        if "-->" in cleaned or "---" in cleaned:
            cleaned = "graph TD\n" + cleaned
        else:
            cleaned = "mindmap\n" + cleaned

    escaped_code = html.escape(cleaned)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
            * {{
                box-sizing: border-box;
                user-select: none;
            }}
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background-color: transparent;
                color: #f1f1f1;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                overflow: hidden;
            }}
            .card {{
                background: #141414;
                border: 1px solid #282828;
                border-radius: 14px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: 100%;
                box-shadow: 0 8px 24px rgba(0,0,0,0.4);
                transition: all 0.2s ease;
            }}
            .card.fullscreen-mode {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                z-index: 999999 !important;
                border-radius: 0 !important;
                border: none !important;
                background: #0d0d0d !important;
            }}
            .toolbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #1c1c1c;
                padding: 10px 18px;
                border-bottom: 1px solid #282828;
                z-index: 10;
                gap: 12px;
            }}
            .toolbar-title {{
                font-size: 14px;
                font-weight: 700;
                color: #f3f4f6;
                display: flex;
                align-items: center;
                gap: 10px;
                white-space: nowrap;
            }}
            .hint-tag {{
                font-size: 11.5px;
                color: #9ca3af;
                background: #262626;
                padding: 3px 10px;
                border-radius: 6px;
                border: 1px solid #363636;
                white-space: nowrap;
            }}
            .btn-group {{
                display: flex;
                gap: 8px;
                align-items: center;
                white-space: nowrap;
            }}
            .btn {{
                background: #272727;
                border: 1px solid #3d3d3d;
                color: #f3f4f6;
                padding: 6px 14px;
                border-radius: 7px;
                font-size: 12.5px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s ease;
                display: flex;
                align-items: center;
                gap: 5px;
                white-space: nowrap;
            }}
            .btn:hover {{
                background: #363636;
                border-color: #ef4444;
                color: #ffffff;
            }}
            .btn.btn-fullscreen {{
                background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
                border-color: #dc2626;
                color: #ffffff;
            }}
            .btn.btn-fullscreen:hover {{
                background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
                box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
            }}
            .zoom-badge {{
                font-size: 12px;
                font-weight: 700;
                color: #e5e7eb;
                min-width: 48px;
                text-align: center;
            }}
            .canvas-area {{
                position: relative;
                width: 100%;
                flex: 1;
                overflow: hidden;
                background: radial-gradient(circle at center, #181818 0%, #0f0f0f 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                cursor: grab;
            }}
            .canvas-area.dragging {{
                cursor: grabbing;
            }}
            #zoom-target {{
                transform-origin: center center;
                transition: transform 0.05s linear;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 40px;
                pointer-events: none;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
                pointer-events: auto;
            }}
            .mermaid svg {{
                max-width: none !important;
                height: auto !important;
                font-family: 'Inter', system-ui, sans-serif !important;
                overflow: visible !important;
            }}

            /* ========================================
               HIGH-CLARITY MERMAID STYLING (MINDMAP & FLOWCHART)
               ======================================== */
            
            /* Universal High-Contrast White Text */
            svg text, 
            svg tspan, 
            .mindmap-node text, 
            .mindmap-node tspan, 
            .node text, 
            .label text,
            g[class*="mindmap"] text {{
                fill: #ffffff !important;
                color: #ffffff !important;
                font-family: 'Inter', system-ui, sans-serif !important;
                font-weight: 700 !important;
                font-size: 14.5px !important;
                letter-spacing: 0.3px !important;
                text-shadow: 0 1px 4px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.8) !important;
            }}

            /* Mindmap Node Shapes - Rounded, Bordered, Vibrant */
            .mindmap-node rect, 
            .mindmap-node circle, 
            .mindmap-node polygon {{
                stroke-width: 2.5px !important;
                filter: drop-shadow(0 4px 10px rgba(0,0,0,0.6)) !important;
                rx: 8px !important;
                ry: 8px !important;
            }}

            /* Root Center Node - Bright Crimson */
            .section-root rect, .section-root circle, .mindmap-node:first-child rect, .mindmap-node:first-child circle {{
                fill: #991b1b !important;
                stroke: #ef4444 !important;
                stroke-width: 3.5px !important;
            }}

            /* Mindmap Branch Palettes (Bright fills & distinct neon borders) */
            .section-0 rect, .section-0 circle, .section-0 polygon {{ fill: #1e3a8a !important; stroke: #60a5fa !important; }}
            .section-1 rect, .section-1 circle, .section-1 polygon {{ fill: #064e3b !important; stroke: #34d399 !important; }}
            .section-2 rect, .section-2 circle, .section-2 polygon {{ fill: #78350f !important; stroke: #fbbf24 !important; }}
            .section-3 rect, .section-3 circle, .section-3 polygon {{ fill: #581c87 !important; stroke: #c084fc !important; }}
            .section-4 rect, .section-4 circle, .section-4 polygon {{ fill: #831843 !important; stroke: #f472b6 !important; }}
            .section-5 rect, .section-5 circle, .section-5 polygon {{ fill: #134e4a !important; stroke: #2dd4bf !important; }}

            /* Remove default underline line that cuts through mindmap words */
            .mindmap-node line {{
                display: none !important;
            }}

            /* Mindmap connector curves */
            path.mindmap-edge,
            .mindmap-edge,
            path[class*="mindmap"] {{
                stroke: #cbd5e1 !important;
                stroke-width: 3px !important;
                stroke-linecap: round !important;
                opacity: 0.9 !important;
            }}

            /* Flowchart Node Styling */
            .node rect, .node circle, .node polygon {{
                fill: #1e293b !important;
                stroke: #ef4444 !important;
                stroke-width: 2.2px !important;
                rx: 8px !important;
                ry: 8px !important;
                filter: drop-shadow(0 3px 8px rgba(0,0,0,0.5)) !important;
            }}
            .edgeLabel {{
                background-color: #262626 !important;
                color: #fca5a5 !important;
                font-weight: 700 !important;
                font-size: 12.5px !important;
                padding: 4px 9px !important;
                border-radius: 5px !important;
                border: 1px solid #454545 !important;
            }}
            .edgePath path {{
                stroke: #ef4444 !important;
                stroke-width: 2.5px !important;
            }}
            .marker {{
                fill: #ef4444 !important;
                stroke: #ef4444 !important;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    darkMode: true,
                    background: '#141414',
                    primaryColor: '#b91c1c',
                    primaryTextColor: '#ffffff',
                    primaryBorderColor: '#ef4444',
                    lineColor: '#cbd5e1',
                    secondaryColor: '#1e3a8a',
                    tertiaryColor: '#064e3b',
                    nodeBorder: '#ef4444',
                    clusterBkg: '#1c1c1c',
                    clusterBorder: '#383838',
                    titleColor: '#ffffff',
                    fontFamily: 'Inter, sans-serif',
                    fontSize: '15px'
                }},
                securityLevel: 'loose'
            }});

            let scale = 1.0;
            let posX = 0;
            let posY = 0;
            let isDragging = false;
            let startX = 0;
            let startY = 0;

            function applyTransform() {{
                const target = document.getElementById('zoom-target');
                const badge = document.getElementById('zoom-val');
                if (target) {{
                    target.style.transform = `translate(${{posX}}px, ${{posY}}px) scale(${{scale}})`;
                }}
                if (badge) {{
                    badge.innerText = `${{Math.round(scale * 100)}}%`;
                }}
            }}

            function zoomIn() {{
                scale = Math.min(scale + 0.2, 3.5);
                applyTransform();
            }}

            function zoomOut() {{
                scale = Math.max(scale - 0.2, 0.4);
                applyTransform();
            }}

            function resetZoom() {{
                scale = 1.0;
                posX = 0;
                posY = 0;
                applyTransform();
            }}

            function toggleFullscreen() {{
                const card = document.getElementById('main-card');
                const btn = document.getElementById('fs-btn');
                
                if (!document.fullscreenElement && !card.classList.contains('fullscreen-mode')) {{
                    if (card.requestFullscreen) {{
                        card.requestFullscreen().catch(() => {{
                            card.classList.add('fullscreen-mode');
                        }});
                    }} else {{
                        card.classList.add('fullscreen-mode');
                    }}
                    btn.innerHTML = '🗗 Exit Fullscreen';
                }} else {{
                    if (document.exitFullscreen && document.fullscreenElement) {{
                        document.exitFullscreen().catch(() => {{}});
                    }}
                    card.classList.remove('fullscreen-mode');
                    btn.innerHTML = '⛶ Fullscreen';
                }}
            }}

            document.addEventListener('fullscreenchange', () => {{
                const card = document.getElementById('main-card');
                const btn = document.getElementById('fs-btn');
                if (!document.fullscreenElement) {{
                    card.classList.remove('fullscreen-mode');
                    btn.innerHTML = '⛶ Fullscreen';
                }} else {{
                    btn.innerHTML = '🗗 Exit Fullscreen';
                }}
            }});

            window.addEventListener('DOMContentLoaded', () => {{
                const canvas = document.getElementById('canvas-wrapper');
                
                // Pan / Drag handlers
                canvas.addEventListener('mousedown', (e) => {{
                    if (e.target.closest('.toolbar') || e.target.closest('button')) return;
                    isDragging = true;
                    startX = e.clientX - posX;
                    startY = e.clientY - posY;
                    canvas.classList.add('dragging');
                }});

                window.addEventListener('mousemove', (e) => {{
                    if (!isDragging) return;
                    posX = e.clientX - startX;
                    posY = e.clientY - startY;
                    applyTransform();
                }});

                window.addEventListener('mouseup', () => {{
                    isDragging = false;
                    canvas.classList.remove('dragging');
                }});

                // Mouse Wheel Zoom
                canvas.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    const delta = e.deltaY < 0 ? 0.12 : -0.12;
                    scale = Math.min(Math.max(scale + delta, 0.4), 3.5);
                    applyTransform();
                }}, {{ passive: false }});

                // Double click to toggle Fullscreen
                canvas.addEventListener('dblclick', (e) => {{
                    if (e.target.closest('.toolbar') || e.target.closest('button')) return;
                    toggleFullscreen();
                }});
            }});
        </script>
    </head>
    <body>
        <div class="card" id="main-card">
            <div class="toolbar">
                <div class="toolbar-title">
                    <span>🗺️ Interactive Knowledge Canvas</span>
                    <span class="hint-tag">🖱️ Drag to Pan • 📜 Scroll to Zoom • Double-Click Fullscreen</span>
                </div>
                <div class="btn-group">
                    <button class="btn" onclick="zoomOut()" title="Zoom Out">➖</button>
                    <span class="zoom-badge" id="zoom-val">100%</span>
                    <button class="btn" onclick="zoomIn()" title="Zoom In">➕</button>
                    <button class="btn" onclick="resetZoom()" title="Reset Canvas">⟲ Reset</button>
                    <button class="btn btn-fullscreen" id="fs-btn" onclick="toggleFullscreen()" title="Maximize / Fullscreen Mode">⛶ Fullscreen</button>
                </div>
            </div>
            <div class="canvas-area" id="canvas-wrapper">
                <div id="zoom-target">
                    <pre class="mermaid">
{escaped_code}
                    </pre>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    components.html(html_content, height=height, scrolling=False)



# ========================================
# UTILITY FUNCTIONS & CONFIGURATION
# ========================================

VISUALIZATION_OPTIONS = {
    "mindmap": {
        "label": "🌳 Concept Mind Map",
        "desc": "Hierarchical knowledge tree of core concepts, subtopics & terminology"
    },
    "flowchart": {
        "label": "🔄 Process Flowchart",
        "desc": "Step-by-step execution pipeline, workflow or algorithm diagram"
    }
}

SUMMARY_STYLE_OPTIONS = {
    "study_notes": {
        "label": "📚 Study Notes",
        "desc": "Structured notes with concepts & practical breakdown"
    },
    "quick_summary": {
        "label": "⚡ Quick Summary",
        "desc": "Short, punchy 2-minute overview for fast understanding"
    },
    "deep_dive": {
        "label": "🧠 Deep Dive",
        "desc": "Detailed academic guide with full context & details"
    },
    "key_points": {
        "label": "📌 Key Points",
        "desc": "Only the most essential takeaways & rules with zero filler"
    },
    "revision_notes": {
        "label": "📝 Revision Notes",
        "desc": "Exam-focused cheat sheet with definitions & self-test checkpoints"
    }
}

QUIZ_MODE_OPTIONS = {
    "quick_test": {
        "label": "⚡ Quick Test",
        "desc": "Basic comprehension (easy + medium) to quickly check understanding",
        "options": [3, 5, 7],
        "default_idx": 1
    },
    "exam_practice": {
        "label": "📝 Exam Practice",
        "desc": "Challenging exam prep focusing on definitions, comparisons & formulas",
        "options": [5, 10, 15],
        "default_idx": 1
    },
    "concept_check": {
        "label": "🧠 Concept Check",
        "desc": "Deep conceptual reasoning & causality ('Why does X happen?')",
        "options": [3, 5, 7],
        "default_idx": 1
    },
    "challenge": {
        "label": "🔥 Challenge",
        "desc": "Difficult scenario & application problem solving with trade-offs",
        "options": [3, 5, 10],
        "default_idx": 1
    }
}

def is_valid_youtube_url(url):
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def get_embed_url(vid, start_seconds=0):
    autoplay = 1 if start_seconds > 0 else 0
    return f"https://www.youtube.com/embed/{vid}?start={int(start_seconds)}&autoplay={autoplay}&rel=0"


def get_timestamp_url(vid, seconds):
    return f"https://www.youtube.com/watch?v={vid}&t={int(seconds)}s"


# ========================================
# SIDEBAR - VIDEO LIBRARY & HISTORY
# ========================================

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        '<span style="font-size:22px;">📚</span>'
        '<span style="font-size:18px;font-weight:700;color:#f1f1f1;">Video Library</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button("➕ Process New Video", key="sidebar_new_video_btn", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)

    search_query = st.text_input(
        "Search History",
        placeholder="🔍 Search title or channel...",
        label_visibility="collapsed",
        key="library_search_query"
    )

    library_videos = call_get_all_videos(limit=50)

    if search_query:
        sq = search_query.lower().strip()
        library_videos = [
            v for v in library_videos
            if sq in (v.get("metadata", {}).get("title", "").lower())
            or sq in (v.get("metadata", {}).get("channel_title", "").lower())
            or sq in (v.get("video_id", "").lower())
        ]

    st.caption(f"💾 **{len(library_videos)}** video{'s' if len(library_videos) != 1 else ''} saved in MongoDB")
    st.markdown("<hr style='border-color:#272727;margin:8px 0;'>", unsafe_allow_html=True)

    if not library_videos:
        st.markdown(
            '<div style="background:#1a1a1a;border:1px dashed #333;border-radius:10px;padding:16px;text-align:center;color:#777;font-size:12px;margin-top:10px;">'
            'No processed videos found yet.<br>Paste a YouTube URL to get started!'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        for idx, item in enumerate(library_videos):
            vid = item.get("video_id")
            meta = item.get("metadata", {})
            v_title = meta.get("title") or f"Video {vid}"
            v_channel = meta.get("channel_title") or "YouTube"
            is_active = (st.session_state.get("video_id") == vid)

            card_border = "1px solid #ff4444" if is_active else "1px solid #272727"
            card_bg = "rgba(255, 0, 0, 0.08)" if is_active else "#181818"
            active_badge = '<span style="color:#ff4444;font-size:10px;font-weight:700;float:right;">● ACTIVE</span>' if is_active else ''

            st.markdown(
                f'<div style="background:{card_bg};border:{card_border};border-radius:10px;padding:10px 12px;margin-bottom:8px;">'
                f'<div style="font-size:12px;font-weight:600;color:#f1f1f1;line-height:1.3;margin-bottom:4px;">'
                f'{v_title[:45] + ("..." if len(v_title) > 45 else "")} {active_badge}'
                f'</div>'
                f'<div style="font-size:11px;color:#888;">{v_channel[:30]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            col_open, col_del = st.columns([3.5, 1])
            with col_open:
                btn_label = "Viewing" if is_active else "📖 Open"
                if st.button(btn_label, key=f"lib_open_{vid}_{idx}", use_container_width=True, disabled=is_active):
                    # Instant zero-latency restore from MongoDB
                    cached_data = call_get_video(vid)
                    if not cached_data and not vid.startswith("upload_"):
                        cached_data = call_process_video(f"https://www.youtube.com/watch?v={vid}")

                    if cached_data:
                        st.session_state.video_data = cached_data
                        st.session_state.video_id = vid
                        st.session_state.validation_result = item.get("validation") or {"is_educational": True, "tier": 1}
                        st.session_state.chat_history = []
                        st.session_state.video_summary = None
                        st.session_state.video_summaries = {}
                        st.session_state.video_quiz = None
                        st.session_state.video_quizzes = {}
                        st.session_state.quiz_states = {}
                        st.session_state.video_visualizations = {}
                        st.session_state.current_timestamp = 0
                        st.session_state.video_processed = True
                        st.rerun()
            with col_del:
                if st.button("🗑️", key=f"lib_del_{vid}_{idx}", help=f"Delete {v_title[:25]} from history"):
                    call_delete_video(vid)
                    if st.session_state.get("video_id") == vid:
                        for k in list(st.session_state.keys()):
                            del st.session_state[k]
                    st.rerun()

            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


# ========================================
# HEADER
# ========================================

h1, h2 = st.columns([1, 8])
with h1:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#ff0000,#cc0000);color:white;'
        'padding:8px 14px;border-radius:10px;font-size:22px;font-weight:800;'
        'text-align:center;box-shadow:0 4px 15px rgba(255,0,0,0.3);">'
        '&#9654; LT</div>',
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        '<div style="padding-top:4px;">'
        '<span style="font-size:22px;font-weight:700;color:#f1f1f1;">LearnTube AI</span>'
        '<br><span style="font-size:13px;color:#717171;">AI-Powered YouTube Learning Assistant</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ========================================
# HOME SCREEN
# ========================================

if not st.session_state.video_processed:

    st.markdown(
        '<div style="text-align:center;padding:20px 0 10px 0;">'
        '<h2 style="color:#f1f1f1;">Learn from YouTube Videos or Local Media</h2>'
        '<p style="color:#717171;">Enter a YouTube link or upload a local video/audio file with on-device Whisper transcription</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        tab_yt_in, tab_upload_in = st.tabs(["🔗 YouTube URL", "📁 Upload Video / Audio (Whisper AI)"])

        with tab_yt_in:
            video_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
                key="url_input",
            )

            _, bc, _ = st.columns([1, 1, 1])
            with bc:
                process_btn = st.button("Process Video", use_container_width=True, key="btn_process_yt")

        with tab_upload_in:
            uploaded_media = st.file_uploader(
                "Upload Media File",
                type=["mp4", "mkv", "mov", "webm", "avi", "mp3", "wav", "m4a", "aac", "flac"],
                help="Supports video (.mp4, .mkv, .mov) and audio (.mp3, .wav, .m4a) files. Transcribed locally via faster-whisper on CPU.",
                key="local_media_uploader"
            )
            st.caption("🎙️ **On-Device Whisper STT:** Transcribes speech to timestamped text with zero captions needed.")

            _, bc_up, _ = st.columns([1, 1, 1])
            with bc_up:
                upload_btn = st.button("Transcribe & Process", use_container_width=True, key="btn_process_upload")

    # ---- YouTube Processing Pipeline ----
    if process_btn:
        if not video_url:
            st.warning("Please paste a YouTube URL first.")
        elif not is_valid_youtube_url(video_url):
            st.error("Invalid YouTube URL format.")
        else:
            with st.status("Processing YouTube video...", expanded=True) as status:
                try:
                    st.write("Fetching video data and transcript...")
                    video_data = call_process_video(video_url.strip())

                    if video_data is None:
                        st.error("Failed to process video. Check the URL or try again.")
                        status.update(label="Processing failed", state="error")
                        st.stop()

                    st.session_state.video_data = video_data
                    st.session_state.video_id = video_data["video_id"]
                    st.session_state.uploaded_file_bytes = None
                    st.session_state.uploaded_file_name = None
                    title = video_data["metadata"].get("title", "Unknown")
                    st.write(f"Found: {title}")

                    st.write("🔍 Validating educational content...")
                    validation = call_validate(video_data["metadata"], video_data["transcript"], video_id=video_data["video_id"])
                    st.session_state.validation_result = validation

                    tier = validation.get("tier", 1)
                    if tier == 2:
                        st.write("🤖 **Skipped Tier 1:** Title was ambiguous. Sent transcript to **Tier 2 (LLM Evaluation)**.")
                    else:
                        st.write("**Tier 1 Matched:** Fast rule-based validation applied.")

                    if not validation.get("is_educational", False):
                        call_delete_video(video_data["video_id"])
                        st.warning(f"❌ Rejected ({validation.get('tier_name', 'Validator')}): {validation.get('reason', 'Not educational')}")
                        st.session_state.video_processed = True
                        st.session_state.ingestion_result = None
                        status.update(label="Video not educational", state="error")
                        time.sleep(1)
                        st.rerun()

                    if tier == 2:
                        st.write(f"✅ Verified as educational via **Tier 2 LLM**.")
                    else:
                        st.write("✅ Verified as educational content.")

                    st.write("Chunking transcript and generating vector embeddings...")
                    ingestion = call_ingest(video_data["transcript"], video_data["video_id"])
                    st.session_state.ingestion_result = ingestion
                    chunks = ingestion.get("chunks_created", 0) if ingestion else 0
                    is_cached = ingestion.get("cached", False) if ingestion else False

                    if is_cached:
                        st.write(f"Loaded {chunks} chunks from database cache (Instant)")
                    else:
                        st.write(f"✅ Successfully indexed {chunks} timestamped chunks in ChromaDB.")

                    st.session_state.video_processed = True
                    status.update(label="Video processed successfully!", state="complete")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    status.update(label="Processing failed", state="error")

    # ---- Local File Upload & Whisper STT Pipeline ----
    if upload_btn:
        if uploaded_media is None:
            st.warning("Please select a video or audio file to upload first.")
        else:
            with st.status("Transcribing and processing media file...", expanded=True) as status:
                try:
                    file_bytes = uploaded_media.getvalue()
                    filename = uploaded_media.name
                    st.write(f"🎙️ Transcribing **{filename}** using local Whisper AI model...")

                    video_data = call_upload_video(file_bytes, filename)

                    if video_data is None:
                        st.error("Failed to transcribe media file. Please check file format.")
                        status.update(label="Transcription failed", state="error")
                        st.stop()

                    st.session_state.video_data = video_data
                    st.session_state.video_id = video_data["video_id"]
                    st.session_state.uploaded_file_bytes = file_bytes
                    st.session_state.uploaded_file_name = filename
                    
                    seg_count = len(video_data.get("transcript", []))
                    st.write(f"✅ Transcribed {seg_count} timestamped speech segments.")

                    st.write("🔍 Validating educational content...")
                    validation = call_validate(video_data["metadata"], video_data["transcript"], video_id=video_data["video_id"])
                    st.session_state.validation_result = validation

                    tier = validation.get("tier", 1)
                    if tier == 2:
                        st.write("🤖 **Skipped Tier 1:** Evaluated transcript with **Tier 2 LLM**.")
                    else:
                        st.write("**Tier 1 Matched:** Fast rule-based validation applied.")

                    if not validation.get("is_educational", False):
                        call_delete_video(video_data["video_id"])
                        st.warning(f"❌ Rejected ({validation.get('tier_name', 'Validator')}): {validation.get('reason', 'Not educational')}")
                        st.session_state.video_processed = True
                        st.session_state.ingestion_result = None
                        status.update(label="Content not educational", state="error")
                        time.sleep(1)
                        st.rerun()

                    st.write("✅ Verified as educational content.")

                    st.write("Chunking transcript and generating vector embeddings...")
                    ingestion = call_ingest(video_data["transcript"], video_data["video_id"])
                    st.session_state.ingestion_result = ingestion
                    chunks = ingestion.get("chunks_created", 0) if ingestion else 0
                    is_cached = ingestion.get("cached", False) if ingestion else False

                    if is_cached:
                        st.write(f"Loaded {chunks} chunks from database cache (Instant)")
                    else:
                        st.write(f"✅ Successfully indexed {chunks} timestamped chunks in ChromaDB.")

                    st.session_state.video_processed = True
                    status.update(label="Media transcribed & indexed successfully!", state="complete")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    status.update(label="Processing failed", state="error")

    # ---- Feature Cards (only when not processing) ----
    if not process_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)

        with f1:
            st.markdown(
                '<div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;'
                'padding:40px 24px;text-align:center;">'
                '<div style="font-size:40px;">&#128172;</div>'
                '<div style="font-size:16px;font-weight:600;color:#aaa;margin:8px 0;">Ask AI</div>'
                '<div style="font-size:13px;color:#666;">Ask any question about the video and get answers with timestamps.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with f2:
            st.markdown(
                '<div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;'
                'padding:40px 24px;text-align:center;">'
                '<div style="font-size:40px;">&#128221;</div>'
                '<div style="font-size:16px;font-weight:600;color:#aaa;margin:8px 0;">Video Summary</div>'
                '<div style="font-size:13px;color:#666;">Get a comprehensive summary of key topics covered.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with f3:
            st.markdown(
                '<div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;'
                'padding:40px 24px;text-align:center;">'
                '<div style="font-size:40px;">&#129504;</div>'
                '<div style="font-size:16px;font-weight:600;color:#aaa;margin:8px 0;">Generate Quiz</div>'
                '<div style="font-size:13px;color:#666;">Test your understanding with AI-generated questions.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ========================================
# MAIN APP (after video processed)
# ========================================

else:
    video_data = st.session_state.video_data
    validation = st.session_state.validation_result or {}
    metadata = video_data.get("metadata", {}) if video_data else {}
    video_id = st.session_state.video_id

    # ---- Non-Educational Rejection Gate ----
    if not validation.get("is_educational", False):
        st.error("This video was not identified as educational content.")
        st.markdown(
            f'<div style="background:#1a1a1a;border:1px solid #272727;border-radius:14px;padding:24px;">'
            f'<p style="color:#aaa;"><b>Title:</b> {metadata.get("title", "N/A")}</p>'
            f'<p style="color:#aaa;"><b>Validation Stage:</b> {validation.get("tier_name", "N/A")}</p>'
            f'<p style="color:#aaa;"><b>Reason:</b> {validation.get("reason", "N/A")}</p>'
            f'<p style="color:#aaa;"><b>Confidence:</b> {validation.get("confidence", 0):.0%}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Try Another Video", key="btn_try_another"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.stop()

    # Restore persistent MongoDB chat history if available
    if not st.session_state.chat_history and video_id:
        db_history = call_get_chat_history(video_id)
        if db_history:
            for item in db_history:
                if item.get("question"):
                    st.session_state.chat_history.append({"role": "user", "content": item["question"]})
                if item.get("answer"):
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": item["answer"],
                        "sources": item.get("sources", [])
                    })

    # Restore persistent MongoDB summary if available
    if not st.session_state.get("video_summary") and video_id:
        cached_sum = call_get_cached_summary(video_id)
        if cached_sum:
            st.session_state.video_summary = cached_sum

    # Restore persistent MongoDB quiz if available
    if not st.session_state.get("video_quizzes") and video_id:
        for q_style in QUIZ_MODE_OPTIONS.keys():
            cached_quiz = call_get_cached_quiz(video_id, style=q_style)
            if cached_quiz:
                st.session_state.video_quizzes[q_style] = cached_quiz
                if not st.session_state.get("video_quiz"):
                    st.session_state.video_quiz = cached_quiz

    # ---- Video Info Bar ----
    title = metadata.get("title", "Untitled")
    channel = metadata.get("channel_title", "Unknown Channel")
    chunks_created = st.session_state.ingestion_result.get("chunks_created", 0) if st.session_state.ingestion_result else 0
    snippet_count = len(video_data.get("transcript", []))
    category = validation.get("category") or "Educational"
    topic = validation.get("topic") or ""
    tier = validation.get("tier", 1)

    tier_badge = (
        '<span style="background:rgba(156,39,176,0.18);color:#ce93d8;padding:5px 12px;border-radius:20px;'
        'font-size:12px;font-weight:600;border:1px solid rgba(156,39,176,0.4);">🤖 Tier 2 (LLM Verified)</span>'
        if tier == 2
        else
        '<span style="background:rgba(0,180,0,0.15);color:#4caf50;padding:5px 12px;border-radius:20px;'
        'font-size:12px;font-weight:600;border:1px solid rgba(0,180,0,0.25);">⚡ Tier 1 (Rule-Based)</span>'
    )

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a1a,#1f1f1f);border:1px solid #272727;'
        f'border-radius:14px;padding:20px;margin-bottom:16px;">'
        f'<div style="font-size:18px;font-weight:700;color:#f1f1f1;margin-bottom:6px;">{title}</div>'
        f'<div style="font-size:14px;color:#aaa;margin-bottom:12px;">{channel}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;">'
        f'{tier_badge}'
        f'<span style="background:rgba(33,150,243,0.12);color:#42a5f5;padding:5px 12px;border-radius:20px;'
        f'font-size:12px;font-weight:500;border:1px solid rgba(33,150,243,0.2);">{chunks_created} Chunks</span>'
        f'<span style="background:rgba(33,150,243,0.12);color:#42a5f5;padding:5px 12px;border-radius:20px;'
        f'font-size:12px;font-weight:500;border:1px solid rgba(33,150,243,0.2);">{snippet_count} Snippets</span>'
        f'<span style="background:#272727;color:#ccc;padding:5px 12px;border-radius:20px;'
        f'font-size:12px;">{video_id}</span>'
        f'{"<span style=" + chr(34) + "background:#272727;color:#ccc;padding:5px 12px;border-radius:20px;font-size:12px;" + chr(34) + ">" + category + "</span>" if category else ""}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # New Video button
    _, btn_col = st.columns([7, 1])
    with btn_col:
        if st.button("New Video", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ---- Main Layout ----
    video_col, chat_col = st.columns([1.2, 1], gap="medium")

    with video_col:
        current_time = st.session_state.get("current_timestamp", 0)
        is_uploaded = metadata.get("source") == "uploaded_file" or str(video_id).startswith("upload_")
        
        if is_uploaded:
            filename = metadata.get("filename") or st.session_state.get("uploaded_file_name") or ""
            ext = os.path.splitext(filename)[1].lower() if filename else ""
            file_bytes = st.session_state.get("uploaded_file_bytes")

            if file_bytes:
                if ext in [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]:
                    st.markdown(f"##### 🎙️ Audio Track: `{filename}`")
                    st.audio(file_bytes)
                else:
                    st.markdown(f"##### 🎬 Video Track: `{filename}`")
                    st.video(file_bytes, start_time=int(current_time))
            else:
                st.markdown(
                    f'<div style="background:#141414;border:1px solid #282828;border-radius:14px;padding:36px 20px;text-align:center;">'
                    f'<div style="font-size:40px;margin-bottom:10px;">📁</div>'
                    f'<div style="font-size:16px;font-weight:700;color:#f1f1f1;margin-bottom:6px;">{metadata.get("title", "Uploaded Media")}</div>'
                    f'<div style="font-size:13px;color:#888;margin-bottom:12px;">Transcribed with Local Whisper STT Engine</div>'
                    f'<span style="background:rgba(255,0,0,0.15);color:#ff4444;padding:4px 12px;border-radius:12px;font-size:11px;font-weight:600;">Indexed & Cached in Database</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            embed_url = get_embed_url(video_id, start_seconds=current_time)
            st.markdown(
                f'<iframe width="100%" height="400" src="{embed_url}" '
                f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
                f'gyroscope; picture-in-picture" allowfullscreen '
                f'style="border-radius:14px;border:1px solid #272727;"></iframe>',
                unsafe_allow_html=True,
            )
        if current_time > 0:
            h = int(current_time) // 3600
            m = (int(current_time) % 3600) // 60
            s = int(current_time) % 60
            ts_str = f"{h:02d}:{m:02d}:{s:02d}"
            info_col1, info_col2 = st.columns([3, 1])
            with info_col1:
                st.caption(f"⏱️ Playing from **{ts_str}**")
            with info_col2:
                if st.button("⏪ Start", key="reset_player_btn", use_container_width=True):
                    st.session_state.current_timestamp = 0
                    st.rerun()

    with chat_col:
        tab_ask, tab_summary, tab_quiz = st.tabs(["Ask AI", "Summary", "Quiz"])

        # ===== ASK AI =====
        with tab_ask:
            c_head1, c_head2 = st.columns([1.4, 1.1])
            with c_head1:
                st.caption("Ask anything about this video with timestamps.")
            with c_head2:
                use_agent_mode = st.toggle("🤖 Agent Mode", value=True, help="Powered by LangGraph (Router ➔ RAG Specialist ➔ Tutor ➔ Guardrail)")

            chat_box = st.container(height=340, border=False)

            with chat_box:
                if not st.session_state.chat_history:
                    st.info("Ask anything about this video! e.g. 'What are the main topics discussed?'")
                else:
                    for msg_idx, msg in enumerate(st.session_state.chat_history):
                        role = "user" if msg["role"] == "user" else "assistant"
                        with st.chat_message(role):
                            st.write(msg["content"])

                            # Show step-by-step agent thought logs if available
                            if msg.get("thought_steps"):
                                with st.expander("🔍 Agent Thought & Reasoning Steps", expanded=False):
                                    for t_step in msg["thought_steps"]:
                                        st.markdown(f"**{t_step.get('agent_name', 'Agent')}** — `{t_step.get('action', '')}`")
                                        if t_step.get("detail"):
                                            st.caption(f"_{t_step.get('detail', '')}_")

                            if msg.get("sources"):
                                cols = st.columns(len(msg["sources"]))
                                for i, src in enumerate(msg["sources"]):
                                    start_sec = int(src.get("start_time", 0))
                                    btn_label = f"▶ {src['formatted_start']}"
                                    with cols[i]:
                                        if st.button(btn_label, key=f"ts_btn_{msg_idx}_{i}_{start_sec}", help=f"Jump player to {src['formatted_start']}"):
                                            st.session_state.current_timestamp = start_sec
                                            st.rerun()

            question = st.chat_input("Ask a question about this video...")

            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})

                try:
                    clean_history = [
                        {"role": m.get("role", "user"), "content": m.get("content", "")}
                        for m in st.session_state.chat_history
                        if m.get("content")
                    ]
                    if use_agent_mode:
                        with st.spinner("Multi-Agent reasoning (Router ➔ RAG Specialist ➔ Tutor ➔ Guardrail)..."):
                            result = call_agent_answer(
                                question=question,
                                video_id=video_id,
                                chat_history=clean_history,
                                metadata=metadata
                            )
                    else:
                        result = call_answer(question, video_id, metadata=metadata)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", "I couldn't generate an answer."),
                        "sources": result.get("sources", []),
                        "thought_steps": result.get("thought_steps", [])
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "sources": [],
                        "thought_steps": []
                    })

                st.rerun()

        # ===== SUMMARY =====
        with tab_summary:
            st.markdown("<p style='font-size:14px;font-weight:600;color:#f1f1f1;margin-bottom:6px;'>How would you like your summary?</p>", unsafe_allow_html=True)

            style_keys = list(SUMMARY_STYLE_OPTIONS.keys())
            style_labels = [SUMMARY_STYLE_OPTIONS[k]["label"] for k in style_keys]

            selected_label = st.selectbox(
                "Summary Style",
                options=style_labels,
                index=0,
                key="summary_style_selector",
                label_visibility="collapsed"
            )

            # Map selected label back to key
            selected_style_key = "study_notes"
            for k, val in SUMMARY_STYLE_OPTIONS.items():
                if val["label"] == selected_label:
                    selected_style_key = k
                    break

            style_info = SUMMARY_STYLE_OPTIONS[selected_style_key]
            st.caption(f"ℹ️ {style_info['desc']}")

            # Check if this style is in session state or MongoDB
            current_style_summary = st.session_state.video_summaries.get(selected_style_key)
            if not current_style_summary:
                cached_db_sum = call_get_cached_summary(video_id, style=selected_style_key)
                if cached_db_sum:
                    st.session_state.video_summaries[selected_style_key] = cached_db_sum
                    current_style_summary = cached_db_sum

            if current_style_summary:
                st.markdown(
                    f'<div style="background:#1a1a1a;border:1px solid #272727;border-radius:14px;padding:24px;line-height:1.7;margin-top:12px;">'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(current_style_summary)
                st.markdown("<br>", unsafe_allow_html=True)

                btn_col1, btn_col2 = st.columns([1.2, 1])
                with btn_col1:
                    if st.button(f"🔄 Regenerate {style_info['label']}", key=f"regen_{selected_style_key}_btn", use_container_width=True):
                        with st.spinner(f"Regenerating {style_info['label']} with LangChain..."):
                            summary_text = call_summary(
                                video_data.get("transcript", []),
                                metadata=metadata,
                                video_id=video_id,
                                style=selected_style_key,
                                force_regenerate=True
                            )
                            st.session_state.video_summaries[selected_style_key] = summary_text
                            st.rerun()

                with btn_col2:
                    try:
                        pdf_bytes = generate_summary_pdf(
                            summary_text=current_style_summary,
                            metadata=metadata,
                            style_label=style_info['label']
                        )
                        raw_title = metadata.get("title", "Summary") if metadata else "Summary"
                        safe_title = re.sub(r'[^\w\-_\. ]', '_', raw_title)[:35].strip()
                        file_name = f"{safe_title}_{selected_style_key}.pdf"

                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=file_name,
                            mime="application/pdf",
                            key=f"dl_pdf_{selected_style_key}",
                            use_container_width=True
                        )
                    except Exception as pdf_err:
                        st.caption(f"⚠️ PDF generation notice: {pdf_err}")
            else:
                icon = style_info['label'].split()[0]
                st.markdown(f"""
                <div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;padding:30px;text-align:center;margin:16px 0;">
                    <div style="font-size:36px;">{icon}</div>
                    <div style="font-size:16px;font-weight:600;color:#f1f1f1;margin:8px 0;">Generate {style_info['label']}</div>
                    <div style="font-size:13px;color:#717171;">{style_info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Generate {style_info['label']}", key=f"gen_{selected_style_key}_btn", use_container_width=True):
                    with st.spinner(f"Generating {style_info['label']} with LangChain..."):
                        summary_text = call_summary(
                            video_data.get("transcript", []),
                            metadata=metadata,
                            video_id=video_id,
                            style=selected_style_key
                        )
                        st.session_state.video_summaries[selected_style_key] = summary_text
                        st.rerun()

        # ===== QUIZ =====
        with tab_quiz:
            st.markdown("<p style='font-size:14px;font-weight:600;color:#f1f1f1;margin-bottom:6px;'>Choose Quiz Mode:</p>", unsafe_allow_html=True)

            quiz_keys = list(QUIZ_MODE_OPTIONS.keys())
            quiz_labels = [QUIZ_MODE_OPTIONS[k]["label"] for k in quiz_keys]

            selected_quiz_label = st.selectbox(
                "Quiz Mode",
                options=quiz_labels,
                index=0,
                key="quiz_mode_selector",
                label_visibility="collapsed"
            )

            # Map selected label back to key
            selected_quiz_key = "quick_test"
            for k, val in QUIZ_MODE_OPTIONS.items():
                if val["label"] == selected_quiz_label:
                    selected_quiz_key = k
                    break

            quiz_info = QUIZ_MODE_OPTIONS[selected_quiz_key]
            st.caption(f"ℹ️ {quiz_info['desc']}")

            # Initialize state for this quiz style
            if selected_quiz_key not in st.session_state.quiz_states:
                st.session_state.quiz_states[selected_quiz_key] = {
                    "submitted": False,
                    "answers": {}
                }

            # Check if quiz exists in session or MongoDB
            current_quiz = st.session_state.video_quizzes.get(selected_quiz_key)
            if not current_quiz:
                cached_db_quiz = call_get_cached_quiz(video_id, style=selected_quiz_key)
                if cached_db_quiz:
                    st.session_state.video_quizzes[selected_quiz_key] = cached_db_quiz
                    current_quiz = cached_db_quiz

            if not current_quiz:
                icon = quiz_info["label"].split()[0]
                st.markdown(f"""
                <div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;padding:26px;text-align:center;margin:16px 0;">
                    <div style="font-size:36px;">{icon}</div>
                    <div style="font-size:16px;font-weight:600;color:#f1f1f1;margin:8px 0;">Generate {quiz_info['label']}</div>
                    <div style="font-size:13px;color:#717171;">{quiz_info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                q_col1, q_col2 = st.columns([1, 1.4])
                with q_col1:
                    q_count = st.selectbox(
                        "Questions Count:",
                        options=quiz_info["options"],
                        index=quiz_info["default_idx"],
                        key=f"qcount_select_{selected_quiz_key}"
                    )
                with q_col2:
                    st.write("")
                    st.write("")
                    gen_clicked = st.button(f"🧠 Generate {quiz_info['label']}", key=f"gen_{selected_quiz_key}_btn", use_container_width=True)

                if gen_clicked:
                    with st.spinner(f"Generating {q_count} questions for {quiz_info['label']} with LangChain..."):
                        quiz_data = call_quiz(
                            video_data.get("transcript", []),
                            metadata=metadata,
                            num_questions=q_count,
                            video_id=video_id,
                            style=selected_quiz_key
                        )
                        if quiz_data:
                            st.session_state.video_quizzes[selected_quiz_key] = quiz_data
                            st.session_state.quiz_states[selected_quiz_key] = {"submitted": False, "answers": {}}
                            st.rerun()
                        else:
                            st.error("Could not generate quiz. Please check backend connection and try again.")
            else:
                q_state = st.session_state.quiz_states[selected_quiz_key]

                # If submitted, calculate and display results
                if q_state["submitted"]:
                    score = 0
                    for q in current_quiz:
                        user_ans = q_state["answers"].get(q.get("id"))
                        if user_ans == q.get("correct_answer"):
                            score += 1

                    pct = int((score / len(current_quiz)) * 100) if current_quiz else 0
                    emoji = "🎉" if pct >= 75 else "👍" if pct >= 50 else "📚"

                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1f1f1f,#272727);border:1px solid #333;border-radius:14px;padding:20px;text-align:center;margin-bottom:20px;">
                        <div style="font-size:24px;font-weight:700;color:#f1f1f1;">{emoji} Score: {score} / {len(current_quiz)} ({pct}%)</div>
                        <div style="font-size:13px;color:#aaa;margin-top:4px;"><b>{quiz_info['label']}</b> &bull; Review your answers and explanations below.</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for i, q in enumerate(current_quiz, 1):
                        user_ans = q_state["answers"].get(q.get("id"))
                        is_correct = user_ans == q.get("correct_answer")
                        border_color = "rgba(0,180,0,0.4)" if is_correct else "rgba(255,0,0,0.4)"
                        status_badge = '<span style="color:#4caf50;font-weight:700;">✓ Correct</span>' if is_correct else '<span style="color:#f44336;font-weight:700;">✗ Incorrect</span>'

                        st.markdown(f"""
                        <div style="background:#1a1a1a;border:1px solid {border_color};border-radius:12px;padding:18px;margin-bottom:14px;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                                <span style="font-weight:600;color:#f1f1f1;">Question {i}</span>
                                {status_badge}
                            </div>
                            <div style="font-size:15px;color:#ddd;margin-bottom:12px;">{q.get('question')}</div>
                            <div style="font-size:13px;color:#aaa;margin-bottom:4px;"><b>Your Answer:</b> {user_ans or 'Not answered'}</div>
                            <div style="font-size:13px;color:#4caf50;margin-bottom:8px;"><b>Correct Answer:</b> {q.get('correct_answer')}</div>
                            <div style="font-size:12px;color:#888;background:#141414;padding:8px 12px;border-radius:6px;margin-top:6px;">💡 <b>Explanation:</b> {q.get('explanation')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    btn_c1, btn_c2, btn_c3 = st.columns([1, 0.8, 1.4])
                    with btn_c1:
                        if st.button("🔁 Retake Quiz", key=f"retake_{selected_quiz_key}_btn", use_container_width=True):
                            st.session_state.quiz_states[selected_quiz_key]["submitted"] = False
                            st.session_state.quiz_states[selected_quiz_key]["answers"] = {}
                            st.rerun()

                    with btn_c2:
                        cur_len = len(current_quiz)
                        def_idx = quiz_info["options"].index(cur_len) if cur_len in quiz_info["options"] else quiz_info["default_idx"]
                        regen_count = st.selectbox(
                            "Questions:",
                            options=quiz_info["options"],
                            index=def_idx,
                            key=f"regen_q_count_{selected_quiz_key}",
                            label_visibility="collapsed"
                        )

                    with btn_c3:
                        if st.button(f"✨ New ({regen_count} Qs)", key=f"regen_{selected_quiz_key}_btn", use_container_width=True):
                            with st.spinner(f"Generating {regen_count} new questions for {quiz_info['label']}..."):
                                quiz_data = call_quiz(
                                    video_data.get("transcript", []),
                                    metadata=metadata,
                                    num_questions=regen_count,
                                    video_id=video_id,
                                    style=selected_quiz_key,
                                    force_regenerate=True
                                )
                                if quiz_data:
                                    st.session_state.video_quizzes[selected_quiz_key] = quiz_data
                                    st.session_state.quiz_states[selected_quiz_key] = {"submitted": False, "answers": {}}
                                    st.rerun()

                else:
                    # Quiz in progress
                    st.markdown(f"<p style='color:#aaa;font-size:14px;'>Answer the {len(current_quiz)} questions below for <b>{quiz_info['label']}</b>:</p>", unsafe_allow_html=True)

                    with st.form(f"quiz_form_{selected_quiz_key}"):
                        form_answers = {}
                        for i, q in enumerate(current_quiz, 1):
                            st.markdown(f"**Q{i}. {q.get('question')}**")
                            selected = st.radio(
                                f"Options for Q{i}",
                                options=q.get("options", []),
                                index=None,
                                key=f"quiz_opt_{selected_quiz_key}_{q.get('id')}",
                                label_visibility="collapsed"
                            )
                            form_answers[q.get("id")] = selected
                            st.markdown("<hr style='border-color:#272727;'>", unsafe_allow_html=True)

                        submitted = st.form_submit_button(f"Submit {quiz_info['label']}", use_container_width=True)
                        if submitted:
                            st.session_state.quiz_states[selected_quiz_key]["answers"] = form_answers
                            st.session_state.quiz_states[selected_quiz_key]["submitted"] = True
                            st.rerun()

                    # Option to re-roll questions before submission
                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    with st.expander("⚙️ Change Question Count / Re-roll"):
                        rc1, rc2 = st.columns([1, 1.5])
                        with rc1:
                            cur_len = len(current_quiz)
                            def_idx = quiz_info["options"].index(cur_len) if cur_len in quiz_info["options"] else quiz_info["default_idx"]
                            pre_regen_count = st.selectbox(
                                "Questions:",
                                options=quiz_info["options"],
                                index=def_idx,
                                key=f"pre_regen_q_count_{selected_quiz_key}"
                            )
                        with rc2:
                            st.write("")
                            st.write("")
                            if st.button("✨ Regenerate Questions", key=f"pre_regen_btn_{selected_quiz_key}", use_container_width=True):
                                with st.spinner(f"Generating {pre_regen_count} questions for {quiz_info['label']}..."):
                                    quiz_data = call_quiz(
                                        video_data.get("transcript", []),
                                        metadata=metadata,
                                        num_questions=pre_regen_count,
                                        video_id=video_id,
                                        style=selected_quiz_key,
                                        force_regenerate=True
                                    )
                                    if quiz_data:
                                        st.session_state.video_quizzes[selected_quiz_key] = quiz_data
                                        st.session_state.quiz_states[selected_quiz_key] = {"submitted": False, "answers": {}}
                                        st.rerun()

    # ========================================
    # FULL-WIDTH VISUAL LEARNING STUDIO (Below Video & Chat)
    # ========================================
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1c1c1c 0%,#161616 100%);border:1px solid #2d2d2d;'
        'border-radius:14px;padding:16px 22px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<span style="font-size:20px;font-weight:800;color:#f1f1f1;">🗺️ Visual Learning Studio</span>'
        '<span style="font-size:13px;color:#888;border-left:1px solid #333;padding-left:12px;">AI-Generated Concept Maps & System Architecture Flowcharts</span>'
        '</div>'
        '<span style="background:rgba(239,68,68,0.12);color:#ef4444;padding:5px 14px;border-radius:20px;'
        'font-size:12px;font-weight:700;border:1px solid rgba(239,68,68,0.3);">⚡ 100% Full-Width Canvas</span>'
        '</div>',
        unsafe_allow_html=True
    )

    v_ctrl1, v_ctrl2 = st.columns([3, 1.2])
    with v_ctrl1:
        visual_keys = list(VISUALIZATION_OPTIONS.keys())
        visual_labels = [VISUALIZATION_OPTIONS[k]["label"] for k in visual_keys]
        selected_visual_label = st.selectbox(
            "Diagram Type",
            options=visual_labels,
            index=0,
            key="visual_type_selector_main",
            label_visibility="collapsed"
        )
        selected_visual_key = "mindmap"
        for k, val in VISUALIZATION_OPTIONS.items():
            if val["label"] == selected_visual_label:
                selected_visual_key = k
                break
        visual_info = VISUALIZATION_OPTIONS[selected_visual_key]

    with v_ctrl2:
        current_diagram = st.session_state.video_visualizations.get(selected_visual_key)
        if not current_diagram:
            cached_db_diagram = call_get_cached_visualization(video_id, visual_type=selected_visual_key)
            if cached_db_diagram:
                st.session_state.video_visualizations[selected_visual_key] = cached_db_diagram
                current_diagram = cached_db_diagram

        if current_diagram and not current_diagram.startswith("Error:"):
            if st.button(f"🔄 Regenerate {visual_info['label']}", key=f"regen_vis_main_{selected_visual_key}_btn", use_container_width=True):
                with st.spinner(f"Regenerating {visual_info['label']} with LangChain..."):
                    diag = call_visualization(
                        video_data.get("transcript", []),
                        metadata=metadata,
                        video_id=video_id,
                        visual_type=selected_visual_key,
                        force_regenerate=True
                    )
                    if diag:
                        st.session_state.video_visualizations[selected_visual_key] = diag
                        st.rerun()

    st.caption(f"💡 **{visual_info['label']}:** {visual_info['desc']}")

    if current_diagram and not current_diagram.startswith("Error:"):
        # Full-width high-clarity responsive diagram
        render_mermaid(current_diagram, height=650)
        
        with st.expander("📋 View Mermaid Diagram Code"):
            st.code(current_diagram, language="mermaid")
    else:
        icon = visual_info.get("label", "").split()[0]
        st.markdown(f"""
        <div style="background:#181818;border:1px dashed #3a3a3a;border-radius:14px;padding:36px 20px;text-align:center;margin:12px 0;">
            <div style="font-size:38px;">{icon}</div>
            <div style="font-size:16px;font-weight:700;color:#f1f1f1;margin:8px 0;">Generate {visual_info['label']}</div>
            <div style="font-size:13px;color:#888;max-width:600px;margin:0 auto 16px auto;">
                Transform this lecture transcript into a wide, interactive visual map with hierarchical concept nodes and relationship flows.
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, gen_btn_col, _ = st.columns([1, 1.5, 1])
        with gen_btn_col:
            if st.button(f"🎨 Generate {visual_info['label']}", key=f"gen_vis_main_{selected_visual_key}_btn", use_container_width=True):
                with st.spinner(f"Architecting {visual_info['label']} with LangChain..."):
                    diag = call_visualization(
                        video_data.get("transcript", []),
                        metadata=metadata,
                        video_id=video_id,
                        visual_type=selected_visual_key
                    )
                    if diag and not diag.startswith("Error:"):
                        st.session_state.video_visualizations[selected_visual_key] = diag
                        st.rerun()
                    else:
                        st.error(f"Failed to generate diagram: {diag}")

