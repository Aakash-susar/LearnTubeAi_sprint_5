import streamlit as st
import re
import time
import sys
import io

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
if "video_quiz" not in st.session_state:
    st.session_state.video_quiz = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "current_timestamp" not in st.session_state:
    st.session_state.current_timestamp = 0


# ========================================
# LAZY BACKEND IMPORTS
# ========================================

def call_process_video(url):
    """Import and call process_youtube_video with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from src.youtube.pipeline import process_youtube_video
        result = process_youtube_video(url)
    finally:
        sys.stdout = old_stdout
    return result


def call_validate(metadata, transcript):
    """Import and call validate_educational_content with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from src.Validation.educational_validator import validate_educational_content
        result = validate_educational_content(metadata, transcript)
    finally:
        sys.stdout = old_stdout
    return result


def call_ingest(transcript, video_id):
    """Import and call ingest_transcript with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from src.RAG.rag_pipeline import ingest_transcript
        result = ingest_transcript(transcript=transcript, video_id=video_id)
    finally:
        sys.stdout = old_stdout
    return result


def call_answer(question, video_id, metadata=None):
    """Import and call answer_question with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from src.RAG.rag_pipeline import answer_question
        result = answer_question(question=question, video_id=video_id, k=5, metadata=metadata)
    finally:
        sys.stdout = old_stdout
    return result


def call_summary(transcript, metadata=None):
    """Import and call generate_video_summary with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from src.RAG.rag_pipeline import generate_video_summary
        result = generate_video_summary(transcript=transcript, metadata=metadata)
    finally:
        sys.stdout = old_stdout
    return result


def call_quiz(transcript, metadata=None, num_questions=4):
    """Import and call generate_video_quiz with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from src.RAG.rag_pipeline import generate_video_quiz
        result = generate_video_quiz(transcript=transcript, metadata=metadata, num_questions=num_questions)
    finally:
        sys.stdout = old_stdout
    return result


# ========================================
# UTILITY FUNCTIONS
# ========================================

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
        '<h2 style="color:#f1f1f1;">Learn from any YouTube video</h2>'
        '<p style="color:#717171;">Paste a YouTube URL below and start asking questions with AI</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        video_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="url_input",
        )

        _, bc, _ = st.columns([1, 1, 1])
        with bc:
            process_btn = st.button("Process Video", use_container_width=True)

    # ---- Processing Pipeline ----
    if process_btn:
        if not video_url:
            st.warning("Please paste a YouTube URL first.")
        elif not is_valid_youtube_url(video_url):
            st.error("Invalid YouTube URL format.")
        else:
            with st.status("Processing video...", expanded=True) as status:
                try:
                    st.write("Fetching video data and transcript...")
                    video_data = call_process_video(video_url.strip())

                    if video_data is None:
                        st.error("Failed to process video. Check the URL or try again.")
                        status.update(label="Processing failed", state="error")
                        st.stop()

                    st.session_state.video_data = video_data
                    st.session_state.video_id = video_data["video_id"]
                    title = video_data["metadata"].get("title", "Unknown")
                    st.write(f"Found: {title}")

                    st.write("🔍 Validating educational content...")
                    validation = call_validate(video_data["metadata"], video_data["transcript"])
                    st.session_state.validation_result = validation

                    tier = validation.get("tier", 1)
                    if tier == 2:
                        st.write("🤖 **Skipped Tier 1:** Title was ambiguous. Sent transcript to **Tier 2 (LLM Evaluation)**.")
                    else:
                        st.write("**Tier 1 Matched:** Fast rule-based validation applied.")

                    if not validation.get("is_educational", False):
                        st.warning(f"❌ Rejected ({validation.get('tier_name', 'Validator')}): {validation.get('reason', 'Not educational')}")
                        st.session_state.video_processed = True
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
    validation = st.session_state.validation_result
    metadata = video_data["metadata"]
    video_id = st.session_state.video_id

    # ---- Non-Educational Rejection ----
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
        if st.button("Try Another Video"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.stop()

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
            chat_box = st.container(height=360, border=False)

            with chat_box:
                if not st.session_state.chat_history:
                    st.info("Ask anything about this video! e.g. 'What are the main topics discussed?'")
                else:
                    for msg_idx, msg in enumerate(st.session_state.chat_history):
                        role = "user" if msg["role"] == "user" else "assistant"
                        with st.chat_message(role):
                            st.write(msg["content"])
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
                    result = call_answer(question, video_id, metadata=metadata)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", "I couldn't generate an answer."),
                        "sources": result.get("sources", []),
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "sources": [],
                    })

                st.rerun()

        # ===== SUMMARY =====
        with tab_summary:
            if st.session_state.get("video_summary"):
                st.markdown(
                    f'<div style="background:#1a1a1a;border:1px solid #272727;border-radius:14px;padding:24px;line-height:1.7;">'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(st.session_state.video_summary)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Regenerate Summary", key="regen_sum_btn"):
                    st.session_state.video_summary = None
                    st.rerun()
            else:
                st.markdown("""
                <div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;padding:30px;text-align:center;margin-bottom:16px;">
                    <div style="font-size:36px;">&#128221;</div>
                    <div style="font-size:16px;font-weight:600;color:#f1f1f1;margin:8px 0;">Generate Comprehensive Study Notes</div>
                    <div style="font-size:13px;color:#717171;">AI will analyze the entire video transcript and generate structured notes with key takeaways and practical steps.</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Generate Summary", key="sum_btn", use_container_width=True):
                    with st.spinner("Analyzing full transcript and generating study notes with LangChain..."):
                        summary_text = call_summary(video_data.get("transcript", []), metadata=metadata)
                        st.session_state.video_summary = summary_text
                        st.rerun()

        # ===== QUIZ =====
        with tab_quiz:
            if not st.session_state.get("video_quiz"):
                st.markdown("""
                <div style="background:#1a1a1a;border:1px dashed #3a3a3a;border-radius:14px;padding:26px;text-align:center;margin-bottom:16px;">
                    <div style="font-size:36px;">&#129504;</div>
                    <div style="font-size:16px;font-weight:600;color:#f1f1f1;margin:8px 0;">Test Your Understanding</div>
                    <div style="font-size:13px;color:#717171;">Choose how many questions you want and generate an interactive quiz from this video.</div>
                </div>
                """, unsafe_allow_html=True)

                q_col1, q_col2 = st.columns([1, 1.2])
                with q_col1:
                    num_questions = st.selectbox(
                        "Questions Count:",
                        options=[3, 5, 10],
                        index=1,
                        key="num_quiz_questions"
                    )
                with q_col2:
                    st.write("")
                    st.write("")
                    gen_clicked = st.button("🧠 Generate Quiz", key="gen_quiz_btn", use_container_width=True)

                if gen_clicked:
                    with st.spinner(f"Generating {num_questions} quiz questions with LangChain..."):
                        quiz_data = call_quiz(
                            video_data.get("transcript", []),
                            metadata=metadata,
                            num_questions=num_questions
                        )
                        if quiz_data:
                            st.session_state.video_quiz = quiz_data
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_user_answers = {}
                        else:
                            st.error("Could not generate quiz. Please try again.")
                        st.rerun()
            else:
                quiz = st.session_state.video_quiz

                # If submitted, calculate and display results
                if st.session_state.quiz_submitted:
                    score = 0
                    for q in quiz:
                        user_ans = st.session_state.quiz_user_answers.get(q.get("id"))
                        if user_ans == q.get("correct_answer"):
                            score += 1

                    pct = int((score / len(quiz)) * 100) if quiz else 0
                    emoji = "🎉" if pct >= 75 else "👍" if pct >= 50 else "📚"

                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1f1f1f,#272727);border:1px solid #333;border-radius:14px;padding:20px;text-align:center;margin-bottom:20px;">
                        <div style="font-size:24px;font-weight:700;color:#f1f1f1;">{emoji} Score: {score} / {len(quiz)} ({pct}%)</div>
                        <div style="font-size:13px;color:#aaa;margin-top:4px;">Review your answers below with explanations from the video.</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for i, q in enumerate(quiz, 1):
                        user_ans = st.session_state.quiz_user_answers.get(q.get("id"))
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

                    if st.button("🔄 Retake / Generate New Quiz", key="new_quiz_btn", use_container_width=True):
                        st.session_state.video_quiz = None
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_user_answers = {}
                        st.rerun()

                else:
                    # Quiz in progress
                    st.markdown("<p style='color:#aaa;font-size:14px;'>Answer the questions below based on the video:</p>", unsafe_allow_html=True)

                    with st.form("quiz_form"):
                        for i, q in enumerate(quiz, 1):
                            st.markdown(f"**Q{i}. {q.get('question')}**")
                            selected = st.radio(
                                f"Options for Q{i}",
                                options=q.get("options", []),
                                index=None,
                                key=f"quiz_opt_{q.get('id')}",
                                label_visibility="collapsed"
                            )
                            st.session_state.quiz_user_answers[q.get("id")] = selected
                            st.markdown("<hr style='border-color:#272727;'>", unsafe_allow_html=True)

                        submitted = st.form_submit_button("Submit Quiz", use_container_width=True)
                        if submitted:
                            st.session_state.quiz_submitted = True
                            st.rerun()
