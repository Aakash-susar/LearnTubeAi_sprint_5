# 🎓 LearnTube AI — YouTube Educational Learning & RAG Assistant

**LearnTube AI** is an intelligent educational assistant that transforms YouTube videos into interactive learning environments. It validates whether a video is educational, indexes its transcript using vector embeddings into ChromaDB, and allows learners to ask questions (with exact clickable video timestamps), generate structured study notes, and take interactive quizzes.

---

## 🎯 Project Objectives

1. **Active Learning from Video Content**: Enable students and professionals to query long educational videos and get precise answers without watching hours of footage.
2. **Exact Timestamp Citations**: Every generated answer links back to the exact video timestamps (`HH:MM:SS`) where the concept was taught.
3. **Educational Content Gatekeeping**: Automatically filter and reject non-educational videos (e.g., entertainment, movie trailers, motivational rants, vlogs) before ingestion.
4. **AI-Powered Study Notes & Quizzes**: Use LangChain LCEL pipelines to generate comprehensive chapter summaries and interactive self-assessment quizzes.

---

## 🏗️ Architecture & Flow Diagram

```text
                           ┌───────────────────────────┐
                           │      YouTube URL Input    │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │  Metadata & Transcript Extraction      │
                    │  • YouTube Data API v3 (Metadata)      │
                    │  • youtube-transcript-api (Timestamps) │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │    2-Tier Educational Validator        │
                    │  Tier 1: Fast Rule & Keyword Matching  │
                    │  Tier 2: OpenAI gpt-4o-mini Validator  │
                    └──────────────┬─────────────────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
        [Rejected: Non-Educational]      [Accepted: Educational]
                   │                               │
                   ▼                               ▼
       Displays Rejection Details     ┌────────────────────────────────────────┐
       (e.g., Entertainment/Vlog)     │      Timestamped Text Chunking         │
                                      │   • RecursiveCharacterTextSplitter     │
                                      │   • Preserves start_time & end_time    │
                                      └────────────────────┬───────────────────┘
                                                           │
                                                           ▼
                                      ┌────────────────────────────────────────┐
                                      │  Multilingual Sentence Embeddings      │
                                      │  • paraphrase-multilingual-MiniLM-L12  │
                                      └────────────────────┬───────────────────┘
                                                           │
                                                           ▼
                                      ┌────────────────────────────────────────┐
                                      │        ChromaDB Vector Store           │
                                      │        (Filtered by video_id)          │
                                      └────────────────────┬───────────────────┘
                                                           │
        ┌──────────────────────────────────────────────────┴──────────────────────────────────────────────────┐
        │                                                  │                                                  │
        ▼                                                  ▼                                                  ▼
┌───────────────────────────────┐        ┌───────────────────────────────────┐      ┌───────────────────────────────────┐
│     💬 Ask AI (RAG Q&A)       │        │        📝 Video Summary           │      │        🧠 Interactive Quiz        │
│ • Distance-filtered search    │        │ • LangChain LCEL Pipeline         │      │ • LangChain LCEL Pipeline         │
│ • Video metadata injection    │        │ • PromptTemplate | ChatOpenAI     │      │ • PromptTemplate | ChatOpenAI     │
│ • OpenAI gpt-4o-mini response │        │   | StrOutputParser               │      │   | JsonOutputParser              │
│ • Clickable timestamp chips   │        │ • Structured Study Notes          │      │ • 4 MCQs with real-time scoring   │
└───────────────┬───────────────┘        └─────────────────┬─────────────────┘      └─────────────────┬─────────────────┘
                │                                          │                                          │
                └──────────────────────────────────────────┴──────────────────────────────────────────┘
                                                           │
                                                           ▼
                                         ┌───────────────────────────────────┐
                                         │       Streamlit Dark-Mode UI      │
                                         │  (Embedded Player + AI Workspace) │
                                         └───────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | **Streamlit** | Dark YouTube-inspired user interface with embedded video player, chat history, and interactive tabs. |
| **LLM & Generative AI** | **OpenAI (`gpt-4o-mini`)** | Fast, high-accuracy model for Question Answering, Video Summarization, and Quiz generation. |
| **Orchestration & Chains** | **LangChain (LCEL)** | Modern declarative pipelines (`PromptTemplate`, `StrOutputParser`, `JsonOutputParser`). |
| **Text Chunking** | **LangChain Text Splitters** | `RecursiveCharacterTextSplitter` configured for 1000-character chunks with timestamp preservation. |
| **Vector Database** | **ChromaDB (`langchain-chroma`)** | Local, persistent vector store for similarity search filtered by `video_id`. |
| **Embedding Model** | **Hugging Face (`sentence-transformers`)** | `paraphrase-multilingual-MiniLM-L12-v2` dense embeddings (384 dimensions). |
| **Data Extraction** | **YouTube Data API v3 & `youtube-transcript-api`** | Fetches video details, channel metadata, and timed transcript snippets. |

---

## ✨ Key Features

1. **🎬 Smart Video Ingestion**:
   - Paste any YouTube link. The system extracts title, channel, description, and timed captions.
2. **🛡️ Automated Educational Validation**:
   - Classifies whether the video is academic/technical vs. entertainment/motivational.
3. **💬 RAG Question Answering with Timestamp Citations**:
   - Ask specific questions about video topics.
   - Clickable red timestamp badges (e.g., `▶ 00:15:58 - 00:17:09`) open YouTube at the exact second the concept is discussed.
4. **📝 Structured Study Notes (LCEL)**:
   - Summarizes long videos into Executive Overview, Key Takeaways, Step-by-Step implementation guides, and Next Steps.
5. **🧠 Interactive Self-Assessment Quiz (LCEL)**:
   - Generates 4 conceptual multiple-choice questions with real-time scoring, answer badges, and explanations.

---

## 📁 Project Structure

```text
Sprint 4/
├── app.py                     # Streamlit Dark UI (Video Player, Chat, Summary, Quiz)
├── main.py                    # Terminal CLI for testing the RAG pipeline
├── requirements.txt           # Python dependencies
├── .env                       # API keys configuration
├── .streamlit/
│   └── config.toml            # Streamlit server & watcher configuration
├── src/
│   ├── youtube/
│   │   ├── video_id.py        # YouTube URL parsing and Video ID extractor
│   │   ├── metadata.py        # YouTube Data API v3 metadata fetcher
│   │   ├── transcript.py      # Transcript and subtitle retrieval
│   │   └── pipeline.py        # Unified YouTube processing pipeline
│   ├── Validation/
│   │   └── educational_validator.py # 2-Tier educational content classifier
│   └── RAG/
│       ├── embeddings.py      # SentenceTransformer embedding wrapper
│       └── rag_pipeline.py    # Core RAG: Chunking, ChromaDB, Q&A, Summary & Quiz
└── chroma_db/                 # Persistent Chroma vector database
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Ensure you have Python 3.10+ installed:

```bash
# Activate virtual environment (Windows PowerShell)
.\Sprint_4\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)

Create a `.env` file in the project root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
YOUTUBE_API_KEY=your_youtube_data_api_key_here
```

### 3. Launch the Application

#### Option A: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

#### Option B: Terminal CLI
```bash
python main.py
```

---

## 👨‍💻 Developed For
**LearnTube AI** — Accelerating education with Retrieval-Augmented Generation (RAG) and LangChain.
