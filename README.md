# 🎓 NCERT AI Tutor

> An AI-powered RAG chatbot for Class 10 NCERT — Mathematics & Science.  
> Built with Groq (Llama 3.1), ChromaDB, LangChain, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%201.5-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🚀 Live Demo

👉 **[Try it here](https://ncert-ai-tutor-appdhcgh4agx3ppbvxt4y7j.streamlit.app)**

---

## 🧠 What This Project Does

Students type questions in **English or Hindi** → the app retrieves relevant chunks from actual NCERT Class 10 PDFs → Llama 3.1 generates a grounded answer.

No hallucinated answers from the LLM's training data. No random internet content. Just your NCERT textbook — explained like a friend or formatted for board exams.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📚 RAG Pipeline | Retrieves answers strictly from NCERT Class 10 PDFs |
| 🔍 Subject Filter | Post-retrieval chunk validation — Maths (`jemh*`) or Science (`jesc*`) |
| 🌐 Bilingual | English and Hindi answer modes |
| 🧠 Answer Modes | Easy Explanation (friend-style) or Exam Preparation (structured points) |
| 💬 Memory | Remembers last 3 conversation turns (`ConversationBufferWindowMemory`) |
| 🛡️ Fallback Defence | 3-layer system prevents out-of-syllabus answers |
| 💡 Dynamic Suggestions | Groq-generated follow-up questions after each answer |
| 📝 Export Summary | Download full session as a `.txt` study summary |
| 🎨 Gen Z Dark UI | Purple gradient, WhatsApp-style animated chat bubbles, glassmorphism |
| ⏱️ Timestamps | Every message shows HH:MM AM/PM |

---

## 🏗️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| UI | Streamlit + Poppins + Custom CSS | 1.58.0 |
| LLM | Groq API → `llama-3.1-8b-instant` | groq 0.37.1 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API) | 5.5.1 |
| Vector DB | ChromaDB (local, persisted to `chroma_db/`) | 1.5.9 |
| PDF Parsing | PyMuPDF via `langchain_community` | 1.27.2 |
| Memory | `langchain_classic.ConversationBufferWindowMemory(k=3)` | 1.0.8 |
| Orchestration | LangChain core + classic + community | 1.3.9 |

---

## 📁 Project Structure

```
ncert-ai-tutor/
│
├── app.py                  # Streamlit UI — Gen Z dark theme redesign
├── rag_pipeline.py         # RAG engine — retrieval, filtering, LLM calls
├── ingest.py               # One-time PDF → ChromaDB ingestion script
├── requirements.txt        # All Python dependencies with exact versions
├── .env                    # Your Groq API key (NOT committed to Git)
├── .gitignore              # Excludes .env, chroma_db/, __pycache__, venv/
│
├── data/
│   └── ncert_pdfs/         # Place your NCERT PDF files here
│       ├── jemh101.pdf     # Class 10 Maths chapters  (jemh prefix)
│       ├── jemh102.pdf
│       └── jesc101.pdf     # Class 10 Science chapters (jesc prefix)
│
└── chroma_db/              # Auto-created after running ingest.py (gitignored)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ayush22cp008/ncert-ai-tutor.git
cd ncert-ai-tutor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

### 5. Add NCERT PDFs

Download Class 10 NCERT PDFs from [ncert.nic.in](https://ncert.nic.in) and place them inside `data/ncert_pdfs/`.

> **Naming matters!** The subject filter reads the filename prefix:
> - Maths PDFs → must start with `jemh` (e.g. `jemh101.pdf`, `jemh105.pdf`)
> - Science PDFs → must start with `jesc` (e.g. `jesc101.pdf`, `jesc202.pdf`)

### 6. Run ingestion (one time only)

```bash
python ingest.py
```

This reads all PDFs, chunks them, embeds with `all-MiniLM-L6-v2`, and saves to `chroma_db/`.  
Re-run only when you add new PDFs.

### 7. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛡️ How the 3-Layer Fallback System Works

Prevents the LLM from answering outside the NCERT Class 10 syllabus:

```
User Question
     │
     ▼
ChromaDB: similarity_search(k=10) — retrieves top-10 candidate chunks
     │
     ▼
[Layer 1] Post-retrieval subject filter
          os.path.basename(source).startswith("jemh" / "jesc")
     │
     ├── 0 valid chunks → STATIC FALLBACK returned immediately
     │                    No LLM call. No API cost.
     │
     └── Valid chunks found ↓
     
[Layer 2] STRICT RULE in prompt (all 4 templates)
          "DO NOT use training knowledge"
          "DO NOT answer outside this context"
          "If not found → say fallback phrase exactly"
     │
     ▼
LLM generates answer using ONLY provided NCERT chunks
     │
     ▼
[Layer 3] Sentinel detection in answer text
          if "not covered in your Class 10" in answer → is_fallback=True
          Effect: dynamic suggestions skipped, memory not updated
```

---

## ⚠️ Known Limitations

Honest limitations — tested and acknowledged:

### 1. Subject filter is post-retrieval, not pre-retrieval
The ChromaDB `where` filter was unreliable because it stored **full absolute Windows paths** (e.g. `C:\Users\...\jemh101.pdf`) at ingest time, but the filter compared against relative paths. The current fix uses `os.path.basename()` after retrieval — which works in practice, but retrieves all 10 candidates first before filtering. A proper fix would be to store a `subject` metadata field (e.g. `"Mathematics"`) at ingest time and filter on that.

### 2. LLM can still answer from training data on very generic questions
The `STRICT RULE` prompt reduces this significantly, but questions like *"What is AI?"* or *"Who is the Prime Minister?"* may still get answered from Llama's training data instead of triggering the fallback. Adding a **query classification step** (is this question answerable from Class 10 NCERT?) before calling the RAG chain would fix this completely.

### 3. Memory is session-only (not persistent across restarts)
`ConversationBufferWindowMemory` lives in `st.session_state` — it resets when the browser tab is closed or Streamlit restarts. Persistent cross-session memory would require saving to a file or database.

### 4. No authentication / multi-user support
This is a single-user local app. All users on the same Streamlit instance share the same global ChromaDB embedding cache (`_vector_store`). Not designed for production deployment as-is.

### 5. Hindi support depends on PDF content language
The Hindi **answer mode** changes the LLM's response language, but the NCERT PDFs ingested are in English. Retrieving chunks in English and responding in Hindi works, but chunk relevance scoring may be slightly lower for Hindi queries.

---

## 🔧 Key Engineering Decisions & Bug Fixes

**PyMuPDF instead of PyPDF**  
PyPDF threw `LimitReachedError` on large NCERT PDFs (>200 pages). PyMuPDF handled all files without issues.

**Post-retrieval filter instead of ChromaDB `where` filter**  
ChromaDB stores the exact path passed at ingest time. On Windows, this was the full absolute path. The `$in` filter compared relative paths — they never matched, so filtering was silently disabled. `os.path.basename()` works regardless of path format.

**`langchain_classic.memory` instead of `langchain.memory`**  
`langchain.memory` import path was deprecated and raises `ModuleNotFoundError`. The maintained module is `langchain_classic.memory`.

**`base_prompt.partial()` for injecting `chat_history`**  
`RetrievalQA` only passes `{context}` and `{question}` to the chain internally — extra keys like `{chat_history}` caused `ValueError: Missing some input keys`. Pre-filling via `.partial()` before passing the prompt to `RetrievalQA` fixed this.

**`@import url()` instead of `<link>` tag for Google Fonts**  
A `<link>` tag placed before `<style>` inside `st.markdown()` was rendered as visible text by Streamlit. Moving the font import inside `<style>` as `@import url(...)` fixed the raw-text CSS display.

**Direct prompt `.replace()` instead of `RetrievalQA`**  
The final refactor removed `RetrievalQA` entirely. We now call `vector_store.similarity_search()` directly, filter results, build the prompt string with `.replace()`, and call `ChatGroq.invoke()`. This gives full control and eliminated all chain-level key errors.

---

## 📦 Full Requirements

```
langchain==1.3.9
langchain-core==1.4.7
langchain-community==0.4.2
langchain-classic==1.0.8
langchain-groq==1.1.3
chromadb==1.5.9
sentence-transformers==5.5.1
pymupdf==1.27.2.3
streamlit==1.58.0
python-dotenv==1.2.2
groq==0.37.1
```

---

## 🙋 Built By

**Ayush Halpati** —  GenAI Developer  
B.Tech Computer Engineering, BVM Engineering College, Gujarat  
[LinkedIn](https://www.linkedin.com/in/ayush-halpati-886527260) · [GitHub](https://github.com/ayush22cp008)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

> _"Not just another chatbot. Your NCERT textbook — finally explained the way you actually learn."_
