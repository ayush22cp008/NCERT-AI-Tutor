"""
app.py — NCERT AI Tutor (Gen Z Redesign)
------------------------------------------
Streamlit UI — Discord + Spotify inspired dark theme.
Run with: streamlit run app.py
"""

import json
import os
from datetime import datetime
import streamlit as st
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from rag_pipeline import get_answer

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCERT AI Tutor — Class 10",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TEMPORARY DEBUG — remove after confirming API key is set on Streamlit Cloud ──
groq_key = os.getenv("GROQ_API_KEY")
st.write(f"DEBUG: GROQ key found = {bool(groq_key)}")
# ─────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (must be the FIRST st.markdown after set_page_config)
# Google Fonts loaded via @import inside <style> — NOT a separate <link> tag
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

/* ── Reset & Base ───────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background-color: #0A0A0F !important;
    color: #F8FAFC !important;
}

/* ── Custom scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A0A0F; }
::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6D28D9; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #111118 !important;
    border-right: 1px solid rgba(124,58,237,0.2) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1rem 0.75rem; }

/* ── Main area ──────────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Radio buttons → pill style ─────────────────────────────────────────── */
div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
div[role="radiogroup"] label {
    background: rgba(124,58,237,0.12) !important;
    border: 1.5px solid rgba(124,58,237,0.35) !important;
    border-radius: 24px !important;
    padding: 6px 16px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    white-space: nowrap !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg,#7C3AED,#3B82F6) !important;
    border-color: transparent !important;
    color: #fff !important;
    box-shadow: 0 0 14px rgba(124,58,237,0.45) !important;
}
div[role="radiogroup"] input[type="radio"] { display: none !important; }

/* ── Streamlit widget labels ────────────────────────────────────────────── */
label[data-testid="stWidgetLabel"] p,
.stRadio > label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 4px !important;
}

/* ── ALL sidebar buttons (default gradient) ─────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 24px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    border: none !important;
    padding: 10px 20px !important;
    background: linear-gradient(135deg, #7C3AED, #3B82F6) !important;
    color: #fff !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.35) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.55) !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: rgba(124,58,237,0.15) !important;
    color: #4B5563 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* ── Clear Chat button (red) — targets by key ───────────────────────────── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="secondary"],
#clear_btn,
button[data-testid="baseButton-secondary"] {
    background: rgba(239,68,68,0.18) !important;
    border: 1.5px solid rgba(239,68,68,0.4) !important;
    color: #FCA5A5 !important;
    box-shadow: none !important;
}
/* Alternate: target the last sidebar button as the clear button */
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: rgba(239,68,68,0.18) !important;
    border: 1.5px solid rgba(239,68,68,0.4) !important;
    color: #FCA5A5 !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    background: rgba(239,68,68,0.32) !important;
    border-color: rgba(239,68,68,0.7) !important;
    color: #fff !important;
    box-shadow: 0 0 20px rgba(239,68,68,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Main area buttons (glassmorphism cards) ────────────────────────────── */
.main .stButton > button {
    background: rgba(124,58,237,0.12) !important;
    border: 1.5px solid rgba(124,58,237,0.35) !important;
    border-radius: 16px !important;
    color: #C4B5FD !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    padding: 10px 14px !important;
    backdrop-filter: blur(10px) !important;
    text-align: left !important;
}
.main .stButton > button:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(59,130,246,0.35)) !important;
    border-color: rgba(124,58,237,0.7) !important;
    color: #fff !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.35) !important;
}

/* ── Download button ────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    border: none !important;
    border-radius: 24px !important;
    color: #fff !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 18px rgba(16,185,129,0.35) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(16,185,129,0.5) !important;
}

/* ── Chat input ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 28px !important;
    border: 1.5px solid rgba(124,58,237,0.4) !important;
    background: #1A1A2E !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.12) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(124,58,237,0.8) !important;
    box-shadow: 0 0 28px rgba(124,58,237,0.3) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #F8FAFC !important;
    font-family: 'Poppins', sans-serif !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1.5px solid rgba(124,58,237,0.2) !important;
    border-radius: 16px !important;
    background: rgba(26,26,46,0.7) !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stExpander"] summary {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: #7C3AED !important; }

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(124,58,237,0.15) !important;
    margin: 12px 0 !important;
}

/* ── Caption ────────────────────────────────────────────────────────────── */
.stCaption { color: #4B5563 !important; font-size: 0.75rem !important; }

/* ── Custom HTML classes ────────────────────────────────────────────────── */
.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7C3AED;
    padding-left: 10px;
    border-left: 3px solid #7C3AED;
    margin: 14px 0 8px 0;
}
.info-tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #C4B5FD;
    padding: 2px 10px;
    margin: 2px 3px 2px 0;
}

/* ── Hero header ────────────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0D0D1A 0%, #1A0A2E 40%, #0A1628 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(124,58,237,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 50% 50%, rgba(124,58,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #F59E0B, #D97706);
    color: #1A1A2E;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 4px 14px;
    border-radius: 999px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    box-shadow: 0 0 20px rgba(245,158,11,0.4);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #F8FAFC 0%, #C4B5FD 50%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94A3B8;
    font-weight: 500;
    margin: 0;
}

/* ── Chat bubbles ────────────────────────────────────────────────────────── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-wrapper { display: flex; flex-direction: column; gap: 14px; margin-bottom: 1rem; }

.msg-row { display: flex; align-items: flex-end; gap: 10px; animation: slideUp 0.35s ease forwards; }
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

.msg-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.msg-avatar.user-av { background: linear-gradient(135deg,#7C3AED,#3B82F6); box-shadow: 0 0 12px rgba(124,58,237,0.5); }
.msg-avatar.bot-av  { background: #1E1B4B; border: 1.5px solid rgba(124,58,237,0.4); }

.msg-bubble {
    max-width: 72%; padding: 12px 16px; border-radius: 16px;
    font-size: 0.9rem; line-height: 1.6; word-wrap: break-word; position: relative;
}
.msg-bubble.user-bubble {
    background: linear-gradient(135deg,#7C3AED,#3B82F6); color: #fff;
    border-bottom-right-radius: 4px; box-shadow: 0 4px 20px rgba(124,58,237,0.35);
}
.msg-bubble.bot-bubble {
    background: #1E1B4B; color: #E2E8F0;
    border-left: 3px solid #7C3AED; border-bottom-left-radius: 4px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.msg-time { font-size: 0.65rem; color: #4B5563; margin-top: 4px; text-align: right; }
.msg-row.bot .msg-time { text-align: left; }

/* ── Source chip ─────────────────────────────────────────────────────────── */
.source-chip {
    display: inline-block;
    background: rgba(124,58,237,0.18); border: 1px solid rgba(124,58,237,0.35);
    border-radius: 999px; font-size: 0.72rem; font-weight: 500;
    color: #C4B5FD; padding: 3px 10px; margin: 3px 4px 3px 0;
}

/* ── Sample / suggestion labels ──────────────────────────────────────────── */
.sample-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7C3AED; margin: 18px 0 10px 0;
    display: flex; align-items: center; gap: 6px;
}
.sug-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
    color: #F59E0B; margin: 16px 0 8px 0;
}

/* ── Session summary card ────────────────────────────────────────────────── */
.summary-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(59,130,246,0.08));
    border: 1.5px solid rgba(124,58,237,0.3); border-radius: 20px;
    overflow: hidden; margin: 1rem 0;
    box-shadow: 0 0 30px rgba(124,58,237,0.12);
}
.summary-header {
    background: linear-gradient(135deg,#7C3AED,#3B82F6);
    padding: 14px 20px; font-size: 1rem; font-weight: 700; color: #fff;
}
.summary-body {
    padding: 16px 20px; color: #E2E8F0;
    font-size: 0.88rem; line-height: 1.8; white-space: pre-wrap;
}

/* ── Mobile responsive ───────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem !important; }
    .msg-bubble { max-width: 90% !important; }
    .main .block-container { padding: 0.5rem 1rem 1rem 1rem !important; }
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = {
    "Mathematics": [
        "📐 What are real numbers?",
        "🔢 What is arithmetic progression?",
        "📏 Explain Pythagoras theorem",
        "🎲 What is probability?",
    ],
    "Science": [
        "🌿 What is photosynthesis?",
        "⚡ Explain reflex action",
        "🧬 What is heredity?",
        "💡 How does electricity work?",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_dynamic_suggestions(last_question: str, subject: str) -> list:
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.5,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        prompt = (
            f"Based on this Class 10 NCERT {subject} question: '{last_question}' "
            f"and its answer, suggest exactly 3 short follow-up questions that are "
            f"STRICTLY from Class 10 NCERT {subject} textbook syllabus only. "
            f"Do not suggest anything outside NCERT Class 10 {subject} curriculum. "
            f'Return ONLY a JSON array of 3 strings. Example: ["Q1?", "Q2?", "Q3?"]'
        )
        response = llm.invoke(prompt)
        raw = response.content.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            suggestions = json.loads(raw[start:end])
            if isinstance(suggestions, list) and len(suggestions) >= 3:
                return [str(s) for s in suggestions[:3]]
    except Exception:
        pass
    return []


def generate_session_summary(messages: list, subject: str, language: str) -> str:
    transcript_lines = []
    for msg in messages:
        role = "Student" if msg["role"] == "user" else "Tutor"
        transcript_lines.append(f"{role}: {msg['content']}")
    chat_history_text = "\n".join(transcript_lines)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    prompt = f"""Create a structured study session summary for a Class 10 student based on this chat:

{chat_history_text}

Format the summary exactly like this:

NCERT AI Tutor - Session Summary
Subject: {subject} | Language: {language}

Topics Covered:
- [topic 1]
- [topic 2]

Key Concepts Learned:
- [concept 1]
- [concept 2]

Total Questions Asked: [count]

Chapters Referenced:
- [chapter info from sources]

Suggested Topics to Study Next:
- [topic 1]
- [topic 2]
- [topic 3]

Keep it concise and student-friendly."""

    response = llm.invoke(prompt)
    return response.content.strip()


def format_timestamp(ts):
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%I:%M %p")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">📘 Class 10 Only</div>
    <h1 class="hero-title">NCERT AI Tutor</h1>
    <p class="hero-subtitle">Your AI Study Partner 🚀</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(
        k=3, memory_key="history", return_messages=False,
    )
if "suggestions" not in st.session_state:
    st.session_state["suggestions"] = []
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "session_summary" not in st.session_state:
    st.session_state["session_summary"] = ""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Language

    st.markdown("<div class='sidebar-section-label'>🌐 Language / भाषा</div>", unsafe_allow_html=True)
    language = st.radio(
        "lang",
        options=["English", "Hindi"],
        index=0,
        key="language_selector",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    # Subject
    st.markdown("<div class='sidebar-section-label'>📖 Subject / विषय</div>", unsafe_allow_html=True)
    subject = st.radio(
        "subj",
        options=["Mathematics", "Science"],
        index=0,
        key="subject_selector",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    # Answer Mode
    st.markdown("<div class='sidebar-section-label'>📝 Answer Mode / उत्तर का तरीका</div>", unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        options=["🧠 Easy Explanation", "📝 Exam Preparation"],
        index=0,
        key="mode_selector",
        label_visibility="collapsed",
    )
    st.divider()

    # How to use
    st.markdown("<div class='sidebar-section-label'>💡 How to Use</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem;color:#94A3B8;line-height:1.9;padding-left:4px;'>
        1. Pick <b style='color:#C4B5FD'>Language</b> &amp; <b style='color:#C4B5FD'>Subject</b><br>
        2. Choose an <b style='color:#C4B5FD'>Answer Mode</b><br>
        3. Ask your question below!<br>
        4. Tutor remembers last <b style='color:#C4B5FD'>3 exchanges</b>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Export Summary
    num_messages = len(st.session_state.get("messages", []))
    if num_messages >= 2:
        if st.button("📥 Export Summary", key="export_summary_btn"):
            with st.spinner("Generating your study summary..."):
                try:
                    summary = generate_session_summary(
                        messages=st.session_state["messages"],
                        subject=subject,
                        language=language,
                    )
                    st.session_state["session_summary"] = summary
                except Exception as e:
                    st.session_state["session_summary"] = f"Error generating summary: {e}"
    else:
        st.button("📥 Export Summary", key="export_summary_btn", disabled=True)
        st.caption("⚠️ Ask some questions first!")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Clear Chat
    if st.button("🗑️ Clear Chat History", key="clear_btn"):
        st.session_state["messages"] = []
        st.session_state["suggestions"] = []
        st.session_state["last_question"] = ""
        st.session_state["session_summary"] = ""
        st.session_state["memory"] = ConversationBufferWindowMemory(
            k=3, memory_key="history", return_messages=False,
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY — WhatsApp-style bubbles
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["messages"]:
    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        ts = format_timestamp(msg.get("ts"))
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='msg-row user'>
                <div class='msg-avatar user-av'>🧑&#8203;</div>
                <div>
                    <div class='msg-bubble user-bubble'>{msg['content']}</div>
                    <div class='msg-time'>{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            content_html = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class='msg-row bot'>
                <div class='msg-avatar bot-av'>🤖</div>
                <div style='max-width:72%'>
                    <div class='msg-bubble bot-bubble'>{content_html}</div>
                    <div class='msg-time'>{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Source chips inside a native Streamlit expander
            if msg.get("sources"):
                with st.expander("📚 Source Documents", expanded=False):
                    chips = []
                    for src in msg["sources"]:
                        raw_src = src.metadata.get("source", "?")
                        src_name = raw_src.replace("\\", "/").split("/")[-1]
                        src_page = src.metadata.get("page", "?")
                        chips.append(f"<span class='source-chip'>📄 {src_name} — p.{src_page}</span>")
                    st.markdown("".join(chips), unsafe_allow_html=True)
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div style='background:rgba(124,58,237,0.08);
                                    border:1px solid rgba(124,58,237,0.2);
                                    border-radius:12px;padding:10px 14px;margin:6px 0;
                                    font-size:0.8rem;color:#94A3B8;'>
                            <b style='color:#C4B5FD'>Source {i+1}</b>&nbsp;
                            {src.page_content[:280]}...
                        </div>
                        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION SUMMARY CARD
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("session_summary"):
    with st.expander("📋 Your Session Summary", expanded=True):
        st.markdown(st.session_state["session_summary"])
        st.download_button(
            label="⬇️ Download Summary (.txt)",
            data=st.session_state["session_summary"],
            file_name="ncert_session_summary.txt",
            mime="text/plain",
            key="download_summary_btn",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE QUESTIONS / DYNAMIC SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────
chat_is_empty = len(st.session_state["messages"]) == 0

if chat_is_empty:
    st.markdown("<div class='sample-label'>💡 Sample Questions</div>", unsafe_allow_html=True)
    sample_qs = SAMPLE_QUESTIONS.get(subject, [])
    cols = st.columns(2)
    for idx, q in enumerate(sample_qs):
        with cols[idx % 2]:
            if st.button(q, key=f"sample_main_{subject}_{idx}"):
                # Strip leading emoji (2 chars: emoji + space)
                clean_q = q[2:].strip()
                st.session_state["user_question"] = clean_q
                st.rerun()

elif st.session_state["suggestions"]:
    st.markdown("<div class='sug-label'>💡 You might want to ask:</div>", unsafe_allow_html=True)
    sug_cols = st.columns(3)
    for idx, suggestion in enumerate(st.session_state["suggestions"]):
        with sug_cols[idx % 3]:
            if st.button(suggestion, key=f"sug_{idx}_{suggestion[:20]}"):
                st.session_state["user_question"] = suggestion
                st.session_state["suggestions"] = []
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
default_q = st.session_state.pop("user_question", "")

user_input = st.chat_input(
    placeholder="Ask anything from Class 10 NCERT... 📚"
    if language == "English"
    else "NCERT se kuch bhi poochho... 📚",
)

if default_q and not user_input:
    user_input = default_q

if user_input:
    st.session_state["suggestions"] = []
    ts_now = datetime.now().isoformat()

    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "ts": ts_now,
    })

    answer = ""
    source_docs = []
    is_fallback = False
    with st.spinner(f"🔍 Searching Class 10 {subject} textbooks..."):
        try:
            result = get_answer(
                question=user_input,
                language=language,
                subject=subject,
                mode=mode,
                memory=st.session_state["memory"],
            )
            answer = result["answer"]
            source_docs = result["sources"]
            is_fallback = result.get("is_fallback", False)
        except FileNotFoundError as e:
            st.error(f"⚠️ {e}", icon="🚨")
            st.info("Please add your NCERT PDFs to `data/ncert_pdfs/` and run: `python ingest.py`")

    if answer:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs,
            "ts": datetime.now().isoformat(),
        })
        st.session_state["last_question"] = user_input

        # Only generate follow-up suggestions when NCERT content was found
        if not is_fallback:
            with st.spinner("💭 Generating follow-up suggestions..."):
                st.session_state["suggestions"] = get_dynamic_suggestions(
                    last_question=user_input,
                    subject=subject,
                )
        else:
            # Clear any stale suggestions on a fallback response
            st.session_state["suggestions"] = []

        st.rerun()

