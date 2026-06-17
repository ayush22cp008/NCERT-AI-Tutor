"""
rag_pipeline.py — NCERT AI Tutor
----------------------------------
Builds and returns answers via a RAG pipeline.

The pipeline:
1. Loads the pre-built ChromaDB vector store (cached globally).
2. Retrieves top-K candidate chunks WITHOUT a ChromaDB filter
   (ChromaDB stores full Windows paths; path-based filters are unreliable).
3. Post-filters retrieved chunks by checking os.path.basename() against the
   subject prefix ("jemh" for Mathematics, "jesc" for Science).
4. If 0 valid chunks remain → returns a subject-specific fallback immediately
   (no LLM call, saving API quota).
5. Otherwise builds the prompt with validated context and calls Groq Llama 3.
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DB_DIR  = "chroma_db"
PDF_DIR        = "data/ncert_pdfs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL     = "llama-3.1-8b-instant"

# Filename prefix that identifies each subject
SUBJECT_PREFIX = {
    "Mathematics": "jemh",
    "Science":     "jesc",
}

# Sentinel phrases embedded in all 4 prompt templates' fallback instructions.
# get_answer() checks the LLM output for these to set is_fallback=True.
FALLBACK_SENTINEL_EN = "not covered in your Class 10"
FALLBACK_SENTINEL_HI = "NCERT ki kitaab mein nahi hai"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES  (2 languages × 2 modes = 4 templates)
# ─────────────────────────────────────────────────────────────────────────────

# ── Easy Explanation ─────────────────────────────────────────────────────────
PROMPT_EASY_ENGLISH = """
You are a friendly NCERT tutor for Class 10 students.
Explain in very simple language like talking to a friend.

*** STRICT RULE — READ THIS FIRST ***
- Answer STRICTLY and ONLY using the NCERT context provided below.
- DO NOT use your training knowledge, general knowledge, or outside information.
- DO NOT answer questions that are NOT covered in the provided context.
- If the context does not contain the answer, you MUST say exactly:
  "This topic is not covered in your Class 10 {subject} NCERT textbook. Try asking something from your {subject} syllabus! 📚"
  Do NOT make up or guess an answer.
*************************************

Previous conversation:
{chat_history}

Style instructions (apply ONLY when answering from the context above):
- Use simple words that a 15-year-old can easily understand.
- Give real life examples (cricket, mobile phone, food, daily life objects).
- Use analogies like 'think of it like...' to explain concepts.
- Keep it short and fun — no heavy jargon.
- End your answer with: 'Hope that makes sense! 😊'

Context from NCERT Textbook:
{context}

Student's Question: {question}

Fun & Simple Answer:
"""

PROMPT_EASY_HINDI = """
Aap ek friendly tutor hain. Bilkul dost ki tarah simple Hindi mein samjhao.

*** SAKHT NIYAM — PEHLE PADHO ***
- Jawab SIRF neeche diye gaye NCERT context se do.
- Apni training knowledge, general knowledge ya bahar ki koi bhi jaankari USE MAT KARO.
- Agar context mein jawab nahi hai toh bilkul yahi likho:
  "Yeh topic aapki Class 10 {subject} NCERT ki kitaab mein nahi hai. {subject} ke syllabus se kuch aur poochho! 📚"
  Andaza lagakar ya kuch bana kar jawab mat do.
*********************************

Pichli baat-cheet:
{chat_history}

Style instructions (sirf tab use karo jab context mein jawab ho):
- Aasaan shabdon mein batao jaise kisi dost ko samjha rahe ho.
- Real life examples do (cricket, mobile, khana, rozana ki cheezein).
- Chhota aur interesting rakho — mushkil words mat use karo.
- Ant mein likho: 'Samajh aa gaya na! 😊'

NCERT Textbook se Context:
{context}

Student ka Sawaal: {question}

Aasaan Jawab:
"""

# ── Exam Preparation ─────────────────────────────────────────────────────────
PROMPT_EXAM_ENGLISH = """
You are an NCERT exam preparation tutor for Class 10.
Give a proper exam-ready answer that a student can directly write in their board exam.

*** STRICT RULE — READ THIS FIRST ***
- Answer STRICTLY and ONLY using the NCERT context provided below.
- DO NOT use your training knowledge, general knowledge, or outside information.
- DO NOT answer questions that are NOT covered in the provided context.
- If the context does not contain the answer, you MUST say exactly:
  "This topic is not covered in your Class 10 {subject} NCERT textbook. Try asking something from your {subject} syllabus! 📚"
  Do NOT make up or guess an answer.
*************************************

Previous conversation:
{chat_history}

Style instructions (apply ONLY when answering from the context above):
- Start with a clear 1-line definition of the concept.
- Use numbered points or bullet points for the explanation.
- Use proper scientific or mathematical terminology.
- Include the formula with variable definitions if applicable.
- Keep the answer concise — between 3 to 5 points maximum.
- End with: 'Key Point: [one line summary of the most important idea]'
- Format exactly as a student would write in their board exam.

Context from NCERT Textbook:
{context}

Student's Question: {question}

Exam-Ready Answer:
"""

PROMPT_EXAM_HINDI = """
Aap NCERT exam tutor hain Class 10 ke liye.
Exam mein likhne layak answer do jo student seedha copy kar sake.

*** SAKHT NIYAM — PEHLE PADHO ***
- Jawab SIRF neeche diye gaye NCERT context se do.
- Apni training knowledge, general knowledge ya bahar ki koi bhi jaankari USE MAT KARO.
- Agar context mein jawab nahi hai toh bilkul yahi likho:
  "Yeh topic aapki Class 10 {subject} NCERT ki kitaab mein nahi hai. {subject} ke syllabus se kuch aur poochho! 📚"
  Andaza lagakar ya kuch bana kar jawab mat do.
*********************************

Pichli baat-cheet:
{chat_history}

Style instructions (sirf tab use karo jab context mein jawab ho):
- Pehle ek line ki definition likho.
- Numbered points mein explanation do.
- Sahi scientific ya mathematical terms use karo.
- Agar formula ho toh zaroor likho with variable definitions.
- 3-5 points mein poora answer complete karo.
- Ant mein likho: 'Mukhya Baat: [ek line mein sabse zaroori baat]'

NCERT Textbook se Context:
{context}

Student ka Sawaal: {question}

Exam-Ready Jawab:
"""

# ─────────────────────────────────────────────────────────────────────────────
# STATIC FALLBACK MESSAGES (returned before calling LLM)
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_MESSAGES = {
    ("Mathematics", "English"): (
        "This topic is not covered in your Class 10 Mathematics NCERT textbook. "
        "Try asking something from Maths syllabus! 📚"
    ),
    ("Science", "English"): (
        "This topic is not covered in your Class 10 Science NCERT textbook. "
        "Try asking something from Science syllabus! 📚"
    ),
    ("Mathematics", "Hindi"): (
        "Yeh topic aapki Class 10 Mathematics NCERT ki kitaab mein nahi hai. "
        "Mathematics ke syllabus se kuch aur poochho! 📚"
    ),
    ("Science", "Hindi"): (
        "Yeh topic aapki Class 10 Science NCERT ki kitaab mein nahi hai. "
        "Science ke syllabus se kuch aur poochho! 📚"
    ),
}


def _get_fallback(subject: str, language: str) -> str:
    """Return the subject+language-specific fallback message."""
    key = (subject, language)
    return FALLBACK_MESSAGES.get(
        key,
        "This topic is not in the NCERT content. Try rephrasing your question! 📚",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CACHE
# ─────────────────────────────────────────────────────────────────────────────
_embeddings   = None
_vector_store = None


def _get_embeddings():
    """Load HuggingFace embeddings once and cache globally."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def load_vector_store():
    """Load the existing ChromaDB vector store (cached globally)."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DB_DIR}'. "
            "Please run 'python ingest.py' first."
        )
    _vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=_get_embeddings(),
    )
    return _vector_store


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — POST-RETRIEVAL SUBJECT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_chunk(doc, subject: str) -> bool:
    """
    Check whether a retrieved Document belongs to the selected subject.

    ChromaDB stores whatever path was used at ingest time — which on Windows
    can be a full absolute path like:
        C:\\Users\\ayush\\Desktop\\NCERT AI Tutor\\data\\ncert_pdfs\\jemh101.pdf
    or a relative path like:
        data/ncert_pdfs\\jemh101.pdf

    Using os.path.basename() reliably extracts "jemh101.pdf" from any variant,
    then we check the prefix against the expected subject prefix.
    """
    source = doc.metadata.get("source", "")
    filename = os.path.basename(source).lower()
    prefix = SUBJECT_PREFIX.get(subject, "").lower()
    if not prefix:
        return True  # Unknown subject — allow all
    return filename.startswith(prefix)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT ROUTING
# ─────────────────────────────────────────────────────────────────────────────
def _pick_prompt_template(language: str, mode: str) -> str:
    """Return the correct prompt template string for language × mode."""
    is_hindi = language == "Hindi"
    is_exam  = "Exam" in mode
    if is_hindi and is_exam:
        return PROMPT_EXAM_HINDI
    if is_hindi:
        return PROMPT_EASY_HINDI
    if is_exam:
        return PROMPT_EXAM_ENGLISH
    return PROMPT_EASY_ENGLISH


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def get_answer(
    question: str,
    language: str = "English",
    subject: str  = "Mathematics",
    mode: str     = "🧠 Easy Explanation",
    memory        = None,
) -> dict:
    """
    High-level RAG wrapper with post-retrieval subject filtering.

    Returns:
        dict with keys:
            "answer"      (str)  — The answer text.
            "sources"     (list) — Valid source Documents used.
            "is_fallback" (bool) — True when no matching NCERT content found.
    """
    # ── Load chat history ────────────────────────────────────────────────────
    chat_history = ""
    if memory is not None:
        vars_ = memory.load_memory_variables({})
        chat_history = vars_.get("history", "")

    # ── STEP 1: Retrieve candidate chunks (NO ChromaDB filter) ───────────────
    # We retrieve more candidates (k=10) so post-filtering has enough to work
    # with even if several chunks belong to the wrong subject.
    vector_store = load_vector_store()
    candidate_docs = vector_store.similarity_search(question, k=10)

    # ── STEP 2: Post-retrieval subject validation ────────────────────────────
    valid_docs = [doc for doc in candidate_docs if is_valid_chunk(doc, subject)]

    # ── STEP 3: Short-circuit if nothing valid found ─────────────────────────
    if not valid_docs:
        fallback_text = _get_fallback(subject, language)
        return {
            "answer":      fallback_text,
            "sources":     [],
            "is_fallback": True,
        }

    # Use at most the top-4 valid chunks for the LLM context
    top_docs = valid_docs[:4]

    # ── STEP 4: Build context string ─────────────────────────────────────────
    context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)

    # ── STEP 5: Fill prompt template ─────────────────────────────────────────
    template = _pick_prompt_template(language, mode)
    prompt_text = (
        template
        .replace("{chat_history}", chat_history)
        .replace("{subject}",      subject)
        .replace("{context}",      context)
        .replace("{question}",     question)
    )

    # ── STEP 6: Call LLM ─────────────────────────────────────────────────────
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        streaming=False,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    response = llm.invoke(prompt_text)
    answer   = response.content.strip()

    # ── STEP 7: Detect LLM-generated fallback ────────────────────────────────
    # Even with valid chunks the LLM can still say "not found" for edge cases.
    is_fallback = (
        FALLBACK_SENTINEL_EN in answer
        or FALLBACK_SENTINEL_HI in answer
    )

    # ── STEP 8: Save to memory only when a real answer was returned ───────────
    if memory is not None and not is_fallback:
        memory.save_context({"input": question}, {"output": answer})

    return {
        "answer":      answer,
        "sources":     top_docs,
        "is_fallback": is_fallback,
    }
