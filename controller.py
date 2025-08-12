# controller.py
import os
import logging
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from data_class.prompt_response import PromptType
from knowledge_base.knowledge_base_helper import VectorDBHelper
from knowledge_base.kb_builder import VectorDbBuilder
from llm_helper.llm_helper import LLMHelper
from utils.constants import Constants
from scripts.next_question_ranker import NextQuestionLR, Candidate

# Optional dependency: transformers for sentiment
try:
    from transformers import pipeline  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# ---------------- Logging ----------------
logger = logging.getLogger("chat_controller")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------- Caches ----------------
@st.cache_resource
def _initialize_backends():
    """One-time backend init (cached)."""
    if Constants.should_build_vector_db():
        VectorDbBuilder.build_db()
    load_dotenv()
    vdb = VectorDBHelper()
    vdb.load_kb()
    llm = LLMHelper(context_buffer=[])
    return vdb, llm

@st.cache_resource
def _load_sentiment_pipeline():
    """Load and cache the HF sentiment pipeline if available."""
    if not _TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available; sentiment gating will use keyword-only heuristics.")
        return None
    try:
        # Lightweight, robust general-purpose sentiment model
        sa = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        logger.info("Sentiment analyzer loaded.")
        return sa
    except Exception as e:
        logger.warning(f"Could not load sentiment analyzer: {e}")
        return None

# ---------------- State helpers ----------------
def ensure_state():
    """Prepare Streamlit session state keys used by the app."""
    if "messages" not in st.session_state:
        # each: {"role": "user"|"assistant", "content": str, "ts": "YYYY-MM-DD HH:MM"}
        st.session_state.messages = []
    if "next_questions" not in st.session_state:
        st.session_state.next_questions = []
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------------- Controller ----------------
class ChatController:
    # Escalation config
    ESCALATION_KEYWORDS = {
        # physical/mental health crisis indicators
        "suicide", "harm myself", "kill myself", "end my life", "self harm", "self-harm",
        "not safe", "unsafe", "panic attack", "severe anxiety", "depressed", "depression",
        "overwhelmed", "i am not well", "i'm not well", "i am not feeling well",
        "mental health", "crisis", "emergency", "i want to die",
        # harassment/abuse indicators
        "assault", "harassed", "abused", "stalked", "threatened",
        # medical urgency
        "fainting", "chest pain", "bleeding", "emergency room"
    }
    SENTIMENT_NEG_THRESHOLD = 0.80  # if NEGATIVE score > 0.8 → escalate
    ESCALATION_MESSAGE = (
        "It sounds like you might need **human support** right now. "
        "I can help you **escalate this to a student advisor**.\n\n"
        "Please book an appointment here: **https://example.edu/book-advisor**\n\n"
        "If this is an **emergency**, contact local emergency services or a crisis line immediately."
    )

    def __init__(self):
        self.vector_db_helper, self.llm_helper = _initialize_backends()
        self.ranker = NextQuestionLR(model_path="models/nextq_lr.joblib")
        self.sentiment_analyzer = _load_sentiment_pipeline()

    # ---- Escalation gate ----
    def check_escalation_needed(self, text: str) -> bool:
        """Return True if we should escalate to a human (skip KB/LLM)."""
        t = (text or "").lower()

        # 1) Keyword heuristic (fast, robust)
        for kw in self.ESCALATION_KEYWORDS:
            if kw in t:
                return True

        # 2) Sentiment model (if available)
        if self.sentiment_analyzer:
            try:
                res = self.sentiment_analyzer(text)[0]
                # Some models return labels like 'LABEL_0'/'NEGATIVE'; normalize
                label = str(res.get("label", "")).upper()
                score = float(res.get("score", 0.0))
                if ("NEG" in label or "NEGATIVE" in label) and score >= self.SENTIMENT_NEG_THRESHOLD:
                    return True
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        return False

    # ---- Suggestions ----
    def make_suggestions(self, user_text, vresp):
        cands_src = getattr(vresp, "vector_results", []) or []
        cands = []
        for it in cands_src:
            q = getattr(it, "question", "")
            if isinstance(q, str) and q.strip():
                dist = float(getattr(it, "confidence", 0.0) or 0.0)
                cands.append(Candidate(question=q, distance=dist))
        if not cands:
            return []

        try:
            order = self.ranker.rerank(user_text, cands, top_k=6) or []
        except Exception:
            order = list(range(min(6, len(cands))))

        suggestions = []
        ul = user_text.lower().strip()
        for idx in order:
            q = cands[idx].question
            if q and q.lower().strip() != ul:
                q = q.replace("Conestoga", "your").replace("conestoga", "your")
                suggestions.append(q)
        return suggestions

    # ---- Main turn handler ----
    def ask_and_respond(self, user_text: str):
        """Main turn handler. Updates session state and triggers a rerender."""
        ensure_state()

        # 1) append user
        st.session_state.messages.append({"role": "user", "content": user_text, "ts": _now()})

        # 2) ESCALATION CHECK (before any KB/LLM work)
        if self.check_escalation_needed(user_text):
            st.session_state.is_typing = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": self.ESCALATION_MESSAGE,
                "ts": _now()
            })
            # Clear suggestions when escalating
            st.session_state.next_questions = []
            st.rerun()
            return

        # 3) typing on
        st.session_state.is_typing = True

        # 4) KB route + LLM
        vresp = self.vector_db_helper.handle_query(user_text)
        if vresp.prompt_type == PromptType.FAQ_MATCH:
            bot_reply = self.llm_helper.ask_llm(user_query=user_text, vector_result=vresp.vector_results)
        else:
            bot_reply = vresp.message

        # 5) typing off, append bot
        st.session_state.is_typing = False
        st.session_state.messages.append({"role": "assistant", "content": bot_reply, "ts": _now()})

        # 6) suggestions
        st.session_state.next_questions = self.make_suggestions(user_text, vresp)

        # 7) show latest
        st.rerun()

    def clear(self):
        ensure_state()
        st.session_state.messages = []
        st.session_state.next_questions = []
        st.session_state.is_typing = False
        st.rerun()
