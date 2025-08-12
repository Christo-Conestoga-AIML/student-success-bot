import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from data_class.prompt_response import PromptType
from knowledge_base.knowledge_base_helper import VectorDBHelper
from knowledge_base.kb_builder import VectorDbBuilder
from llm_helper.llm_helper import LLMHelper
from utils.constants import Constants
from scripts.next_question_ranker import NextQuestionLR, Candidate


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


class ChatController:
    def __init__(self):
        self.vector_db_helper, self.llm_helper = _initialize_backends()
        # lazy, lightweight handle to LR ranker (loads model on first call)
        self.ranker = NextQuestionLR(model_path="models/nextq_lr.joblib")

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

        # Try LR re-ranking; fall back to FAISS order if anything goes wrong.
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

    def ask_and_respond(self, user_text: str):
        """Main turn handler. Updates session state and triggers a rerender."""
        ensure_state()

        # 1) append user
        st.session_state.messages.append({"role": "user", "content": user_text, "ts": _now()})

        # 2) typing on
        st.session_state.is_typing = True

        # 3) KB route + LLM
        vresp = self.vector_db_helper.handle_query(user_text)
        if vresp.prompt_type == PromptType.FAQ_MATCH:
            bot_reply = self.llm_helper.ask_llm(user_query=user_text, vector_result=vresp.vector_results)
        else:
            bot_reply = vresp.message

        # 4) typing off, append bot
        st.session_state.is_typing = False
        st.session_state.messages.append({"role": "assistant", "content": bot_reply, "ts": _now()})

        # 5) suggestions
        st.session_state.next_questions = self.make_suggestions(user_text, vresp)

        # 6) show latest
        st.rerun()

    def clear(self):
        ensure_state()
        st.session_state.messages = []
        st.session_state.next_questions = []
        st.session_state.is_typing = False
        st.rerun()
