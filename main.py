import os, base64
import streamlit as st
from dotenv import load_dotenv

from data_class.prompt_response import PromptType
from knowledge_base.knowledge_base_helper import VectorDBHelper
from knowledge_base.kb_builder import VectorDbBuilder
from llm_helper.llm_helper import LLMHelper
from utils.constants import Constants
from scripts.next_question_ranker import NextQuestionLR, Candidate

# ---------- Init ----------
@st.cache_resource
def initialize():
    if Constants.should_build_vector_db():
        VectorDbBuilder.build_db()
    load_dotenv()
    vdb = VectorDBHelper()
    vdb.load_kb()
    llm = LLMHelper(context_buffer=[])
    return vdb, llm

def make_suggestions(user_text, vresp):
    suggestions = []
    cands_src = getattr(vresp, "vector_results", []) or []
    cands = [
        Candidate(question=it.question, distance=getattr(it, "confidence", 0.0))
        for it in cands_src
        if isinstance(getattr(it, "question", None), str)
    ]
    ranker = NextQuestionLR(model_path="models/nextq_lr.joblib")
    order = ranker.rerank(user_text, cands, top_k=6)
    for idx in order:
        q = cands[idx].question
        if q.lower() != user_text.lower():
            q = q.replace("Conestoga", "your").replace("conestoga", "your")
            suggestions.append(q)
    return suggestions

st.set_page_config(page_title="ABC Student Support Chatbot", page_icon="🎓", layout="centered")
vector_db_helper, llm_helper = initialize()

# ---------- State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"|"assistant", "content": str}]
if "next_questions" not in st.session_state:
    st.session_state.next_questions = []

# ---------- Assets ----------
def _b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

STUDENT_B64 = _b64("images/student.png")
BOT_B64 = _b64("images/bot.png")

# ---------- Styles ----------
st.markdown("""
<style>
.block-container {max-width: 920px; padding-top: 1.5rem;}
/* Chat Card */
.chat-card {
  border: 1px solid rgba(120,120,120,.18);
  background: rgba(40,44,52,.55);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.28);
  padding: 8px 10px 8px 10px;
}
/* Header */
.chat-header {
  padding: 8px 4px 14px 4px;
  font-size: 32px; font-weight: 800;
}
/* Body */
.chat-body {
  padding: 6px 10px 10px 10px;
  background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
}
/* Row + alignment */
.chat-row{display:flex; gap:10px; margin:10px 8px; align-items:flex-end;}
.chat-row.user{justify-content:flex-end;}
.chat-row.bot{justify-content:flex-start;}
/* Avatar */
.chat-avatar{width:36px;height:36px;border-radius:50%;border:1px solid rgba(255,255,255,.18);object-fit:cover;}
/* Bubble */
.bubble{max-width:72%; padding:12px 14px; border-radius:16px; line-height:1.45; font-size:16px; white-space:pre-wrap; word-wrap:break-word;}
.bubble.user{background:#1f6feb21; border:1px solid #1f6feb55; border-bottom-right-radius:8px;}
.bubble.bot{background:#2b303b; border:1px solid #3a3f4a; border-bottom-left-radius:8px;}
/* Footer */
.chat-footer{padding:10px 10px 12px 10px; border-top:1px solid rgba(120,120,120,.18); background: rgba(255,255,255,.02);}
.suggestion-title{font-weight:700; font-size:18px; margin:10px 0 6px 2px;}
</style>
""", unsafe_allow_html=True)

# ---------- Logic ----------
def ask_and_respond(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})

    vresp = vector_db_helper.handle_query(user_text)
    if vresp.prompt_type == PromptType.FAQ_MATCH:
        bot_reply = llm_helper.ask_llm(user_query=user_text, vector_result=vresp.vector_results)
    else:
        bot_reply = vresp.message

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # NEW: use LR re-ranker to build suggestions
    st.session_state.next_questions = make_suggestions(user_text, vresp)

    st.rerun()

def render_bubble(role: str, content: str):
    if role == "user":
        right = f"""
        <div class="chat-row user">
          <div class="bubble user">{content}</div>
          <img class="chat-avatar" src="data:image/png;base64,{STUDENT_B64}"/>
        </div>"""
        st.markdown(right, unsafe_allow_html=True)
    else:
        left = f"""
        <div class="chat-row bot">
          <img class="chat-avatar" src="data:image/png;base64,{BOT_B64}"/>
          <div class="bubble bot">{content}</div>
        </div>"""
        st.markdown(left, unsafe_allow_html=True)

# ---------- UI ----------
st.markdown('<div class="chat-card">', unsafe_allow_html=True)
st.markdown('<div class="chat-header">ABC Student Support Chatbot</div>', unsafe_allow_html=True)

# Body (inside the highlighted card)
st.markdown('<div class="chat-body">', unsafe_allow_html=True)
if not st.session_state.messages:
    render_bubble("assistant", "Hi! Ask me about fees, registration, schedules, or student services.")
else:
    for m in st.session_state.messages:
        render_bubble("assistant" if m["role"] == "assistant" else "user", m["content"])
st.markdown('</div>', unsafe_allow_html=True)

# Footer (suggestions + clear)
with st.container():
    if st.session_state.next_questions:
        st.markdown('<div class="suggestion-title">Related questions you might want to ask:</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, q in enumerate(st.session_state.next_questions[:6]):
            with cols[i % 3]:
                if st.button(q, key=f"suggest_{i}"):
                    ask_and_respond(q)

    c1, c2 = st.columns([1,6])
    with c1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.next_questions = []
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # close chat-card

# Native bottom input (stays under the card)
prompt = st.chat_input("Type your message and press Enter")
if prompt:
    ask_and_respond(prompt)
