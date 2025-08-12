# app.py
import os, base64
import streamlit as st

from controller import ChatController, ensure_state  # logic only
# FIX: your file is `translation.py` at project root, not scripts/translation
from scripts.translation import translate_text  # ← CHANGED

st.set_page_config(page_title="ABC Student Support Chatbot", page_icon="🎓", layout="centered")

# ---------- State & Controller ----------
ensure_state()
controller = ChatController()

# ---------- Sidebar: Language Switcher ----------
LANG_LABELS = {"en": "English", "fr": "Français", "hi": "हिन्दी"}
with st.sidebar:
    st.header("Language")
    # keep persistent selection
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    lang = st.radio(
        "Display language",
        options=list(LANG_LABELS.keys()),
        index=list(LANG_LABELS.keys()).index(st.session_state.lang),
        format_func=lambda k: LANG_LABELS[k],
    )
    st.session_state.lang = lang
    st.caption("Bot answers and suggestions will be shown in the selected language.")

# ---------- Assets (avatars) ----------
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
/* Timestamp */
.msg-time{opacity:.6; font-size:12px; margin-top:4px;}
/* Footer */
.chat-footer{padding:10px 10px 12px 10px; border-top:1px solid rgba(120,120,120,.18); background: rgba(255,255,255,.02);}
.suggestion-title{font-weight:700; font-size:18px; margin:10px 0 6px 2px;}
/* Typing dots */
.typing {display:inline-block; min-width:28px}
.typing span{display:inline-block; width:6px; height:6px; border-radius:50%; margin:0 2px; background:#cbd5e1; animation: blink 1.2s infinite;}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
</style>
""", unsafe_allow_html=True)

# ---------- Small UI helpers ----------
def _t(text: str) -> str:
    """Translate helper for bot-facing strings based on chosen lang."""
    try:
        return translate_text(text, st.session_state.lang)
    except Exception:
        return text  # graceful fallback

def render_bubble(role: str, content: str, ts: str | None = None):
    # Only translate the bot messages; keep user's text as typed
    if role == "assistant":
        content = _t(content)

    time_html = f'<div class="msg-time">{ts}</div>' if ts else ""
    if role == "user":
        st.markdown(f"""
        <div class="chat-row user">
          <div class="bubble user">{content}{time_html}</div>
          <img class="chat-avatar" src="data:image/png;base64,{STUDENT_B64}"/>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row bot">
          <img class="chat-avatar" src="data:image/png;base64,{BOT_B64}"/>
          <div class="bubble bot">{content}{time_html}</div>
        </div>""", unsafe_allow_html=True)

def render_typing():
    st.markdown(f"""
    <div class="chat-row bot">
      <img class="chat-avatar" src="data:image/png;base64,{BOT_B64}"/>
      <div class="bubble bot"><span class="typing"><span></span><span></span><span></span></span></div>
    </div>""", unsafe_allow_html=True)

# ---------- UI ----------
st.markdown('<div class="chat-card">', unsafe_allow_html=True)
st.markdown('<div class="chat-header">ABC Student Support Chatbot</div>', unsafe_allow_html=True)

# Body
st.markdown('<div class="chat-body">', unsafe_allow_html=True)
if not st.session_state.messages:
    welcome = "Hi! Ask me about fees, registration, schedules, or student services."
    render_bubble("assistant", _t(welcome), None)  # translated welcome
else:
    for m in st.session_state.messages:
        render_bubble("assistant" if m["role"] == "assistant" else "user", m["content"], m.get("ts"))
    if st.session_state.is_typing:
        render_typing()
st.markdown('</div>', unsafe_allow_html=True)

# Footer (suggestions + clear)
with st.container():
    if st.session_state.next_questions:
        st.markdown(f"<div class='suggestion-title'>{_t('Related questions you might want to ask:')}</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, q in enumerate(st.session_state.next_questions[:6]):
            # Show translated label, but send the original English query to controller
            display_q = _t(q)
            with cols[i % 3]:
                if st.button(display_q, key=f"suggest_{i}"):
                    controller.ask_and_respond(q)

st.markdown('</div>', unsafe_allow_html=True)  # close chat-card

# Bottom input
prompt = st.chat_input("Type your message and press Enter")
if prompt:
    controller.ask_and_respond(prompt)
