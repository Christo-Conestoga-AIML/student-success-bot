import streamlit as st
from data_class.prompt_response import PromptType
from knowledge_base.knowledge_base_helper import VectorDBHelper
from llm_helper.llm_helper import LLMHelper
from utils.constants import Constants
from knowledge_base.kb_builder import VectorDbBuilder
from dotenv import load_dotenv
import os

@st.cache_resource
def initialize():
    if Constants.should_build_vector_db():
        VectorDbBuilder.build_db()
    load_dotenv()
    vector_db_helper = VectorDBHelper()
    vector_db_helper.load_kb()
    llm_helper = LLMHelper(context_buffer=[])
    return vector_db_helper, llm_helper

st.set_page_config(page_title="ABC Chatbot", page_icon="🎓")
st.title("🎓 ABC Student Support Chatbot")

vector_db_helper, llm_helper = initialize()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "next_questions" not in st.session_state:
    st.session_state.next_questions = []

if "set_input" not in st.session_state:
    st.session_state.set_input = ""

def clear_set_input():
    st.session_state.set_input = ""

# Controlled input field, value controlled by set_input
user_input = st.text_input(
    "Ask a question:",
    value=st.session_state.set_input,
    key="user_input",
    on_change=clear_set_input,
    label_visibility="collapsed"
)

def show_next_questions():
    if st.session_state.next_questions:
        st.markdown("### 💡 Related questions you might want to ask:")
        for i, question in enumerate(st.session_state.next_questions):
            if st.button(question, key=f"next_q_{i}"):
                # Update set_input to fill input box with suggestion
                st.session_state.set_input = question
                # No st.experimental_rerun() needed — Streamlit reruns automatically on button click!

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    vector_response = vector_db_helper.handle_query(user_input)

    if vector_response.prompt_type == PromptType.FAQ_MATCH:
        llm_response = llm_helper.ask_llm(
            user_query=user_input,
            vector_result=vector_response.vector_results
        )
        bot_reply = llm_response
    else:
        bot_reply = vector_response.message

    st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

    # Collect up to 3 next question suggestions (excluding the exact question)
    suggestions = []
    if hasattr(vector_response, "vector_results") and vector_response.vector_results:
        for ans in vector_response.vector_results:
            if ans.question.lower() != user_input.lower():
                ans.question = ans.question.replace("Conestoga", "your").replace("conestoga", "your")
                suggestions.append(ans.question)
            if len(suggestions) >= 3:
                break
    st.session_state.next_questions = suggestions

# Display chat messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"🧑‍🎓 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['content']}")

# Show suggested next questions
show_next_questions()

# Clear chat button
st.markdown("---")
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.next_questions = []
    st.session_state.set_input = ""
