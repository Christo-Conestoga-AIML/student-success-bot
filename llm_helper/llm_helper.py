import os
from typing import List
from openai import OpenAI
from data_class.open_ap_chat_model import OpenAPChatModel
from data_class.prompt_response import KbQuestionAnswer

SYSTEM_PROMPT = (
    "You are a helpful student support assistant. "
    "Use the provided FAQ context to answer concisely. "
    "If the answer is unclear or not in the context, say so briefly and suggest booking with a student advisor. "
    "Never mention internal KB or embeddings. "
    "Replace any institution-specific name with 'your college' in the final answer."
)

class LLMHelper:
    def __init__(self, context_buffer: List[OpenAPChatModel]):
        self.context_buffer: List[OpenAPChatModel] = context_buffer or []
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_llm(self, user_query: str, vector_result: List[KbQuestionAnswer]) -> str:
        # Build a compact, structured context block
        context_lines = []
        for i, item in enumerate(vector_result[:5], start=1):
            context_lines.append(f"[{i}] Q: {item.question}\nA: {item.answer}")
        context_block = "\n\n".join(context_lines) if context_lines else "No context."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nUser question: {user_query}\n\nAnswer:"}
        ]

        resp = self.openai.chat.completions.create(
            model="gpt-4o-mini",           # accurate + fast + cheap
            temperature=0.2,
            max_tokens=450,
            messages=messages,
        )

        output_text = resp.choices[0].message.content
        # neutralize institution name
        output_text = output_text.replace("Conestoga", "your college").replace("conestoga", "your college")

        # keep light conversation memory
        self.context_buffer.append(OpenAPChatModel(prompt=user_query, response=output_text, vector_results=vector_result))
        if len(self.context_buffer) > 10:
            self.context_buffer = self.context_buffer[-10:]

        return output_text
