import os
from typing import List

from openai import OpenAI

from data_class.open_ap_chat_model import OpenAPChatModel
from data_class.prompt_response import KbQuestionAnswer


class LLMHelper:
    def __init__(self, context_buffer: List[OpenAPChatModel]):
        self.context_buffer:List[OpenAPChatModel] = context_buffer
        if self.context_buffer is None:
            self.context_buffer = [OpenAPChatModel(prompt='You are a helpful student support assistant. Use the provided context to answer the question.',response='ok',vector_results=[])]
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



    def ask_llm(self, user_query: str, vector_result: List[KbQuestionAnswer]) -> str:
        messages = []

        for item in self.context_buffer:
            messages.append({"role": "user", "content": item.prompt})
            messages.append({"role": "system", "content": item.response})

        json_vector_results = []

        for item in vector_result:
            entry = {
                "question": item.question,
                "answer": item.answer
            }
            json_vector_results.append(entry)


        messages.append({"role": "user", "content": str(json_vector_results)+ user_query})

        response = self.openai.responses.create(model="gpt-3.5-turbo",input=messages)

        output_text = response.output_text.replace("Conestoga", "your").replace("conestoga", "your")

        self.context_buffer.append(OpenAPChatModel(prompt=user_query,response=output_text,vector_results=vector_result))

        return output_text