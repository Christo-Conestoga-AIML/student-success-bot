from typing import List

from data_class.prompt_response import KbQuestionAnswer


class OpenAPChatModel:
    def __init__(self, prompt: str, response: str, vector_results: List[KbQuestionAnswer]):
        self.prompt = prompt
        self.response = response
        self.vector_results = vector_results
