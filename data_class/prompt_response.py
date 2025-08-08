from enum import Enum
from typing import List


class PromptType(Enum):
    FAQ_MATCH = "faq_match"
    HUMAN_SUPPORT = "human_support"
    LOW_CONFIDENCE = "low_confidence"


class KbQuestionAnswer:
    def __init__(self, question: str, answer: str, confidence: float):
        self.question = question
        self.answer = answer
        self.confidence = confidence


class VectorPromptResponse:
    def __init__(self, vector_result:List[KbQuestionAnswer], prompt_type: PromptType, message: str):
        self.vector_results = vector_result
        self.prompt_type=prompt_type
        self.message = message



    @staticmethod
    def from_kb(kb_response: List[KbQuestionAnswer], confidence: float) -> "VectorPromptResponse":
        return VectorPromptResponse(
            vector_result=kb_response,
            prompt_type=PromptType.FAQ_MATCH,
            message='Okay, I can help you with that.'
        )

    @staticmethod
    def low_confidence() -> "VectorPromptResponse":
        return VectorPromptResponse(prompt_type=PromptType.LOW_CONFIDENCE, vector_result=[],
                                    message="I'm not sure what you're asking for. Can you be more detailing?"
                                    )

    @staticmethod
    def human_support() -> "VectorPromptResponse":
        return VectorPromptResponse(
            vector_result=[],
            message="Please follow the given link to book an appointment with a student advisor: https://www.university.edu/student-advisors",
            prompt_type=PromptType.HUMAN_SUPPORT
        )



