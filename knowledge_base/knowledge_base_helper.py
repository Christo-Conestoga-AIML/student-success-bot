import pickle
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from data_class.prompt_response import PromptResponse, KbQuestionAnswer
from utils.constants import Constants


class KnowledgeBaseHelper:

    knowledge_base_index = None
    knowledge_base_model = None
    knowledge_base_meta_data = None

    def load_kb(self):
        self.knowledge_base_index = faiss.read_index("data/generated/faq_index.faiss")
        with open("data/generated/faq_metadata.pkl", "rb") as f:
            self.knowledge_base_meta_data = pickle.load(f)

        # Load embedding model
        self.knowledge_base_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_faq_response_from_kb(self, user_query, top_k=Constants.default_kb_count())->List[KbQuestionAnswer]:
        query_vector = self.knowledge_base_model.encode([user_query])
        distances, indices = self.knowledge_base_index.search(query_vector, top_k)

        responses = []
        for i in range(top_k):
            idx = indices[0][i]
            score = distances[0][i]
            faq_entry = self.knowledge_base_meta_data[idx]

            response = KbQuestionAnswer(
                question=faq_entry["Question"],
                answer=faq_entry["Answer"],
                confidence=float(score)
            )
            responses.append(response)

        return responses


    def handle_query(self, user_query: str, confidence_threshold: float = Constants.confidence_for_not_related()) -> PromptResponse:
        query_lower = user_query.lower()

        # Step 1: Check if human support is needed (keyword based)
        human_keywords = [
            "can't log in", "forgot password", "access denied",
            "not working", "appointment", "reset password", "disappointed","disappointing", "frustrated",
        ]

        if any(kw in query_lower for kw in human_keywords):
            return PromptResponse.human_support()

        # Step 2: Get top KB matches (top_k can be >1 for richer responses)
        kb_answers = self.get_faq_response_from_kb(user_query, top_k=Constants.default_kb_count())

        # Step 3: Check confidence of best match
        best_confidence = kb_answers[0].confidence if kb_answers else float('inf')

        if best_confidence < confidence_threshold:
            # Return all retrieved matches as kb_response with FAQ_MATCH prompt type
            return PromptResponse.from_kb(kb_answers, best_confidence)
        else:
            # Low confidence fallback
            return PromptResponse.low_confidence()

