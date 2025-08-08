from huggingface_hub.utils._dotenv import load_dotenv

from data_class.prompt_response import PromptType
from knowledge_base.knowledge_base_helper import KnowledgeBaseHelper
from utils.constants import Constants
from knowledge_base.kb_builder import KnowledgeBaseBuilder


def main():

    # Build the knowledge base if required(Change boolean value in constants.py if needed)
    if Constants.should_build_kb():
        KnowledgeBaseBuilder.build_kb()

    load_dotenv()

    # Load the knowledge base
    knowledge_base_helper = KnowledgeBaseHelper()
    knowledge_base_helper.load_kb()



    # Example user query
    user_query = "I want an appointment with a student advisor"
    response = knowledge_base_helper.handle_query(user_query)

    if response.prompt_type == PromptType.FAQ_MATCH:
        for idx, kb_answer in enumerate(response.kb_response, start=1):
            print(f"{idx}. Q: {kb_answer.question}\n   A: {kb_answer.answer} (Confidence: {kb_answer.confidence:.4f})\n")
    else:
        print(response.message)





if __name__ =="__main__":
    main()