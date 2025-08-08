from data_class.prompt_response import PromptType
from knowledge_base.knowledge_base_helper import VectorDBHelper
from llm_helper.llm_helper import LLMHelper
from utils.constants import Constants
from knowledge_base.kb_builder import VectorDbBuilder
from dotenv import load_dotenv
import streamlit as st

def main():

    # Build the knowledge base if required(Change boolean value in constants.py if needed)
    if Constants.should_build_vector_db():
        VectorDbBuilder.build_db()

    load_dotenv()

    # Load the knowledge base
    print('Loading knowledge base...')
    vector_db_helper = VectorDBHelper()
    vector_db_helper.load_kb()
    print('Done')


    llm_helper = LLMHelper(context_buffer=[])


    # Example user query
    user_query = "How to i pay my fees?"
    vector_search_response = vector_db_helper.handle_query(user_query)

    if vector_search_response.prompt_type == PromptType.FAQ_MATCH:

        llm_response = llm_helper.ask_llm(user_query=user_query, vector_result=vector_search_response.vector_results)
        print(f"LLM Response: {llm_response}")

    else:
        print(vector_search_response.message)



if __name__ =="__main__":
    main()