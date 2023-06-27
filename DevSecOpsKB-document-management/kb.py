from llama_index import SimpleDirectoryReader, LLMPredictor, GPTVectorStoreIndex, load_index_from_storage
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.service_context import ServiceContext
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai
import gradio as gr
import sys, os
import logging

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#define LLM service
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

#set the global service context object, avoiding passing service_context when building the index 
from llama_index import set_global_service_context
set_global_service_context(service_context)

def load_index(directory_path):
     
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    print(f"loaded documents with {len(documents)} pages")
    
    try:
        # Rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # Try to load the index from storage
        index = load_index_from_storage(storage_context)
        logging.info("Index loaded from storage.")
    except FileNotFoundError:
        # If index not found, create a new one
        logging.info("Index not found. Creating a new one...")
        index = GPTVectorStoreIndex.from_documents(documents)
        # Persist index to disk
        index.storage_context.persist()
        logging.info("New index created and persisted to storage.")

    # Run refresh_ref_docs method to check for document updates
    refreshed_docs = index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
    print(refreshed_docs)
    print('Number of newly inserted/refreshed docs: ', sum(refreshed_docs))

    index.storage_context.persist()
    logging.info("Index refreshed and persisted to storage.")

    return index


def data_querying(input_text):

    # Load index
    index = load_index("data")

    #queries the index with the input text
    response = index.as_query_engine().query(input_text)
    
    return response.response

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Wenqi's DevSecOps Knowledge Base")

iface.launch(share=False)
