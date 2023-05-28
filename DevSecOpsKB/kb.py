from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper, StorageContext, ServiceContext, GPTVectorStoreIndex, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = 'YOUR-OPENAI-API-KEY'

def create_service_context():

    #constraint parameters
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    #allows the user to explicitly set certain constraint parameters
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    #LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    #constructs service_context
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return service_context


def data_ingestion_indexing(directory_path):

    #loads data from the specified directory path
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    #when first building the index
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )

    #persist index to disk, default "storage" folder
    index.storage_context.persist()

    return index

def data_querying(input_text):

    #rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    #loads index from storage
    index = load_index_from_storage(storage_context, service_context=create_service_context())
    
    #queries the index with the input text
    response = index.as_query_engine().query(input_text)
    
    return response.response

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your question"),
                     outputs="text",
                     title="Wenqi's Custom-trained DevSecOps Knowledge Base")

#passes in data directory
index = data_ingestion_indexing("data")
iface.launch(share=False)
