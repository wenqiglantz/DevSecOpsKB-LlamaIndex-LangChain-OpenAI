from llama_index import SimpleDirectoryReader, LLMPredictor, StorageContext, ServiceContext, GPTVectorStoreIndex, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import gradio as gr
import os
import graphsignal

load_dotenv()

graphsignal.configure(api_key=os.getenv('GRAPHSIGNAL_API_KEY'), deployment='DevSecOpsKB')


# set context window
context_window = 4096
# set number of output tokens
num_output = 512

#LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_output))

#constructs service_context
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, context_window=context_window, num_output=num_output)

#set the global service context object
from llama_index import set_global_service_context
set_global_service_context(service_context)



def data_ingestion_indexing(directory_path):

    #loads data from the specified directory path
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    #when first building the index
    index = GPTVectorStoreIndex.from_documents(documents)

    #persist index to disk, default "storage" folder
    index.storage_context.persist()

    return index

def data_querying(input_text):

    #rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    #loads index from storage
    index = load_index_from_storage(storage_context)
    
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
