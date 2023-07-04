from llama_index import SimpleDirectoryReader, LLMPredictor, GPTVectorStoreIndex, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.service_context import ServiceContext
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pinecone
from llama_index.vector_stores import PineconeVectorStore
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

# init pinecone
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])

# set the size of the context window of the LLM. Typically this is set automatically with the model metadata. We can also explicitly override via this parameter for additional control
context_window = 4096
# set number of output tokens from the LLM. Typically this is set automatically with the model metadata. It doesn't limit the model output, but affects the amount of “space” we save for the output, when computing available context window size for packing text from retrieved Nodes
num_output = 512

#LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_output))

#define LLM service
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                context_window=context_window,
                                                num_output=num_output)

#set the global service context object, avoiding passing service_context when building the index 
from llama_index import set_global_service_context
set_global_service_context(service_context)

def load_index(directory_path):
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    print(f"loaded {len(documents)} documents")

    indexes = pinecone.list_indexes()
    print(indexes)
    if 'devsecops-wiki' not in indexes:
        logging.info("Index not found. Creating a new one...")
        pinecone.create_index("devsecops-wiki", dimension=1536, metric="cosine", pod_type="Starter")
        vector_store = PineconeVectorStore(pinecone.Index("devsecops-wiki"))
        storage_context = StorageContext.from_defaults(vector_store = vector_store)
        loaded_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logging.info("New index created and persisted to Pinecone.")
    else:
        logging.info("Loading index from Pinecone.")    
        vector_store = PineconeVectorStore(pinecone.Index("devsecops-wiki"))
        loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logging.info("Refreshing docs.")
        #refreshed_docs = loaded_index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
        refreshed_docs = loaded_index.refresh_ref_docs(documents) #TODO
        print(refreshed_docs)
        print('Number of newly inserted/refreshed docs: ', sum(refreshed_docs))

    return loaded_index
    

def data_querying(input_text):

    # Load index
    loaded_index = load_index("data")

    #queries the index with the input text
    response = loaded_index.as_query_engine().query(input_text)
    
    return response.response

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Wenqi's DevSecOps Knowledge Base")

iface.launch(share=False)
