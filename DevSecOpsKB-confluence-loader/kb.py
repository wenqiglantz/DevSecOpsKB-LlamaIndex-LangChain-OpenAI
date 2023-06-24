from llama_index import LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_hub.confluence.base import ConfluenceReader
from langchain.chat_models import ChatOpenAI
import pinecone
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from dotenv import load_dotenv
import gradio as gr
import os, sys
import graphsignal
import logging

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#configures Graphsignal tracer
graphsignal.configure(api_key=os.environ.get("GRAPHSIGNAL_API_KEY"), deployment='DevSecOpsKB')

# init pinecone
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment="asia-southeast1-gcp-free")

# set the size of the context window of the LLM. Typically this is set automatically with the model metadata. We can also explicitly override via this parameter for additional control
context_window = 4096
# set number of output tokens from the LLM. Typically this is set automatically with the model metadata. It doesn't limit the model output, but affects the amount of “space” we save for the output, when computing available context window size for packing text from retrieved Nodes
num_output = 512

#LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_output))

#constructs service_context
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                                context_window=context_window,
                                                num_output=num_output)

#set the global service context object, avoiding passing service_context when building the index or when loading index from vector store
from llama_index import set_global_service_context
set_global_service_context(service_context)

def data_ingestion_indexing():

    # Confluence base url and space key
    base_url = "https://wenqiglantz.atlassian.net/wiki"
    space_key = "SD"

    # construct ConfluenceReader and load data from Confluence
    loader = ConfluenceReader(base_url=base_url)
    documents = loader.load_data(space_key=space_key, page_ids=[], include_attachments=True)

    # call pinecone to create index, dimension is for text-embedding-ada-002
    try:
        pinecone.create_index("confluence-wiki", dimension=1536, metric="cosine", pod_type="Starter")
    except Exception:
        # most likely index already exists
        pass
    pinecone_index = pinecone.Index("confluence-wiki")

    # build the PineconeVectorStore and GPTVectorStoreIndex
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return index

def data_querying(input_text):

    #queries the index with the input text
    response = index.as_query_engine().query(input_text)
    
    return response.response

#construct Gradio UI
iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=5, label="Enter your question"),
                     outputs="text",
                     title="Wenqi's DevSecOps Knowledge Base")

#calls data_ingestion_indexing for data ingestion and indexing
index = data_ingestion_indexing()

#launch Gradio UI, disable share, make app accessible on docker local network by setting server_name to "0.0.0.0".
iface.launch(share=False, server_name="0.0.0.0")
