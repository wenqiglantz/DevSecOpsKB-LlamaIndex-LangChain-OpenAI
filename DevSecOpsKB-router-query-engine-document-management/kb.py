from llama_index import (
    VectorStoreIndex,
    ListIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
)
from llama_index.indices.loading import load_index_from_storage
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai
import gradio as gr
import sys, os
import logging
import atexit
import uuid

# loads dotenv lib to retrieve API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# set number of output tokens from the LLM. Typically this is set automatically with the model metadata. It doesn't limit the model output, but affects the amount of “space” we save for the output, when computing available context window size for packing text from retrieved Nodes
num_output = 512

# LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_output))

# define LLM service context
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=1024)

# file path for storing the variables for list_id and vector_id
variables_file = "variables.txt"

def load_variables():
    global list_id, vector_id

    # check if the variables file exists
    if os.path.isfile(variables_file):
        with open(variables_file, "r") as file:
            # read the values from the file
            values = file.read().split(",")
            list_id = values[0]
            vector_id = values[1]

def save_variables():
    global list_id, vector_id

    # write the values to the file
    with open(variables_file, "w") as file:
        file.write(f"{list_id},{vector_id}")
        
def load_index(directory_path):
    # declare the variables as global
    global list_id, vector_id  
    
    # load data
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    print(f"loaded {len(documents)} documents")
    
    # get nodes
    nodes = service_context.node_parser.get_nodes_from_documents(documents)

    try:
        # retrieve existing storage context and load list_index and vector_index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        list_index = load_index_from_storage(storage_context=storage_context, index_id=list_id)
        vector_index = load_index_from_storage(storage_context=storage_context, index_id=vector_id)
        logging.info("list_index and vector_index loaded")
        
    except FileNotFoundError:
        logging.info("storage context not found. Add nodes to docstore")
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        
        # construct list_index and vector_index from storage_context and service_context
        list_index = ListIndex(nodes, storage_context=storage_context, service_context=service_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context)

        # persist both indexes to disk
        list_index.storage_context.persist()
        vector_index.storage_context.persist()

        # update the global variables of list_id and vector_id
        list_id = list_index.index_id
        vector_id = vector_index.index_id

        # save the variables to the file
        save_variables()

    # define list_query_engine and vector_query_engine
    list_query_engine = list_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    # build list_tool and vector_tool
    list_tool = QueryEngineTool.from_defaults(
        query_engine=list_query_engine,
        description="Useful for summarization questions on DevSecOps tooling.",
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context on DevSecOps tooling.",
    )

    # construct RouterQueryEngine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            list_tool,
            vector_tool,
        ]
    )
    
    # run refresh_ref_docs function to check for document updates
    list_refreshed_docs = list_index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
    print(list_refreshed_docs)
    print('Number of newly inserted/refreshed docs: ', sum(list_refreshed_docs))

    list_index.storage_context.persist()
    logging.info("list_index refreshed and persisted to storage.")

    vector_refreshed_docs = vector_index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
    print(vector_refreshed_docs)
    print('Number of newly inserted/refreshed docs: ', sum(vector_refreshed_docs))

    vector_index.storage_context.persist()
    logging.info("vector_index refreshed and persisted to storage.")

    return query_engine
    

def data_querying(input_text):

    # Load index
    query_engine = load_index("data")

    #queries the index with the input text
    response = query_engine.query(input_text)
    
    return response.response

# load the variables at app startup
load_variables()

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Wenqi's DevSecOps Knowledge Base")

# save the variables before app exit
atexit.register(save_variables)

iface.launch(share=False)
