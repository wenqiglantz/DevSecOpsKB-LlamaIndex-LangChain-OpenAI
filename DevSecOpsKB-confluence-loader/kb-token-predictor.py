# import both MockLLMPredictor and MockEmbedding classes
from llama_index import MockLLMPredictor, MockEmbedding, ServiceContext, GPTVectorStoreIndex
from llama_hub.confluence.base import ConfluenceReader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os, sys
import graphsignal
import logging

load_dotenv()

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Use Graphsignal for observability
graphsignal.configure(api_key=os.environ.get("GRAPHSIGNAL_API_KEY"), deployment='DevSecOpsKB-confluence')

# the "mock" llm predictor is our token counter
llm_predictor = MockLLMPredictor(max_tokens=256)

# specify a MockEmbedding
embed_model = MockEmbedding(embed_dim=1536)

#constructs service_context
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

#set the global service context object
from llama_index import set_global_service_context
set_global_service_context(service_context)

base_url = "https://wenqiglantz.atlassian.net/wiki"
space_key = "SD"

loader = ConfluenceReader(base_url=base_url)
documents = loader.load_data(space_key=space_key, page_ids=[], include_attachments=True)

#when first building the index
index = GPTVectorStoreIndex.from_documents(documents)

# get number of tokens used for index construction
print("token used for index construction: " + str(embed_model.last_token_usage))

#queries the index with the input text
index.as_query_engine().query("Is parallel processing in github actions a good practice?  Why?")

# get number of tokens used for query
print("token used for query: " + str(llm_predictor.last_token_usage))
