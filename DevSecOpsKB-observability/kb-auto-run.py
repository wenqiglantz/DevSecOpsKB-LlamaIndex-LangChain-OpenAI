from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import graphsignal
import logging
import time
import random

load_dotenv()

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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

#loads data from the specified directory path
documents = SimpleDirectoryReader("./data").load_data()

#when first building the index
index = GPTVectorStoreIndex.from_documents(documents)

def data_querying(input_text):
    
    #queries the index with the input text
    response = index.as_query_engine().query(input_text)
    
    return response.response
    

# predefine a list of 10 questions
questions = [
    'what does Trivy image scan do?',
    'What are the main benefits of using Harden Runner?',
    'What is the 3-2-1 rule in DevOps self-service model?',
    'What is Infracost?  and what does it do?',
    'What is the terraform command to auto generate README?',
    'How to pin Terraform module source to a particular branch?',
    'What are the benefits of reusable Terraform modules?',
    'How do I resolve error "npm ERR! code E400"?',
    'How to fix error "NoCredentialProviders: no valid providers in chain"?',
    'How to fix error "Credentials could not be loaded, please check your action inputs: Could not load credentials from any providers"?'
]

start_time = time.time()

while time.time() - start_time < 1800:  # let it run for 30 minutes (1800 seconds)
    try:
        num = random.randint(0, len(questions) - 1)
        print("Question: ", questions[num])
        answer = data_querying(questions[num])
        print("Answer: ", answer)
    except:
        logger.error("Error during data query", exc_info=True)

    time.sleep(5 * random.random())
    
    
