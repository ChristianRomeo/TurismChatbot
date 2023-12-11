import glob
import os
import shutil
import time
import g4f
from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
import pandas as pd
import dask.dataframe as dd
from llama_index import PromptTemplate, VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import pinecone
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import PineconeVectorStore
import torch
from llama_index.llms import HuggingFaceLLM

system_prompt = (
    "This is a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
)

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.6, system_prompt=system_prompt))

# ask the user if he wants to delete the storage and compute it again
print("Do you want to delete the storage and compute it again? (y/n)")
if input()== "y" and os.path.exists("./storage"):
    shutil.rmtree("./storage")

PINECONE_KEY = os.environ.get('PINECONE_KEY')
index_name = "chatbot"

pinecone.init(api_key=PINECONE_KEY, environment="gcp-starter")
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536,  # 1536 dim of text-embedding-ada-002
        #metadata_config={'indexed': ['wiki-id', 'title']}
    )
else:
    pinecone.delete_index(index_name)
# wait for index to be initialized
#while not pinecone.describe_index(index_name).status['ready']: # type: ignore
time.sleep(1)
    

if not os.path.exists("./storage"):
    documents = []

    # Get a list of all CSV files in the directory
    csv_files = glob.glob('./data/*.csv')

    storage_context = StorageContext.from_defaults(
        vector_store=PineconeVectorStore(pinecone.Index(index_name))
    )

    for file in csv_files:
        #df = dd.read_csv(file, dtype=str, encoding="mac_roman")
        # Load the CSV file into a DataFrame
        chunksize = 100000 # Adjust this value based on your system's capabilities
        for chunk in pd.read_csv(file, dtype=str, encoding="mac_roman", parse_dates=True, chunksize=chunksize):
            print("hjello")
            # Convert the DataFrame into a list of Document objects
            docs = [Document(doc_id=str(i), text=row.to_string()) for i, row in chunk.iterrows()]
            # Add the documents to the list
            documents.extend(docs)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context, show_progress=True)
    print("hello")
    storage_context.persist(persist_dir=f"./storage")
    print("hwello")

else:
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=f"./storage"), service_context=service_context)

chat_engine = index.as_chat_engine(verbose=True)

print(chat_engine.chat(input("Please enter your question: ")).response)