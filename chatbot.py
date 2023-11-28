import os
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from langchain.text_splitter import CharacterTextSplitter

PINECONE_KEY = os.environ.get('PINECONE_KEY')
if PINECONE_KEY is None:
    raise ValueError("Pinecone key not found. Please set the PINECONE_KEY environment variable.")
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI key not found. Please set the OPENAI_API_KEY environment variable.")
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
if HUGGINGFACE_API_KEY is None:
    raise ValueError("Hugging Face key not found. Please set the HUGGINGFACE_API_KEY environment variable.")

index_name = "texst1"
pinecone.init(api_key=PINECONE_KEY, environment="gcp-starter")
model_name = 'text-embedding-ada-002'

system_prompt = (
    "This is a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
)

# Load a small test dataset from Hugging Face Hub
dataset = load_dataset("jayantdocplix/falcon-small-test-dataset")

# Convert the 'train' split of the dataset to a pandas DataFrame and take the first 50 rows
df = dataset['train'].to_pandas().head(50) # type: ignore

# Initialize a character-based text splitter
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(GPT2TokenizerFast.from_pretrained("gpt2-xl"), chunk_overlap=50)
#overlap: how many characters will be duplicated at the end of one chunk and the start of the next chunk. 
#           Helpful because it can prevent the model from missing or misinterpreting the context that might occur at the boundaries of chunks

# Apply the text splitter to each row of text in the DataFrame
df['text'] = df['text'].apply(text_splitter.split_text)

# Flatten the list of lists of text chunks into a single list
flattened_texts = [text for sublist in df['text'].tolist() for text in sublist]

# Embed the flattened texts using OpenAI embeddings
embed = OpenAIEmbeddings().embed_documents(flattened_texts)

index = pinecone.GRPCIndex(index_name)

items_to_upsert = [
   {
    # Each item to upsert contains a unique ID, the embedding values, and associated metadata
       "id": str(index),
       "values": embedding,
       "metadata": {"text": text}
   }
   for index, (embedding, text) in enumerate(zip(embed, df['text'].tolist()))
]

index.upsert(items_to_upsert)

def query_pinecone(question, top_k=5):
       # Generate the embedding for the question
       question_embedding = OpenAIEmbeddings().embed_query(question)

       # Query the Pinecone index with the question embedding to retrieve the top_k closest matches
       query_results = index.query(
           vector=question_embedding, 
           top_k=top_k, 
           include_metadata=True
       )

       # Extract the text metadata from the query results
       return [item['metadata']['text'] for item in query_results['matches']]

def generate_response(context, question):
    # Format the context and question into a prompt for the language model
    prompt = "\n".join([" ".join(sub_list) for sub_list in context]) + "\n\n" + question

    # Generate the response using OpenAI GPT-3.5 Turbo
    chat_completion = openai.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt,}
        ],
        model="gpt-3.5-turbo",
    )
    # Return the content of the chat completion, which is the model's response, stripped of leading/trailing whitespace
    if chat_completion.choices[0].message.content:
        return chat_completion.choices[0].message.content.strip()
    else:
         return "I'm sorry, I couldn't generate a response."


while True:
    # Read a line of input from the user
    question = input("Please enter your question: ")

    # Query Pinecone
    context = query_pinecone(question)

    # Generate a response
    response = generate_response(context, question)

    # Print the response
    print(response)


print("Complete")

# questions to evaluate
# 1. What is the best time to visit Paris?
#LLAMAINDEX for alternatives for report in case
# data processing

