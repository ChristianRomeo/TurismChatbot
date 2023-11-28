'''
DOCUMENT_SOURCE_DIRECTORY = r"C:\Users\chris\OneDrive\Desktop\EIT\2°Year\project\data"
# Target root files directory to for documents to load
loader = DirectoryLoader(r"C:\Users\chris\OneDrive\Desktop\EIT\2°Year\project\data", glob='*')
# Load documents
documents = loader.load()
# Split documents into chunks before creating embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
docs = text_splitter.split_documents(documents)

endpoint_url = "https://data.odeuropa.eu/sparql"
query = """
SELECT *
WHERE {
  ?s ?p ?o.
}
LIMIT 100
"""

def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    query = endpoint_url + "?" + urlencode({"query": query})
    sparql.setQuery(query)
    sparql.setReturnFormat(XML)
    return sparql.query().convert()

results = get_results(endpoint_url, query)


data = results["results"]["bindings"]
# Assuming that the data contains 'subject', 'predicate', and 'object'.

df = pd.DataFrame(columns=["subject", "predicate", "object"])
for row in data:
    df = df.append({
        "subject": row["subject"]["value"],
        "predicate": row["predicate"]["value"],
        "object": row["object"]["value"]
    }, ignore_index=True)

index_name = 'odeuropa'
vector_dimension = 1536
# Check if the index exists, if not, create one
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=vector_dimension)

# Connect to the index
index = pinecone.Index(index_name)


def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-similarity-babbage-001",  # For example, can use any available model
    )
    return response["data"][0]["embedding"]

# Example: Generate embeddings for a series texts (assume `df["object"]` contains text)
embeddings = df["object"].apply(lambda x: get_embedding(x))

# Next, you'd upsert these embeddings into Pinecone. You'll want to pair each vector with a unique ID.
for idx, embedding in enumerate(embeddings):
    index.upsert(items=[(str(idx), embedding)])

def chatbot_response(user_input):
    embedding = get_embedding(user_input)
    query_results = index.query(queries=[embedding], top_k=1)
    most_relevant_id = query_results["matches"][0]["id"]
    most_relevant_text = df.loc[int(most_relevant_id), "object"]
    
    response = openai.Completion.create(
        prompt=f"Based on the following knowledge: {most_relevant_text}. How would you respond to: {user_input}?",
        model="gpt-3.5-turbo",  # Assuming using OpenAI's GPT-3.5 model. Replace with appropriate model ID
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example usage:
user_message = "Tell me about historical odors in Europe."
print(chatbot_response(user_message))

sparql = SPARQLWrapper("https://data.odeuropa.eu/sparql")
sparql.setQuery("""
SELECT *
WHERE {
    ?s ?p ?o .
}
LIMIT 100
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

print (pinecone_datasets.list_datasets())

dataset = pinecone_datasets.load_dataset('wikipedia-simple-text-embedding-ada-002')
# we drop sparse_values as they are not needed for this example
dataset.documents.drop(['sparse_values', 'metadata'], axis=1, inplace=True)
dataset.documents.rename(columns={'blob': 'metadata'}, inplace=True)
# we will use rows of the dataset up to index 30_000
dataset.documents.drop(dataset.documents.index[30_000:], inplace=True)

index_name = 'chatbot-onboarding'

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536,  # 1536 dim of text-embedding-ada-002
        metadata_config={'indexed': ['wiki-id', 'title']}
    )
    # wait a moment for the index to be fully initialized
    time.sleep(1)

index = pinecone.GRPCIndex(index_name)
# wait a moment for the index to be fully initialized
time.sleep(1)

index.describe_index_stats()

index.upsert_from_dataframe(dataset.documents, batch_size=100)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

query_text = "What is machine learning?"
query_embedding = get_embedding(query_text)

# Query Pinecone for similar vectors
top_k = 5  # number of closest vectors to return
results = index.query(queries=[query_embedding], top_k=top_k)

# 'results' now contains the ids and distances of the top_k closest vectors to the query_embedding
for result in results.results:
    print(f"ID: {result.id}, Distance: {result.distance}")

# Initialize the model
model = g4f.models.gpt_4  # or another model you want to use

# Create a chat completion
response = g4f.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": query_text}],
)

# 'response' now contains the model's response to the query_text
print(response)

index_name = 'odeon-europa'
model_name = 'text-embedding-davinci-002'
embed = OpenAIEmbeddings(
    document_model_name=model_name,
    query_model_name=model_name,
    openai_api_key=OPENAI_API_KEY
)
embed_dimension = 1536

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=embed_dimension
    )
index = pinecone.Index(index_name)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

batch_limit = 100
texts = []
metadatas = []

# Fetch and process your data here. The following is a placeholder.
data = [{'id': 1, 'url': 'https://data.odeuropa.eu/sparql', 'title': 'Sample Title', 'content': 'Sample Content'}]


for i, record in enumerate(tqdm(data)):
    metadata = {
        'item_uuid': str(record['id']),
        'source': record['url'],
        'title': record['title']
    }
    record_texts = text_splitter.split_text(record['content'])
    record_metadatas = [{ "chunk": j, "text": text, **metadata } for j, text in enumerate(record_texts)]
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)

    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []
'''