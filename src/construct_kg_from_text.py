#!/usr/bin/env python
# coding: utf-8

# ### Import packages and set up Neo4j

# In[1]:


from dotenv import load_dotenv
import os

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI


# Warning control
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Note the code below is unique to this course environment, and not a 
# standard part of Neo4j's integration with OpenAI. Remove if running 
# in your own environment.
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

# Global constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'


# ### Take a look at a Form 10-K json file
# 
# - Publicly traded companies are required to fill a form 10-K each year with the Securities and Exchange Commision (SEC)
# - You can search these filings using the SEC's [EDGAR database](https://www.sec.gov/edgar/search/)
# - For the next few lessons, you'll work with a single 10-K form for a company called [NetApp](https://www.netapp.com/)

# In[3]:


first_file_name = "./data/form10k/0000950170-23-027948.json"


# In[4]:


first_file_as_object = json.load(open(first_file_name))


# In[5]:


type(first_file_as_object)


# In[6]:


for k,v in first_file_as_object.items():
    print(k, type(v))


# In[7]:


item1_text = first_file_as_object['item1']


# In[8]:


item1_text[0:1500]


# ### Split Form 10-K sections into chunks
# - Set up text splitter using LangChain
# - For now, split only the text from the "item 1" section 

# In[9]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)


# In[10]:


item1_text_chunks = text_splitter.split_text(item1_text)


# In[11]:


type(item1_text_chunks)


# In[12]:


len(item1_text_chunks)


# In[13]:


item1_text_chunks[0]


# - Set up helper function to chunk all sections of the Form 10-K
# - You'll limit the number of chunks in each section to 20 to speed things up

# In[14]:


def split_form10k_data_from_file(file):
    chunks_with_metadata = [] # use this to accumlate chunk records
    file_as_object = json.load(open(file)) # open the json file
    for item in ['item1','item1a','item7','item7a']: # pull these keys from the json
        print(f'Processing {item} from {file}') 
        item_text = file_as_object[item] # grab the text of the item
        item_text_chunks = text_splitter.split_text(item_text) # split the text into chunks
        chunk_seq_id = 0
        for chunk in item_text_chunks[:20]: # only take the first 20 chunks
            form_id = file[file.rindex('/') + 1:file.rindex('.')] # extract form id from file name
            # finally, construct a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk, 
                # metadata from looping...
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                # constructed metadata...
                'formId': f'{form_id}', # pulled from the filename
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                # metadata from file...
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata


# In[15]:


first_file_chunks = split_form10k_data_from_file(first_file_name)


# In[16]:


first_file_chunks[0]


# ### Create graph nodes using text chunks

# In[17]:


merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET 
        mergedChunk.names = $chunkParam.names,
        mergedChunk.formId = $chunkParam.formId, 
        mergedChunk.cik = $chunkParam.cik, 
        mergedChunk.cusip6 = $chunkParam.cusip6, 
        mergedChunk.source = $chunkParam.source, 
        mergedChunk.f10kItem = $chunkParam.f10kItem, 
        mergedChunk.chunkSeqId = $chunkParam.chunkSeqId, 
        mergedChunk.text = $chunkParam.text
RETURN mergedChunk
"""


# - Set up connection to graph instance using LangChain

# In[18]:


kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


# - Create a single chunk node for now

# In[19]:


kg.query(merge_chunk_node_query, 
         params={'chunkParam':first_file_chunks[0]})


# - Create a uniqueness constraint to avoid duplicate chunks

# In[20]:


kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
""")


# In[21]:


kg.query("SHOW INDEXES")


# - Loop through and create nodes for all chunks
# - Should create 23 nodes because you set a limit of 20 chunks in the text splitting function above

# In[22]:


node_count = 0
for chunk in first_file_chunks:
    print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
    kg.query(merge_chunk_node_query, 
            params={
                'chunkParam': chunk
            })
    node_count += 1
print(f"Created {node_count} nodes")


# In[23]:


kg.query("""
         MATCH (n)
         RETURN count(n) as nodeCount
         """)


# ### Create a vector index

# In[24]:


kg.query("""
         CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}
""")


# In[25]:


kg.query("SHOW INDEXES")


# ### Calculate embedding vectors for chunks and populate index
# - This query calculates the embedding vector and stores it as a property called `textEmbedding` on each `Chunk` node.

# In[26]:


kg.query("""
    MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text, 
      "OpenAI", 
      {
        token: $openAiApiKey, 
        endpoint: $openAiEndpoint
      }) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """, 
    params={"openAiApiKey":OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )


# In[27]:


kg.refresh_schema()
print(kg.schema)


# ### Use similarity search to find relevant chunks

# - Setup a help function to perform similarity search using the vector index

# In[28]:


def neo4j_vector_search(question):
  """Search for similar nodes using the Neo4j vector index"""
  vector_search_query = """
    WITH genai.vector.encode(
      $question, 
      "OpenAI", 
      {
        token: $openAiApiKey,
        endpoint: $openAiEndpoint
      }) AS question_embedding
    CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
    RETURN score, node.text AS text
  """
  similar = kg.query(vector_search_query, 
                     params={
                      'question': question, 
                      'openAiApiKey':OPENAI_API_KEY,
                      'openAiEndpoint': OPENAI_ENDPOINT,
                      'index_name':VECTOR_INDEX_NAME, 
                      'top_k': 10})
  return similar


# - Ask a question!

# In[29]:


search_results = neo4j_vector_search(
    'In a single sentence, tell me about Netapp.'
)


# In[30]:


search_results[0]


# ### Set up a LangChain RAG workflow to chat with the form

# In[31]:


neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    text_node_properties=[VECTOR_SOURCE_PROPERTY],
    embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
)


# In[32]:


retriever = neo4j_vector_store.as_retriever()


# - Set up a RetrievalQAWithSourcesChain to carry out question answering
# - You can check out the LangChain documentation for this chain [here](https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html)

# In[33]:


chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever
)


# In[34]:


def prettychain(question: str) -> str:
    """Pretty print the chain's response to a question"""
    response = chain({"question": question},
        return_only_outputs=True,)
    print(textwrap.fill(response['answer'], 60))


# - Ask a question!

# In[35]:


question = "What is Netapp's primary business?"


# In[36]:


prettychain(question)


# In[37]:


prettychain("Where is Netapp headquartered?")


# In[38]:


prettychain("""
    Tell me about Netapp. 
    Limit your answer to a single sentence.
""")


# In[39]:


prettychain("""
    Tell me about Apple. 
    Limit your answer to a single sentence.
""")


# In[42]:


prettychain("""
    Tell me about Apple. 
    Limit your answer to a single sentence.
    If you are unsure about the answer, say you don't know.
""")


# In[44]:


prettychain("""
    Whats NetApp

""")

