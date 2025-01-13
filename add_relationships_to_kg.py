#!/usr/bin/env python
# coding: utf-8

# ### Import packages and set up Neo4j

# In[ ]:


from dotenv import load_dotenv
import os

# Common data processing
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Warning control
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

# Global constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'


# In[ ]:


kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


# ### Create a Form 10-K node
# - Create a node to represent the entire Form 10-K
# - Populate with metadata taken from a single chunk of the form

# In[ ]:


cypher = """
  MATCH (anyChunk:Chunk) 
  WITH anyChunk LIMIT 1
  RETURN anyChunk { .names, .source, .formId, .cik, .cusip6 } as formInfo
"""
form_info_list = kg.query(cypher)

form_info_list


# In[ ]:


form_info = form_info_list[0]['formInfo']


# In[ ]:


form_info


# In[ ]:


cypher = """
    MERGE (f:Form {formId: $formInfoParam.formId })
      ON CREATE 
        SET f.names = $formInfoParam.names
        SET f.source = $formInfoParam.source
        SET f.cik = $formInfoParam.cik
        SET f.cusip6 = $formInfoParam.cusip6
"""

kg.query(cypher, params={'formInfoParam': form_info})


# In[ ]:


kg.query("MATCH (f:Form) RETURN count(f) as formCount")


# ### Create a linked list of Chunk nodes for each section
# - Start by identifying chunks from the same section

# In[ ]:


cypher = """
  MATCH (from_same_form:Chunk)
    WHERE from_same_form.formId = $formIdParam
  RETURN from_same_form {.formId, .f10kItem, .chunkId, .chunkSeqId } as chunkInfo
    LIMIT 10
"""

kg.query(cypher, params={'formIdParam': form_info['formId']})


# - Order chunks by their sequence ID

# In[ ]:


cypher = """
  MATCH (from_same_form:Chunk)
    WHERE from_same_form.formId = $formIdParam
  RETURN from_same_form {.formId, .f10kItem, .chunkId, .chunkSeqId } as chunkInfo 
    ORDER BY from_same_form.chunkSeqId ASC
    LIMIT 10
"""

kg.query(cypher, params={'formIdParam': form_info['formId']})


# - Limit chunks to just the "Item 1" section, the organize in ascending order

# In[ ]:


cypher = """
  MATCH (from_same_section:Chunk)
  WHERE from_same_section.formId = $formIdParam
    AND from_same_section.f10kItem = $f10kItemParam // NEW!!!
  RETURN from_same_section { .formId, .f10kItem, .chunkId, .chunkSeqId } 
    ORDER BY from_same_section.chunkSeqId ASC
    LIMIT 10
"""

kg.query(cypher, params={'formIdParam': form_info['formId'], 
                         'f10kItemParam': 'item1'})


# - Collect ordered chunks into a list

# In[ ]:


cypher = """
  MATCH (from_same_section:Chunk)
  WHERE from_same_section.formId = $formIdParam
    AND from_same_section.f10kItem = $f10kItemParam
  WITH from_same_section { .formId, .f10kItem, .chunkId, .chunkSeqId } 
    ORDER BY from_same_section.chunkSeqId ASC
    LIMIT 10
  RETURN collect(from_same_section) // NEW!!!
"""

kg.query(cypher, params={'formIdParam': form_info['formId'], 
                         'f10kItemParam': 'item1'})


# ### Add a NEXT relationship between subsequent chunks
# - Use the `apoc.nodes.link` function from Neo4j to link ordered list of `Chunk` nodes with a `NEXT` relationship
# - Do this for just the "Item 1" section to start

# In[ ]:


cypher = """
  MATCH (from_same_section:Chunk)
  WHERE from_same_section.formId = $formIdParam
    AND from_same_section.f10kItem = $f10kItemParam
  WITH from_same_section
    ORDER BY from_same_section.chunkSeqId ASC
  WITH collect(from_same_section) as section_chunk_list
    CALL apoc.nodes.link(
        section_chunk_list, 
        "NEXT", 
        {avoidDuplicates: true}
    )  // NEW!!!
  RETURN size(section_chunk_list)
"""

kg.query(cypher, params={'formIdParam': form_info['formId'], 
                         'f10kItemParam': 'item1'})


# In[ ]:


kg.refresh_schema()
print(kg.schema)


# - Loop through and create relationships for all sections of the form 10-K

# In[ ]:


cypher = """
  MATCH (from_same_section:Chunk)
  WHERE from_same_section.formId = $formIdParam
    AND from_same_section.f10kItem = $f10kItemParam
  WITH from_same_section
    ORDER BY from_same_section.chunkSeqId ASC
  WITH collect(from_same_section) as section_chunk_list
    CALL apoc.nodes.link(
        section_chunk_list, 
        "NEXT", 
        {avoidDuplicates: true}
    )
  RETURN size(section_chunk_list)
"""
for form10kItemName in ['item1', 'item1a', 'item7', 'item7a']:
  kg.query(cypher, params={'formIdParam':form_info['formId'], 
                           'f10kItemParam': form10kItemName})


# ### Connect chunks to their parent form with a PART_OF relationship

# In[ ]:


cypher = """
  MATCH (c:Chunk), (f:Form)
    WHERE c.formId = f.formId
  MERGE (c)-[newRelationship:PART_OF]->(f)
  RETURN count(newRelationship)
"""

kg.query(cypher)


# ### Create a SECTION relationship on first chunk of each section

# In[ ]:


cypher = """
  MATCH (first:Chunk), (f:Form)
  WHERE first.formId = f.formId
    AND first.chunkSeqId = 0
  WITH first, f
    MERGE (f)-[r:SECTION {f10kItem: first.f10kItem}]->(first)
  RETURN count(r)
"""

kg.query(cypher)


# ### Example cypher queries
# - Return the first chunk of the Item 1 section

# In[ ]:


cypher = """
  MATCH (f:Form)-[r:SECTION]->(first:Chunk)
    WHERE f.formId = $formIdParam
        AND r.f10kItem = $f10kItemParam
  RETURN first.chunkId as chunkId, first.text as text
"""

first_chunk_info = kg.query(cypher, params={
    'formIdParam': form_info['formId'], 
    'f10kItemParam': 'item1'
})[0]

first_chunk_info


# - Get the second chunk of the Item 1 section

# In[ ]:


cypher = """
  MATCH (first:Chunk)-[:NEXT]->(nextChunk:Chunk)
    WHERE first.chunkId = $chunkIdParam
  RETURN nextChunk.chunkId as chunkId, nextChunk.text as text
"""

next_chunk_info = kg.query(cypher, params={
    'chunkIdParam': first_chunk_info['chunkId']
})[0]

next_chunk_info


# In[ ]:


print(first_chunk_info['chunkId'], next_chunk_info['chunkId'])


# - Return a window of three chunks

# In[ ]:


cypher = """
    MATCH (c1:Chunk)-[:NEXT]->(c2:Chunk)-[:NEXT]->(c3:Chunk) 
        WHERE c2.chunkId = $chunkIdParam
    RETURN c1.chunkId, c2.chunkId, c3.chunkId
    """

kg.query(cypher,
         params={'chunkIdParam': next_chunk_info['chunkId']})


# ### Information is stored in the structure of a graph
# - Matched patterns of nodes and relationships in a graph are called **paths**
# - The length of a path is equal to the number of relationships in the path
# - Paths can be captured as variables and used elsewhere in queries

# In[ ]:


cypher = """
    MATCH window = (c1:Chunk)-[:NEXT]->(c2:Chunk)-[:NEXT]->(c3:Chunk) 
        WHERE c1.chunkId = $chunkIdParam
    RETURN length(window) as windowPathLength
    """

kg.query(cypher,
         params={'chunkIdParam': next_chunk_info['chunkId']})


# ### Finding variable length windows
# - A pattern match will fail if the relationship doesn't exist in the graph
# - For example, the first chunk in a section has no preceding chunk, so the next query won't return anything

# In[ ]:


cypher = """
    MATCH window=(c1:Chunk)-[:NEXT]->(c2:Chunk)-[:NEXT]->(c3:Chunk) 
        WHERE c2.chunkId = $chunkIdParam
    RETURN nodes(window) as chunkList
    """
# pull the chunk ID from the first 
kg.query(cypher,
         params={'chunkIdParam': first_chunk_info['chunkId']})


# - Modify `NEXT` relationship to have variable length

# In[ ]:


cypher = """
  MATCH window=
      (:Chunk)-[:NEXT*0..1]->(c:Chunk)-[:NEXT*0..1]->(:Chunk) 
    WHERE c.chunkId = $chunkIdParam
  RETURN length(window)
  """

kg.query(cypher,
         params={'chunkIdParam': first_chunk_info['chunkId']})


# - Retrieve only the longest path

# In[ ]:


cypher = """
  MATCH window=
      (:Chunk)-[:NEXT*0..1]->(c:Chunk)-[:NEXT*0..1]->(:Chunk)
    WHERE c.chunkId = $chunkIdParam
  WITH window as longestChunkWindow 
      ORDER BY length(window) DESC LIMIT 1
  RETURN length(longestChunkWindow)
  """

kg.query(cypher,
         params={'chunkIdParam': first_chunk_info['chunkId']})


# ### Customize the results of the similarity search using Cypher
# - Extend the vector store definition to accept a Cypher query
# - The Cypher query takes the results of the vector similarity search and then modifies them in some way
# - Start with a simple query that just returns some extra text along with the search results

# In[ ]:


retrieval_query_extra_text = """
WITH node, score, "Andreas knows Cypher. " as extraText
RETURN extraText + "\n" + node.text as text,
    score,
    node {.source} AS metadata
"""


# - Set up the vector store to use the query, then instantiate a retriever and Question-Answer chain in LangChain
# 

# In[ ]:


vector_store_extra_text = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name=VECTOR_INDEX_NAME,
    text_node_property=VECTOR_SOURCE_PROPERTY,
    retrieval_query=retrieval_query_extra_text, # NEW !!!
)

# Create a retriever from the vector store
retriever_extra_text = vector_store_extra_text.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
chain_extra_text = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever_extra_text
)


# - Ask a question!

# In[ ]:


chain_extra_text(
    {"question": "What topics does Andreas know about?"},
    return_only_outputs=True)


# - Note, the LLM hallucinates here, using the information in the retrieved text as well as the extra text.
# - Modify the prompt to try and get a more accurate answer

# In[ ]:


chain_extra_text(
    {"question": "What single topic does Andreas know about?"},
    return_only_outputs=True)


# ### Try for yourself!
# - Modify the query below to add your own additional text
# - Try engineering the prompt to refine your results
# - Note, you'll need to reset the vector store, retriever, and chain each time you change the Cypher query.

# In[ ]:


# modify the retrieval extra text here then run the entire cell
retrieval_query_extra_text = """
WITH node, score, "Andreas knows Cypher. " as extraText
RETURN extraText + "\n" + node.text as text,
    score,
    node {.source} AS metadata
"""

vector_store_extra_text = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name=VECTOR_INDEX_NAME,
    text_node_property=VECTOR_SOURCE_PROPERTY,
    retrieval_query=retrieval_query_extra_text, # NEW !!!
)

# Create a retriever from the vector store
retriever_extra_text = vector_store_extra_text.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
chain_extra_text = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever_extra_text
)


# ### Expand context around a chunk using a window
# - First, create a regular vector store that retrieves a single node

# In[ ]:


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
# Create a retriever from the vector store
windowless_retriever = neo4j_vector_store.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
windowless_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=windowless_retriever
)


# - Next, define a window retrieval query to get consecutive chunks

# In[ ]:


retrieval_query_window = """
MATCH window=
    (:Chunk)-[:NEXT*0..1]->(node)-[:NEXT*0..1]->(:Chunk)
WITH node, score, window as longestWindow 
  ORDER BY length(window) DESC LIMIT 1
WITH nodes(longestWindow) as chunkList, node, score
  UNWIND chunkList as chunkRows
WITH collect(chunkRows.text) as textList, node, score
RETURN apoc.text.join(textList, " \n ") as text,
    score,
    node {.source} AS metadata
"""


# - Set up a QA chain that will use the window retrieval query

# In[ ]:


vector_store_window = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name=VECTOR_INDEX_NAME,
    text_node_property=VECTOR_SOURCE_PROPERTY,
    retrieval_query=retrieval_query_window, # NEW!!!
)

# Create a retriever from the vector store
retriever_window = vector_store_window.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
chain_window = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever_window
)


# ### Compare the two chains

# In[ ]:


question = "In a single sentence, tell me about Netapp's business."


# In[ ]:


answer = windowless_chain(
    {"question": question},
    return_only_outputs=True,
)
print(textwrap.fill(answer["answer"]))


# In[ ]:


answer = chain_window(
    {"question": question},
    return_only_outputs=True,
)
print(textwrap.fill(answer["answer"]))

