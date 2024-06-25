from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

import requests
import certifi

load_dotenv()
wait_for_all_tracers()

# Load web content
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = WebBaseLoader(url)
try:
    docs = loader.load()
except requests.exceptions.SSLError as e:
    print(f"SSLError: {e}")
    # Optionally, retry with SSL verification using certifi
    docs = loader.load(verify=certifi.where())

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)

# Index the split documents
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1")
)
retriever = vectorstore.as_retriever()

print(retriever.get_relevant_documents("Agent memory"))



from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_mistralai.chat_models import ChatMistralAI

retriever = vectorstore.as_retriever()

### Retrieval Grader
# Define the relevance scoring prompt
prompt = PromptTemplate(
    template = """
    You are a grader assessing relevance of a retrieved document to a user question.  
    Here is the retrieved document: 
    {context}
    Here is the user question: 
    {question} 
    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """,
    input_variables = ["question", "document"],
)

# Define the chain for relevance scoring
chain = prompt |  ChatOllama(model = "mistral:instruct") | JsonOutputParser()

# Test the relevance scoring
question = "agent memory"
docs = retriever.get_relevant_documents(question)

score = chain.invoke({"question": question, "context": docs[0].page_content})