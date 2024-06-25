## Corrective RAG

### Project Introduction

The project involves building a self-reflective Retrieval-Augmented Generation (RAG) application using open-source,
Language Models (LLMs). It focuses on retrieving and refining information based on relevance grading and, if necessary,
supplementing with web searches. This method ensures more accurate and reliable responses by leveraging local models and
a logical workflow.

**Key Workflow:**
<div align="center">
<img height="300" src="https://i.imghippo.com/files/87RDl1718984600.jpg" alt="" border="0">
</div>

1. **User starts with a question**: The user inputs a query.
2. **Perform retrieval based on the question**: The system retrieves documents based on the initial query.
3. **Evaluate retrieved documents**:
    - **If documents are good**: Use the retrieved documents to generate a response.
    - **If documents are not good**: Transform the query, perform a web search, and then generate a response based on
      the new set of documents.

### Prerequisites

- Python 3.11
- pip (Python package installer)
- Git (optional)

### Step 1: Initial Setup

#### 1. Initialize the Environment

First, let's set up the environment and install necessary dependencies.

1. **Create a `.env` file:**

2. This file will store your API keys and other configuration settings. Ensure it is included in your `.gitignore` file
   to prevent it from being committed to your repository.

   Example `.env` file:
   ```plaintext
   LANGCHAIN_API_KEY="your_langchain_api_key"
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_PROJECT="CorrectiveRAG"
   
   OPENAI_API_KEY="your_openai_api_key"
   ```

2. **Install required packages:**
   ```bash
   pip install langchain langchain_community openai streamlit python-dotenv beautifulsoup4 langchain-nomic
   ```
   ```bash
   pip install -U langchain-cli
   ```
    ```bash
   pip install langchain-nomic tiktoken chromadb
   ```
   ```bash
   pip install langchainhub
   ```
   ```bash
   pip install langgraph
   ```
   ```bash
   pip install tavily-python
   ```
   ```bash
   pip install langchain-mistralai
   ```
3. **Login to nomic by your API key:**

   ```bash
   nomic login your_nomic_api_key
   ```

#### Key Concepts

##### 1. Langgraph

- **Definition**: `langgraph` is a package designed to facilitate the integration of graph-based data structures with
  LangChain applications. It provides tools for building, managing, and querying graphs that represent relationships and
  dependencies between various data points.
- **Usage**: By using the `langgraph` package, developers can leverage graph-based representations to enhance the
  capabilities of their LangChain applications. This integration allows for advanced data modeling, relationship
  analysis, and complex querying, which can improve the accuracy and relevance of generated responses.

##### 2. Tavily

- **Definition**: `Tavily` is a data retrieval and analysis platform that provides tools and services to fetch, process,
  and analyze data from various sources. It offers APIs for developers to integrate its functionalities into their
  applications.
- **Usage**: In LangChain applications, the `tavily-python` package is used to integrate Tavily's data retrieval and
  analysis capabilities. This enables the application to fetch and process data from Tavily's platform, enhancing the
  information available for generating accurate and contextually relevant responses.

##### 3. Mistralai

- **Definition**: Mistralai is an advanced language model designed to provide high-quality natural language processing
  capabilities. It offers tools and models for various language tasks, including text generation, understanding, and
  transformation.
- **Usage**: Mistralai is used in applications that require sophisticated language processing abilities. By integrating
  Mistralai, developers can enhance their applications with advanced text generation, contextual understanding, and
  precise language handling, making the applications more effective in processing and responding to natural language
  inputs.

### Step 2: Setup LangServe and LangSmith

#### 1. LangServe Setup

Set up LangServe to manage our application deployment.
Use the LangServe CLI to create a new application called `pinecone-serverless`.

```bash
langchain app new corrective-rag
```   

#### 2. LangSmith Setup

Make sure u have created a LangSmith project for this lab.

**Project Name:** CorrectiveRAG

### Step 3: Setup Tavily

#### 1: Create a Account

- **Access Tavily:**

  Navigate to [Tavily](https://tavily.com/).

#### 2: Get Your Own API Key

1. **Navigate to API Keys:**


2. **Update your `.env` file:**

   Copy the generated API key and store it securely.
   Add it to your `.env` file:

    ```plaintext
    TAVILY_API_KEY="your_tavily_key"
    ```

### Step 4: Add Web Content Loading, Splitting, and Indexing with Chroma

In this step, we will add the functionality to load web content, split it into manageable chunks, and index the content
using Chroma and NomicEmbeddings. This process ensures that the content is properly prepared for efficient retrieval and
analysis.

#### 1. Create `chain.py` to Integrate Web Content Loading, Splitting, and Indexing

Here, we will set up the necessary components to load web content, split it into manageable chunks, and store the
embeddings in a vector store for retrieval.

**File**: `corrective-rag/app/chain.py`

**Code for `chain.py`:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

# Load web content
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = WebBaseLoader(url)
docs = loader.load()

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 500, chunk_overlap = 100
)
all_splits = text_splitter.split_documents(docs)

# Index the split documents
vectorstore = Chroma.from_documents(
    documents = all_splits,
    collection_name = "rag-chroma",
    embedding = NomicEmbeddings(model = "nomic-embed-text-v1"),
)
retriever = vectorstore.as_retriever()

print(retriever.get_relevant_documents("Agent memory"))
```

In this `chain.py` file:

- **WebBaseLoader** is used to retrieve content from a specified URL.
- **RecursiveCharacterTextSplitter** is utilized to split the retrieved documents into smaller chunks for efficient
  processing.
- **Chroma** is employed as the vector store to store and manage the document embeddings.
- **NomicEmbeddings** provides the embeddings for the documents, which are stored in the vector store.
- Finally, a retriever is created to fetch relevant documents based on queries.

#### 2. Test the Chain

Run the `chain.py` file and inspect the results:

<img src="https://i.imghippo.com/files/La2w71718983547.jpg" alt="" border="0">

The output contains a list of documents that are relevant to the query "Agent memory." Each document entry provides a
brief snippet of the content, along with the source and additional metadata.

#### The Whole Workflow

The whole workflow involves:

1. Retrieving and splitting documents.
2. Embedding and indexing them using Nomic embeddings.
3. Running an LLM to process them and provide answers.

#### Key Concepts

##### 1. WebBaseLoader

- **Definition**: `WebBaseLoader` is a component used to retrieve web content from specified URLs. It fetches the HTML
  content and processes it into a format suitable for further processing and analysis.
- **Usage**: In LangChain applications, `WebBaseLoader` is used to fetch content from the web that can be processed,
  analyzed, and stored for retrieval.

##### 2. RecursiveCharacterTextSplitter

- **Definition**: `RecursiveCharacterTextSplitter` is a utility for splitting text documents into smaller chunks based
  on character count. It ensures that each chunk is within a specified size limit, allowing for efficient processing and
  analysis.
- **Usage**: This utility is used to divide large documents into manageable pieces, making it easier to process and
  analyze large volumes of text in LangChain applications.

##### 3. Chroma

- **Definition**: `Chroma` is a vector store designed for storing and querying vector embeddings. It is optimized for
  performance and scalability, making it suitable for applications involving large-scale embedding-based retrieval and
  analysis.
- **Usage**: In LangChain applications, `Chroma` is used to store and manage vector embeddings generated by language
  models. This allows for efficient retrieval and analysis of embeddings, enabling advanced search and retrieval
  capabilities based on vector similarity.

##### 4. NomicEmbeddings

- **Definition**: `NomicEmbeddings` provides embeddings for text documents, which can be used for various tasks such as
  similarity search, clustering, and classification. These embeddings represent the semantic meaning of the text,
  allowing for advanced analysis and retrieval.
- **Usage**: By using `NomicEmbeddings`, LangChain applications can generate embeddings for text documents and store
  them in a vector store like Chroma for efficient retrieval and analysis.

### Step 5: Setup Mistral

#### 1: Create a Account

- **Access Mistral:**

  Navigate to [Mistral](https://mistral.ai/).

#### 2: Get Your Own API Key

1. **Navigate to API Keys:**


2. **Update your `.env` file:**

   Copy the generated API key and store it securely.
   Add it to your `.env` file:

    ```plaintext
    MISTRAL_API_KEY="your_mistral_api_key"
    ```

### Step 6: Add Relevance Scoring to Retrieve and Grade Documents

In this step, we will add a relevance scoring mechanism to assess the retrieved documents' relevance to the user query.
This will ensure that the responses are accurate and contextually appropriate.

#### 1. Pull the Mistral Instruct Model

First, ensure you have the necessary model for relevance grading:

You need to first install a local Ollama from [here](https://github.com/ollama/ollama?tab=readme-ov-file), if your dont
have one

```bash
ollama pull mistral:instruct
```

#### 2. Integrate Relevance Scoring

We will update the `chain.py` file to include relevance scoring using the `ChatMistralAI` model or `ChatOllama`. This
involves setting
up a prompt and a chain to assess the relevance of the retrieved documents.

**File**: `corrective-rag/app/chain.py`

**Updated Code for `chain.py`**:

```python

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# LLM
# llm = ChatOllama(model = "mistral:instruct")
llm = ChatMistralAI(model = "mistral-medium", temperature = 0)

prompt = PromptTemplate(
    template = """You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables = ["question", "context"],
)

chain = prompt | llm | JsonOutputParser()
question = "Explain how the different types of agent memory work?"
docs = retriever.get_relevant_documents(question)
score = chain.invoke({"question": question, "context": docs[0].page_content})
print(score)
```

#### 3. Test the Relevance Scoring

Run the `chain.py` file and inspect the results to ensure that the relevance scoring is functioning correctly. The
output will indicate whether the retrieved document is relevant to the user question based on the binary score.

<img src="https://i.imghippo.com/files/t2zTL1719034163.jpg" alt="" border="0">

This output indicates that the chain is working correctly, retrieving and scoring the relevance of the document.

### Step 7: Implement Advanced Retrieval and Generation Pipeline

In this step, we will integrate advanced features to handle SSL errors, manage conditional states for query
transformations and web searches, and implement workflow nodes for comprehensive document processing and query
answering.

#### 1. Update `chain.py` to Implement Advanced Retrieval and Generation Pipeline

We will update the `chain.py` file to include components for handling SSL errors, managing conditional state
transitions, and defining nodes for retrieval, document grading, generation, query transformation, and web search.

**File**: `corrective-rag/app/chain.py`

**Updated Code for `chain.py`**:

```python
from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage
import json
import operator
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from langgraph.graph import END, StateGraph


# Define GraphState for state management
class GraphState(TypedDict):
    keys: Dict[str, any]


def retrieve(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model = "mistral:instruct")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"keys": {"documents": documents, "question": question, "generation": generation}}


def grade_documents(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    llm = ChatOllama(model = "mistral:instruct")

    prompt = PromptTemplate(
        template = """
        You are a grader assessing relevance of a retrieved document to a user question. 
        Here is the retrieved document: 
        \n\n{context}\n\n
        Here is the user question: 
        \n\n{question}\n\n
        If the document contains keywords related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        """,
        input_variables = ["question", "context"],
    )

    chain = prompt | llm | JsonOutputParser()
    filtered_docs = []
    search = "No"

    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}


def transform_query(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = PromptTemplate(
        template = """
        You are generating questions that is well optimized for retrieval. 
        Look at the input and try to reason about the underlying semantic intent/meaning. 
        Here is the initial question:
        {question}
        Provide an improved question without any preamble, only respond with the updated question.
        """,
        input_variables = ["question"],
    )

    llm = ChatOllama(model = "mistral:instruct")
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"question": better_question, "documents": documents}}


def web_search(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content = web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "question": question}}


def decide_to_generate(state):
    state_dict = state["keys"]
    search = state_dict["run_web_search"]
    if search == "Yes":
        return "transform_query"
    else:
        return "generate"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

inputs = {"keys": {"question": "Explain how the different types of agent memory work?"}}

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node {key}:")
        pprint("\n------\n")

pprint(value["keys"]["generation"])
```

### 2. Test the chain

To test the updated chain, execute the `chain.py` file and inspect the results.

<img src="https://i.imghippo.com/files/8dNFR1719344691.jpg" alt="" border="0">

### 3. Test the chain with question irrelevant with the provide docs

Now change the question with something not related with the given docs

```python
# Run
inputs = {
    "keys": {
        "question": "Who is Sheev Palpatine",
    }
}
```

Execute the `chain.py` file and inspect the results.

<img src="https://i.imghippo.com/files/uEreF1719345012.jpg" alt="" border="0">

In this test, the system should initially retrieve documents based on the irrelevant question. Since the retrieved
documents will not be relevant to the question, the system will trigger the query transformation process, followed by a
web search to find more relevant information. Finally, the system will generate a response based on the updated and
relevant documents from the web search.

This process ensures that the system can handle cases where the initial document retrieval does not provide relevant
information, thereby improving the accuracy and relevance of the generated responses through query transformation and
supplemental web searches.
 