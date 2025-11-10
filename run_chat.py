import os
import pymongo 

# If you keep credentials in a .env file, try to load them into the environment at runtime.
try:
    # python-dotenv is listed in requirements.txt; this import is optional in environments where
    # environment variables are set externally (CI, system env, etc.).
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # No .env file found in the project hierarchy.
        pass
except Exception:
    # If python-dotenv isn't installed or fails to load, we continue — the script will still
    # read variables from the actual environment.
    print("python-dotenv not available; relying on environment variables")
    pass

from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.storage import create_kv_docstore
from typing import List, TypedDict, Annotated, Sequence
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
import operator
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_lc_store
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mongo_docstore import MongoDocStore


# Global counter for how many times the LLM is called.
llm_call_count = 0


def invoke_chain(chain, payload: dict):
    """Invoke a chain/llm and increment the global llm_call_count.

    Returns the raw response from chain.invoke(payload).
    """
    global llm_call_count
    resp = chain.invoke(payload)
    llm_call_count += 1
    return resp

# --- Configuration (No changes here) ---
# HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
# REPO_ID = "Qwen/Qwen2-7B-Instruct"

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
DB_FAISS_PATH = "vector_store/db_faiss"
# EMBED_MODEL = "l3cube-pune/indic-sentence-similarity-sbert"
# --- IMPORTANT: Add the path to your PDF data for the retriever setup ---
DATA_PATH = "data/" 

# --- LLM and Retriever Setup (Reused and adapted from previous scripts) ---

def load_chat_model() -> ChatCohere: # CHANGED: Type hint to ChatCohere
    """
    Loads the Cohere chat model.
    """
    if not COHERE_API_KEY:
        raise RuntimeError("Set COHERE_API_KEY in your environment.")
    
    # CHANGED: Instantiate ChatCohere instead of HuggingFaceEndpoint
    # Using 'command-s' as a powerful and balanced model choice.
    model = ChatCohere(
        model="command-a-03-2025", 
        cohere_api_key=COHERE_API_KEY,
        temperature=0.1 # Lower temperature for more deterministic grading/rewriting
    )
    return model



def get_retriever():
    """
    Loads the FAISS vector store and connects to MongoDB to reconstruct
    the ParentDocumentRetriever.
    """
    # Step 1: Load the SAME Cohere embedding model used for creation
    print("Loading Cohere embedding model...")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables.")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)

    # Step 2: Load FAISS index (child embeddings)
    print("Loading FAISS vector store...")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Step 3: Connect to MongoDB for the parent docstore
    print("Connecting to MongoDB for parent documents...")
    mongo_uri = os.getenv("MONGO_DB_URI")
    if not mongo_uri:
        raise ValueError("MONGO_DB_URI not found in environment variables.")
    
    client = pymongo.MongoClient(mongo_uri)
    collection = client["rag_database"]["parent_documents"]
    store = MongoDocStore(collection)

    # Step 4: Use the same splitters as during creation
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)
    
    # Step 5: Rebuild the ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    print("✅ Retriever loaded successfully with MongoDB docstore.")
    return retriever


# 1. Define the State
# The state is the "memory" of our agent. It's a dictionary that gets passed between nodes.
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List
    iterations: int
    context_too_large: bool

# 2. Define the Nodes
# Each node is a function that performs an action and updates the state.

def retrieve(state):
    """
    Retrieve documents from the vector store.
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]
    # Support multiple retriever APIs (langchain has several: get_relevant_documents, retrieve, etc.)
    try:
        documents = retriever.get_relevant_documents(question)
    except AttributeError:
        try:
            documents = retriever.retrieve(question)
        except AttributeError:
            # Fallback: if the retriever implements an invoke-style API used by the graph
            documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    This is our "critic" node.
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    iterations = state.get("iterations", 0) + 1
    
    # Simple grading prompt that returns 'yes' or 'no' in plain text
    grade_prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "If the document contains keywords or clear answers related to the user question, reply with 'yes'.\n"
            "Otherwise, reply with 'no'.\n\n"
            "Retrieved document:\n{document}\n\nUser question:\n{question}\n\nAnswer:" 
        ),
        input_variables=["document", "question"],
    )
    
    # Use the chat LLM directly (no structured output). We'll parse 'yes'/'no' from the model text.
    chain = grade_prompt | llm
    
    filtered_docs = []
    for d in documents:
        resp = invoke_chain(chain, {"question": question, "document": d.page_content})
        print(f"Grading response: {getattr(resp, 'content', str(resp))}")
        # The LLM returns text; normalize and check for 'yes' or 'no'
        text = getattr(resp, "content", str(resp)).strip().lower()
        is_yes = False
        if text.startswith("yes") or " yes" in f" {text} ":
            is_yes = True

        if is_yes:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    return {"documents": filtered_docs, "iterations": iterations}

def generate(state):
    """
    Generate an answer using the retrieved documents.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    print(documents)
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT, input_variables=["context", "question"]
    )
    
    rag_chain = prompt | llm
    generation = invoke_chain(rag_chain, {"context": documents, "question": question})
    # Ensure we return the text content when available
    return {"generation": getattr(generation, "content", str(generation))}

def rewrite_query(state):
    """
    Transform the query to produce a better question.
    """
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    
    # Prompt
    system = """You are a query re-writer. Given a user question, your task is to rephrase it to be more
    aligned with the language and terminology found in legal and policy documents.
    Do not answer the question, only rewrite it."""
    
    rewrite_prompt = PromptTemplate(
        template="Original question: {question}",
        input_variables=["question"],
    )
    
    rewriter_chain = rewrite_prompt | llm
    rewritten_question = invoke_chain(rewriter_chain, {"question": question})
    print(f"Rewritten question: {getattr(rewritten_question, 'content', str(rewritten_question))}")
    return {"question": getattr(rewritten_question, "content", str(rewritten_question))}

def check_context_size(state):
    """
    Check if the combined context of retrieved documents is too large for the LLM.
    """
    print("---NODE: CHECK CONTEXT SIZE---")
    documents = state["documents"]
    # Simple token count estimation (adjust limit as needed for your model)
    # A more robust method would use the model's tokenizer.
    CONTEXT_LIMIT = 3000 
    total_tokens = sum(len(doc.page_content.split()) for doc in documents)
    
    if total_tokens > CONTEXT_LIMIT:
        print(f"---CONTEXT OVERFLOW: {total_tokens} tokens > {CONTEXT_LIMIT}---")
        return {"context_too_large": True}
    else:
        print(f"---CONTEXT OK: {total_tokens} tokens <= {CONTEXT_LIMIT}---")
        return {"context_too_large": False}

def summarize_map(state):
    """
    Summarize each document individually if the context is too large.
    This is the "Map" part of Map-Reduce.
    """
    print("---NODE: SUMMARIZE MAP---")
    question = state["question"]
    documents = state["documents"]

    summarizer_prompt = PromptTemplate(
        template="Summarize the key points in the following text that are relevant to the user's query: '{question}'\n\nText: {document}",
        input_variables=["question", "document"],
    )
    summarizer_chain = summarizer_prompt | llm

    # Summarize each document in parallel (or sequentially)
    summaries = []
    for doc in documents:
        # ✅ Call the LLM only ONCE per document
        result = invoke_chain(summarizer_chain, {"question": question, "document": doc.page_content})
        summary_text = getattr(result, "content", str(result))
        summaries.append(summary_text)
    
    # Create new Document objects from the summaries
    summary_docs = [
        Document(page_content=summary, metadata=(getattr(doc, "metadata", {}) or {}))
        for summary, doc in zip(summaries, documents)
    ]
    return {"documents": summary_docs}

# 3. Define the Edges (The control flow)

def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate the question.
    """
    print("---EDGE: DECIDE TO GENERATE---")
    documents = state["documents"]
    iterations = state.get("iterations", 0)
    
    if not documents or iterations >= 3: # If no relevant docs or max retries reached
        if iterations >= 3:
            print("---DECISION: MAX RETRIES REACHED, ENDING---")
        else:
            print("---DECISION: NO RELEVANT DOCUMENTS, REWRITING QUERY---")
        return "rewrite"
    else:
        print("---DECISION: RELEVANT DOCUMENTS FOUND, PROCEEDING TO GENERATE---")
        return "generate"

def decide_context_path(state):
    """
    Decides whether to use the full context or apply Map-Reduce.
    """
    print("---EDGE: DECIDE CONTEXT PATH---")
    if state["context_too_large"]:
        return "summarize"
    else:
        return "direct_generate"

# --- Build the Graph ---

# Initialize LLM and Retriever globally for the graph nodes
llm = load_chat_model()
retriever = get_retriever()
CUSTOM_PROMPT = """Use ONLY the information in the context to answer the user's question. If the answer is not in the context, say you don't know. Do not make anything up.

Context: {context}
Question: {question}
Answer:"""

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("check_context_size", check_context_size)
workflow.add_node("summarize_map", summarize_map)
workflow.add_node("generate", generate)

# Set entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite": "rewrite",
        "generate": "check_context_size",
    },
)
workflow.add_edge("rewrite", "retrieve") # This creates the self-correction loop
workflow.add_conditional_edges(
    "check_context_size",
    decide_context_path,
    {
        "summarize": "summarize_map",
        "direct_generate": "generate",
    },
)
workflow.add_edge("summarize_map", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# --- Run the Chatbot ---
if __name__ == "__main__":
    app = workflow.compile()
    query = input("Enter your query: ").strip()
    if not query:
        raise SystemExit("Empty query. Exiting.")

    # Invoke the graph once (do not stream and then invoke again - that causes two runs)
    inputs = {"question": query, "iterations": 0}
    final_state = app.invoke(inputs)

    # Print only the final answer and the LLM call count
    print("\n\n--- FINAL ANSWER ---")
    print(final_state.get("generation"))
    print(f"\nLLM calls made: {llm_call_count}")

