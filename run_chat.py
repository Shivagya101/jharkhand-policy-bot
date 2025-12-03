import os
import pymongo
import time
from langchain_pinecone import PineconeVectorStore

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
# Rate limiting for Cohere Trial key (10 calls/minute = 0.6 seconds min between calls)
RATE_LIMIT_DELAY = 0.7  # seconds between API calls


def invoke_chain(chain, payload: dict):
    """Invoke a chain/llm and increment the global llm_call_count.

    Returns the raw response from chain.invoke(payload).
    Adds delay to respect Cohere Trial rate limit.
    """
    global llm_call_count
    # Add delay to respect rate limit (Cohere Trial: 10 calls/minute)
    time.sleep(RATE_LIMIT_DELAY)
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
    Connects to Pinecone and MongoDB to reconstruct the ParentDocumentRetriever.
    """
    # 1. Load Embeddings
    print("Loading Cohere embedding model...")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)

    # 2. Connect to Pinecone
    print("Connecting to Pinecone...")
    # pinecone_api_key = os.getenv("PINECONE_API_KEY") # No longer needed here
    index_name = "jharkhand-policy-rag"
    
    # FIX: Remove the pinecone_api_key argument
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. Connect to MongoDB 
    print("Connecting to MongoDB for parent documents...")
    mongo_uri = os.getenv("MONGO_DB_URI")
    client = pymongo.MongoClient(mongo_uri)
    collection = client["rag_database"]["parent_documents"]
    store = MongoDocStore(collection)

    # 4. Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # 5. Rebuild Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    print("✅ Retriever connected (Pinecone + MongoDB).")
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
        documents = retriever.invoke(
            question, 
            config={"configurable": {"search_kwargs": {"k": 8}}}
        )
    except AttributeError:
        retriever.vectorstore.search_kwargs = {"k": 8}
        documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    This is our "critic" node.
    OPTIMIZED: Only grade top-3 documents to reduce API calls (Trial key has 10 calls/min limit)
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    iterations = state.get("iterations", 0) + 1
    
    # OPTIMIZATION: Only grade top 3 documents instead of all
    # FAISS already ranked by relevance, so top results are usually good
    top_k_to_grade = min(5, len(documents))
    docs_to_grade = documents[:top_k_to_grade]
    
    # If we retrieved very few docs, keep them all
    if len(documents) <= 2:
        print(f"---KEEPING ALL {len(documents)} DOCUMENTS (too few to filter)---")
        return {"documents": documents, "iterations": iterations}
    
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
    for i, d in enumerate(docs_to_grade):
        print(f"Grading doc {i+1}/{top_k_to_grade}...")
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
    
    # If all graded docs were rejected, keep the top 1 to ensure we have something
    if not filtered_docs and docs_to_grade:
        print("---ALL GRADED DOCS REJECTED, KEEPING TOP 1---")
        filtered_docs = [docs_to_grade[0]]
    
    return {"documents": filtered_docs, "iterations": iterations}

def generate(state):
    """
    Generate a detailed answer using the retrieved documents.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Format documents into readable context
    formatted_context = "\n---\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
    )
    
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT, input_variables=["context", "question"]
    )
    
    rag_chain = prompt | llm
    generation = invoke_chain(rag_chain, {"context": formatted_context, "question": question})
    # Ensure we return the text content when available
    return {"generation": getattr(generation, "content", str(generation))}

def rewrite_query(state):
    """
    Transform the query to produce a better question for document retrieval.
    """
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    
    # More effective rewriting prompt
    rewrite_prompt = PromptTemplate(
        template=(
            "You are an expert at reformulating user questions to better match government policy documents.\n\n"
            "Rewrite the following question to:\n"
            "1. Use official terminology (e.g., 'scheme', 'eligibility', 'benefits', 'implementation')\n"
            "2. Add relevant keywords that might appear in policy documents\n"
            "3. If it's about a specific policy, try to identify what type it is\n\n"
            "Original question: {question}\n\n"
            "Rewritten question:"
        ),
        input_variables=["question"],
    )
    
    rewriter_chain = rewrite_prompt | llm
    rewritten_question = invoke_chain(rewriter_chain, {"question": question})
    rewritten_text = getattr(rewritten_question, "content", str(rewritten_question)).strip()
    print(f"Rewritten question: {rewritten_text}")
    return {"question": rewritten_text}

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
CUSTOM_PROMPT = """You are an expert on Jharkhand Government policies and schemes. Use the provided context to answer the user's question comprehensively.

Context:
{context}

User Question: {question}

Instructions:
1. Answer based ONLY on the information provided in the context above
2. If multiple policies/schemes are mentioned, explain each one separately with details like:
   - Name and purpose
   - Key benefits
   - Eligibility criteria
   - Implementation process
3. Be detailed and thorough - provide specific information, not just general statements
4. If the answer is not in the context, clearly state "This information is not available in the provided documents"
5. Use structured formatting (bullet points, numbered lists) for clarity

Detailed Answer:"""

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

