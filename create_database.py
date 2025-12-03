import os
import pymongo
from dotenv import load_dotenv
from mongo_docstore import MongoDocStore
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- NEW IMPORTS ---
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ParentDocumentRetriever

load_dotenv()

# --- Configuration ---
DATA_PATH = "data/"
INDEX_NAME = "jharkhand-policy-rag" 

def load_text_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, autodetect_encoding=True),
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()

def process_and_upload_documents(documents, embedding_model):
    """
    Initializes ParentDocumentRetriever with Pinecone (Vectors) and MongoDB (Docs).
    """
    # 1. MongoDB Connection
    mongo_uri = os.getenv("MONGO_DB_URI")
    client = pymongo.MongoClient(mongo_uri)
    collection = client["rag_database"]["parent_documents"]
    store = MongoDocStore(collection) 

    # 2. Text Splitters (RESTORED TO YOUR ORIGINAL VALUES)
    # Parent: 8000 chars (Large context for the LLM)
    # Child: 1000 chars (The vector searchable chunk)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 3. Initialize Pinecone Vector Store
    print(f"Connecting to Pinecone Index: {INDEX_NAME}...")
    # Ideally, rely on env variable PINECONE_API_KEY being set.
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_model,
        pinecone_api_key=os.getenv("PINECONE_API_KEY") 
    )

    # 4. Initialize Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 5. Add Documents
    print("Chunking and uploading documents to Pinecone & MongoDB...")
    retriever.add_documents(documents)
    print("✅ Upload complete!")

if __name__ == "__main__":
    # Load Embeddings
    cohere_api_key = os.getenv("COHERE_API_KEY")
    embedding_model = CohereEmbeddings(
        model="embed-multilingual-v3.0", 
        cohere_api_key=cohere_api_key
    )

    print("Loading text documents...")
    docs = load_text_files(DATA_PATH)
    
    # Run the upload
    process_and_upload_documents(docs, embedding_model)