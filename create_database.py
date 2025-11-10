import os
import pymongo
from dotenv import load_dotenv
from mongo_docstore import MongoDocStore
from langchain_cohere import CohereEmbeddings

load_dotenv()


from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss # Use the direct faiss import



# --- NEW IMPORT: The key to solving the error ---
from langchain.storage import create_kv_docstore

# --- Paths ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/db_faiss"
# DOCSTORE_PATH = "vector_store/parent_store"

def load_text_files(data_path):
    """
    Loads all text documents from a specified directory, with UTF-8 encoding.
    """
    # Using TextLoader directly with specified encoding
    loader = DirectoryLoader(
        data_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding='utf-8'),
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    return documents


def get_retriever(vectorstore, documents):
    """
    Initializes the ParentDocumentRetriever with MongoDB as the docstore,
    using a connection URI from the environment file.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)

    # --- MODIFICATION START ---
    # 1. Get the MongoDB URI from environment variables
    mongo_uri = os.getenv("MONGO_DB_URI")
    if not mongo_uri:
        raise ValueError("MONGO_DB_URI not found in environment variables. Check your .env file.")

    # 2. Connect to your MongoDB database and collection
    client = pymongo.MongoClient(mongo_uri)
    collection = client["rag_database"]["parent_documents"]
    
    # 3. Use your custom MongoDocStore
    store = MongoDocStore(collection)
    # --- MODIFICATION END ---

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Adding documents to the retriever...")
    retriever.add_documents(documents, ids=None) 
    print("Documents added successfully.")
    return retriever

def get_embedding_model():
    """Initializes the Cohere embedding model."""
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables.")
    
    # Use a standard, powerful Cohere embedding model
    embedding_model = CohereEmbeddings(
        model="embed-multilingual-v3.0", 
        cohere_api_key=cohere_api_key
    )
    return embedding_model


if __name__ == "__main__":
    print("Loading text documents...")
    documents = load_text_files(DATA_PATH)
    print(f"Loaded {len(documents)} text documents.")

    embedding_model = get_embedding_model()

    # --- Initialize FAISS vector store ---
    dummy_embedding = embedding_model.embed_query("test")
    embedding_size = len(dummy_embedding)
    index = faiss.IndexFlatL2(embedding_size)
    faiss_docstore = InMemoryDocstore()
    index_to_docstore_id = {}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=faiss_docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # --- Create and store embeddings ---
    retriever = get_retriever(vectorstore, documents)

    print(f"Saving FAISS index to {DB_FAISS_PATH}...")
    retriever.vectorstore.save_local(DB_FAISS_PATH)
    print("✅ Vector database created and saved successfully.")