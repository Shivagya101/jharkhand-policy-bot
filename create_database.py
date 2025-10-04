import os
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
DOCSTORE_PATH = "vector_store/parent_store"

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
    Initializes the ParentDocumentRetriever with a parent-child splitting strategy.
    """
    # Recommendation: Consider larger parent chunks for more context
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=70)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    # --- MODIFICATION START ---
    # 1. Create the low-level file store
    fs = LocalFileStore(DOCSTORE_PATH)
    # 2. Wrap it with create_kv_docstore to handle Document object serialization
    store = create_kv_docstore(fs)
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
    """
    Initializes the HuggingFace embedding model (Hindi + English support).
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="l3cube-pune/indic-sentence-similarity-sbert"
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
    print(f"Parent docstore saved to {DOCSTORE_PATH}")
    print("✅ Vector database created and saved successfully.")