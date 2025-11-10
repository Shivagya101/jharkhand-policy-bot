import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# These MUST match the settings used in your create_database.py script
DB_FAISS_PATH = "vector_store/db_faiss"
EMBED_MODEL = "l3cube-pune/indic-sentence-similarity-sbert"

def inspect_vector_store(db_path, embedding_model, limit=5):
    """
    Loads a FAISS vector store and prints the text and a snippet of the
    corresponding embedding for a specified number of items.

    Args:
        db_path (str): Path to the saved FAISS database.
        embedding_model (HuggingFaceEmbeddings): The embedding model used to create the DB.
        limit (int): The number of items to print.
    """
    print(f"Loading vector store from: {db_path}")
    try:
        # Load the vector store with the correct embedding function
        vectorstore = FAISS.load_local(
            db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Please ensure you have run 'create_database.py' first and the path is correct.")
        return

    try:
        # Access the internal components of the FAISS object
        index = vectorstore.index
        docstore = vectorstore.docstore
        index_to_docstore_id = vectorstore.index_to_docstore_id

        num_items = index.ntotal
        if num_items == 0:
            print("The vector store is empty.")
            return

        # Ensure we don't try to print more items than exist
        display_limit = min(limit, num_items)
        
        print("\n--- INSPECTING STORED EMBEDDINGS AND TEXT ---")
        print(f"Showing the first {display_limit} of {num_items} embedded chunks:\n")

        for i in range(display_limit):
            # 1. Get the document ID from the FAISS index mapping
            doc_id = index_to_docstore_id[i]
            
            # 2. Get the document content from the docstore using the ID
            doc = docstore.search(doc_id)
            if not doc:
                print(f"--- Chunk {i+1} ---")
                print("Could not find document in docstore for this index.")
                print("-" * 20 + "\n")
                continue

            text_content = doc.page_content.replace('\n', ' ') # Clean up for readability
            
            # 3. Reconstruct the vector embedding directly from the FAISS index
            embedding = index.reconstruct(i)
            
            print(f"--- Chunk {i+1} ---")
            print(f"Text: \"{text_content}\"")
            
            # Print a snippet of the embedding vector to keep it readable
            embedding_snippet = np.concatenate((embedding[:5], embedding[-5:]))
            print(f"Embedding Snippet (first 5 and last 5 dims): {embedding_snippet}")
            print(f"Embedding Dimensions: {len(embedding)}")
            print("-" * 20 + "\n")

    except Exception as e:
        print(f"An error occurred while inspecting the vector store: {e}")


if __name__ == "__main__":
    # Initialize the same embedding model used for creation
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # Run the inspection function
    inspect_vector_store(DB_FAISS_PATH, embeddings, limit=5)