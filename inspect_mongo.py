import pymongo
from pymongo import MongoClient
from bson.json_util import dumps

# --- Configuration ---
MONGO_URI = "mongodb+srv://dbUSERsidhss:heracross54321@cluster0.zvyfclc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "rag_database"
COLLECTION_NAME = "parent_documents"

def print_collection_entries(uri: str, db_name: str, coll_name: str):
    client = None
    try:
        # 1. Establish connection to the MongoDB cluster
        client = MongoClient(uri)
        client.admin.command('ping') # Check connection status
        print(f"✅ Connected to MongoDB cluster.")

        # 2. Access the specific database and collection
        db = client[db_name]
        collection = db[coll_name]
        
        # 3. Use .find({}) to retrieve ALL documents (entries)
        # .find() returns a cursor object.
        cursor = collection.find({}) 

        print(f"\n--- Entries in '{coll_name}' Collection ({db_name} DB) ---")
        
        document_count = 0
        # 4. Iterate through the cursor and print each document
        for document in cursor:
            document_count += 1
            print(f"Document {document_count}:")
            # Use bson.json_util.dumps to properly serialize ObjectId, datetime, etc.
            print(dumps(document, indent=2))
            print("-" * 30)

        if document_count == 0:
            print("The collection is empty.")

    except Exception as e:
        print(f"\n❌ Error processing MongoDB request: {e}")
        print("Ensure the database and collection names are correct and the user has read access.")
    
    finally:
        if client:
            client.close()
            print("Connection closed.")

if __name__ == "__main__":
    print_collection_entries(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)