# Comprehensive Run Guide & File Reference

This guide explains how to run the LAWBOT application, the system architecture, what data is stored where, and the purpose of every file in the project.

---

## 🚀 Quick Start (TL;DR)

```powershell
# 1. Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Set up environment variables (create .env file or use PowerShell)
# .env file format:
# MONGO_DB_URI="your_mongo_uri_here"
# COHERE_API_KEY="your_cohere_api_key_here"

# 3. Place your policy documents as .txt files in data/ folder

# 4. Create the database and vector index
python create_database.py

# 5. Run the Streamlit application
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## 📊 System Architecture Overview

```
Data Files (.txt)
    ↓
[Create Database] → Chunks text using splitters
    ↓
├─→ Parent Documents (full chunks) → MONGODB (rag_database.parent_documents)
└─→ Child Embeddings (small chunks) → FAISS (local vector_store/db_faiss)
    ↓
[Run Chat] → Uses LangGraph for multi-step retrieval & generation
    ↓
Streamlit UI (app.py) OR CLI (run_chat.py)
```

---

## 💾 What's Stored Where

### **MongoDB**

Stores **parent documents** (full chunks of ~500 tokens with 70 token overlap):

- **Database**: `rag_database`
- **Collection**: `parent_documents`
- **Fields**:
  - `_id`: Document ID
  - `page_content`: Full text of the parent chunk
  - `metadata`: Source file information and metadata
- **Purpose**: Original full text retrieved by the system for generating answers

### **FAISS (Local)**

Stores **child embeddings** (small chunks of ~100 tokens):

- **Location**: `vector_store/db_faiss/` (local directory on your machine)
- **Files created**:
  - `index.faiss`: The vector index with all embeddings
  - `index.pkl`: Metadata and document mapping
  - Additional metadata files
- **Embedding Model**: Cohere `embed-multilingual-v3.0` (1024-dimensional vectors)
- **Purpose**: Fast similarity search to find relevant chunks
- **Storage**: ✅ **Local (not cloud)** - embeddings are stored on your machine

### **Pinecone** (Optional - Not Currently Used)

- Found in `pnc.py` for optional production migration only
- Would mirror FAISS but in cloud for scalability
- ⚠️ **Note**: Uses different embedding model (`sentence-transformers/all-MiniLM-L6-v2`), so not currently integrated

---

## 📋 Step-by-Step Setup & Execution

### **1. Prerequisites**

- Python 3.9+ installed
- Working internet connection (for Cohere API calls)
- MongoDB Atlas account (or local MongoDB) with connection URI
- Cohere API key

### **2. Create Virtual Environment & Install Dependencies**

```powershell
# Create virtual environment
python -m venv venv

# Activate (PowerShell)
venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

Minimal install (if you prefer):

```powershell
pip install cohere pymongo faiss-cpu streamlit python-dotenv langchain langgraph
```

### **3. Setup Environment Variables**

**Option A: Create `.env` file** (recommended)

In the project root directory, create a file named `.env`:

```
MONGO_DB_URI="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=YourApp"
COHERE_API_KEY="your_cohere_api_key_here"
PINECONE_API_KEY="optional_only_for_production"
```

**Option B: Set in PowerShell** (temporary for current session)

```powershell
$env:MONGO_DB_URI = 'your_mongo_uri'
$env:COHERE_API_KEY = 'your_cohere_key'
```

⚠️ **Important**: Do NOT commit `.env` to source control

### **4. Prepare Your Data**

1. Place all policy documents as `.txt` files in the `data/` folder
2. If you have PDFs, convert them to text first:

```powershell
python -c "
from pypdf import PdfReader
r = PdfReader('path\to\file.pdf')
text = '\n'.join(page.extract_text() or '' for page in r.pages)
open('data\output.txt','w', encoding='utf-8').write(text)
"
```

3. Ensure all `.txt` files are in UTF-8 encoding

### **5. Create Database & Index**

```powershell
python create_database.py
```

**What this does:**

- Loads all `.txt` files from `data/` folder
- Splits documents into:
  - **Parent chunks**: 500 tokens (70 token overlap)
  - **Child chunks**: 100 tokens (40 token overlap)
- Creates Cohere embeddings for child chunks
- Saves FAISS index locally to `vector_store/db_faiss/`
- Saves parent documents to MongoDB collection `rag_database.parent_documents`
- Displays progress and confirmation

**Output:**

- `vector_store/db_faiss/index.faiss` and metadata files (created locally)
- Parent documents added to MongoDB

### **6. Optional: Verify Your Data**

```powershell
# View MongoDB parent documents
python inspect_mongo.py

# To get document count and sample data from Mongo
```

### **7. Run the Application**

**Option A: Streamlit UI** (Recommended - Beautiful Interface)

```powershell
streamlit run app.py
```

- Opens automatically to `http://localhost:8501`
- Features:
  - Real-time chat interface
  - LLM call counter
  - Document retrieval metrics
  - Query rewrite tracking
  - Chat history
  - System status dashboard

**Option B: CLI Interface** (No UI)

```powershell
python run_chat.py
```

- Simple command-line loop
- Type question, press Enter
- Get answer + LLM call count
- No fancy UI, but fully functional

---

## 🔄 How the Chat Works (LangGraph Workflow)

```
RETRIEVE
  ↓ Query FAISS for top-k similar child chunks
  ↓
GRADE_DOCUMENTS
  ↓ LLM evaluates if retrieved docs are relevant
  ↓
  ├─→ [If not relevant AND iterations < 3]
  │   ↓
  │   REWRITE_QUERY
  │   ↓ Rephrase question for better search
  │   ↓ (Loop back to RETRIEVE)
  │
  └─→ [If relevant]
      ↓
      CHECK_CONTEXT_SIZE
      ↓
      ├─→ [If context > 3000 tokens]
      │   ↓
      │   SUMMARIZE_MAP
      │   ↓ Summarize each document individually
      │
      └─→ [If context acceptable]
          ↓ (Direct path)
          ↓
      GENERATE
      ↓ LLM generates final answer
      ↓
      RETURN RESULT
```

**Key Configuration:**

- LLM Model: `ChatCohere` with `command-a-03-2025`
- Temperature: 0.1 (more deterministic)
- Max Query Rewrites: 3 iterations
- Context Token Limit: ~3000 tokens
- Max Tokens Tracked: Yes (displays LLM call count)

---

## 📁 Complete File Reference

### **Core Application Files**

#### **`app.py`**

**Purpose**: Streamlit frontend UI for the chatbot

- **Functionality**:
  - Builds beautiful web interface for chat
  - Manages session state and chat history
  - Displays metrics (LLM calls, documents used, rewrite iterations)
  - Real-time message display
  - System status indicator
  - Clear history and reset counter buttons
  - Example questions for users
- **Entry Point**: `streamlit run app.py`
- **Output**: Web UI at `http://localhost:8501`
- **Dependencies**: Streamlit, run_chat module

#### **`run_chat.py`**

**Purpose**: LangGraph RAG pipeline - the core intelligence engine

- **Functionality**:
  - Defines `GraphState` TypedDict (memory/state management)
  - Implements 6 nodes:
    1. `retrieve()`: Fetches similar docs from FAISS
    2. `grade_documents()`: LLM grades relevance
    3. `rewrite_query()`: Rephrases question for better search
    4. `check_context_size()`: Checks if context is too large
    5. `summarize_map()`: Summarizes large contexts
    6. `generate()`: LLM generates final answer
  - Implements conditional edges and decision logic
  - Compiles StateGraph into executable workflow
  - Tracks LLM call count globally
- **Entry Point**: `python run_chat.py` (CLI mode) or imported by `app.py`
- **Output**: Answer + LLM call metrics
- **Key Variables**:
  - `llm`: ChatCohere instance
  - `retriever`: ParentDocumentRetriever
  - `llm_call_count`: Global counter
  - `app`: Compiled LangGraph

#### **`create_database.py`**

**Purpose**: Data indexing and database initialization

- **Functionality**:
  - Loads all `.txt` files from `data/` folder
  - Creates parent and child text splitters
  - Initializes Cohere embedding model
  - Creates FAISS vector store locally
  - Connects to MongoDB for parent document storage
  - Uses `ParentDocumentRetriever` for hybrid chunking
  - Saves FAISS index and adds documents to MongoDB
- **Entry Point**: `python create_database.py`
- **Outputs**:
  - `vector_store/db_faiss/` directory with index files
  - Parent documents in MongoDB `rag_database.parent_documents`
- **Environment Variables Required**:
  - `COHERE_API_KEY`
  - `MONGO_DB_URI`
- **Key Functions**:
  - `load_text_files()`: Loads `.txt` files with UTF-8 encoding
  - `get_embedding_model()`: Initializes Cohere embeddings
  - `get_retriever()`: Sets up ParentDocumentRetriever with MongoDB

#### **`mongo_docstore.py`**

**Purpose**: Custom MongoDB wrapper for LangChain

- **Functionality**:
  - Implements `BaseStore[str, Document]` interface
  - Provides custom MongoDB backend for document storage
  - Implements CRUD operations for parent documents:
    - `mget()`: Retrieve documents by IDs
    - `mset()`: Store/update documents
    - `mdelete()`: Remove documents
    - `yield_keys()`: Iterate all keys
  - Uses bulk write operations for efficiency
- **Used By**: `create_database.py` and `run_chat.py`
- **Database**: MongoDB collection specified at initialization
- **No Direct Entry Point**: This is a library module

---

### **Utility & Inspection Files**

#### **`inspect_mongo.py`**

**Purpose**: Debug tool to inspect MongoDB contents

- **Functionality**:
  - Connects to MongoDB using hardcoded URI
  - Retrieves all parent documents from `rag_database.parent_documents`
  - Displays document count and full content
  - Useful for verification and debugging
- **Entry Point**: `python inspect_mongo.py`
- **Output**: Console output of all stored documents
- **⚠️ Important**: Has hardcoded MongoDB URI - update before running
- **Use Case**: Verify that `create_database.py` successfully stored documents

#### **`pnc.py`**

**Purpose**: Optional migration script from FAISS → Pinecone (production deployment)

- **Functionality**:
  - Loads FAISS index from local storage
  - Extracts vectors and metadata
  - Initializes Pinecone client
  - Creates or reuses Pinecone index
  - Uploads vectors in batches (batch size: 100)
  - Handles error logging for failed batches
- **Entry Point**: `python pnc.py`
- **Prerequisite**: `PINECONE_API_KEY` environment variable
- **⚠️ Important Issues**:
  - Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dims)
  - FAISS uses Cohere embeddings (1024 dims)
  - **Dimension mismatch**: Must update `pnc.py` to use Cohere embeddings for consistency
- **Use Case**: Scaling to production with cloud vector database
- **Currently**: Not used in the active pipeline

---

### **Configuration & Data Files**

#### **`requirements.txt`**

**Purpose**: Python dependency specifications

- **Contains**: 50+ packages for:
  - LLM interactions: `cohere`, `langchain*`, `langgraph*`
  - Vector databases: `faiss-cpu`, `sentence-transformers`
  - Data processing: `pymongo`, `pypdf`, `pandas`, `numpy`
  - Web UI: `streamlit`, `altair`
  - Utilities: `python-dotenv`, `requests`, `pydantic`
- **Installation**: `pip install -r requirements.txt`

#### **`.env` (Create This)**

**Purpose**: Store sensitive credentials

- **Required Variables**:
  ```
  MONGO_DB_URI="your_mongodb_connection_string"
  COHERE_API_KEY="your_cohere_api_key"
  ```
- **Optional Variables**:
  ```
  PINECONE_API_KEY="your_pinecone_key"
  ```
- **⚠️ Important**: Do NOT commit to version control
- **Security**: Add to `.gitignore`

#### **`data/` (Folder)**

**Purpose**: Input folder for document ingestion

- **Contents**: Place all policy `.txt` files here
- **Supported Format**: UTF-8 encoded `.txt` files
- **Used By**: `create_database.py`
- **Examples**:
  - `jharkhand_schemes.txt`
  - `page9_jharGov.txt`
  - `page10_jharGov.txt`
  - `1.txt`, `3.txt`, `4.txt`, etc.

#### **`vector_store/db_faiss/` (Folder - Auto-created)**

**Purpose**: Local storage for FAISS vector index

- **Created By**: `create_database.py`
- **Contains**:
  - `index.faiss`: Main index file with all embeddings
  - `index.pkl`: Metadata and document mapping
  - Additional metadata files
- **Used By**: `run_chat.py` for retrieval
- **Local Storage**: ✅ All data stored on your machine

#### **`Readme` (Optional)**

**Purpose**: Project overview

- Contains general project information

#### **`RUN_INSTRUCTIONS.md` (This File)**

**Purpose**: Complete setup and execution guide

- Explains architecture, setup, file purposes, and troubleshooting

---

## 🔧 Common Operations

### **Full Workflow (Fresh Start)**

```powershell
# 1. Setup
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
# Create .env file with MONGO_DB_URI and COHERE_API_KEY

# 3. Index
python create_database.py

# 4. Run
streamlit run app.py
```

### **Just Run the App (After First Setup)**

```powershell
venv\Scripts\Activate.ps1
streamlit run app.py
```

### **Check Stored Data**

```powershell
python inspect_mongo.py
```

### **Rebuild Index (Update Documents)**

```powershell
# 1. Update files in data/
# 2. Run
python create_database.py
# 3. Restart app
```

### **Use CLI Instead of Web UI**

```powershell
python run_chat.py
# Type your question and press Enter
```

### **Migrate to Production (Pinecone)**

```powershell
# Update .env with PINECONE_API_KEY
# Fix pnc.py to use Cohere embeddings (dim 1024)
python pnc.py
```

---

## ❌ Troubleshooting

| Error                              | Cause                          | Solution                                     |
| ---------------------------------- | ------------------------------ | -------------------------------------------- |
| `COHERE_API_KEY not found`         | Missing API key                | Set in `.env` or `$env:COHERE_API_KEY`       |
| `MONGO_DB_URI not found`           | Missing MongoDB URI            | Set in `.env` or environment                 |
| `Connection refused`               | MongoDB offline/unreachable    | Check MongoDB Atlas connection, whitelist IP |
| `FAISS index not found`            | Never ran `create_database.py` | Run `python create_database.py` first        |
| `No such file or directory: data/` | Missing data folder            | Create `data/` folder and add `.txt` files   |
| `Cannot load FAISS`                | Embeddings dimension mismatch  | Ensure same embedding model used             |
| `Streamlit port already in use`    | Port 8501 occupied             | `streamlit run app.py --server.port 8502`    |

---

## 📊 Data Flow Summary

1. **Input**: `.txt` files in `data/`
2. **Processing**: `create_database.py` chunks and embeds
3. **Storage**:
   - FAISS (local): Child embeddings for search
   - MongoDB: Parent documents for context
4. **Retrieval**: `run_chat.py` queries FAISS + MongoDB
5. **Generation**: LLM generates answer using LangGraph
6. **Output**: Streamlit UI or CLI display

---

## 🎯 Key Takeaways

✅ **Embeddings**: Stored locally in FAISS (not cloud)
✅ **Parent Documents**: Stored in MongoDB
✅ **Pinecone**: Optional, not currently integrated
✅ **LLM Model**: Cohere `command-a-03-2025`
✅ **Embedding Model**: Cohere `embed-multilingual-v3.0`
✅ **Multi-step RAG**: Uses LangGraph for intelligent retrieval
✅ **Easy UI**: Streamlit for user-friendly interface
