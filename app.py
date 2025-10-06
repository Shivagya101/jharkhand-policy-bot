import streamlit as st
import sys
from pathlib import Path

# Add the current directory to Python path to import run_chat
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import everything from your run_chat.py
from run_chat import app, llm, retriever, llm_call_count

# Page configuration
st.set_page_config(
    page_title="Jharkhand Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .status-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'session_llm_calls' not in st.session_state:
    st.session_state.session_llm_calls = 0

def get_current_llm_count():
    """Get the current LLM call count from run_chat module"""
    import run_chat
    return run_chat.llm_call_count

def main():
    # Header
    st.title("🏛️ Jharkhand Government Policy Assistant")
    st.markdown("Ask questions about Jharkhand Government policies, schemes, and programs")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.total_queries)
        with col2:
            st.metric("LLM Calls", get_current_llm_count())
        
        st.divider()
        
        st.header("🔧 System Status")
        
        # Check if models are loaded
        try:
            if llm is not None and retriever is not None:
                st.markdown("""
                <div class="status-box status-success">
                    <strong>✅ System Ready</strong><br>
                    All models loaded successfully
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-box status-info">
                    <strong>⏳ Loading...</strong><br>
                    Initializing models
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you explore Jharkhand Government policies:
        - 🔍 Searches through government policy documents
        - ✅ Verifies document relevance
        - 🔄 Refines queries for better results
        - 📝 Handles large policy documents
        - 💡 Provides accurate, context-based answers
        - 🏛️ Focuses on Jharkhand state policies and schemes
        """)
        
        st.divider()
        
        st.header("📋 Graph Workflow")
        st.markdown("""
        ```
        Retrieve → Grade Documents
            ↓           ↓
        Rewrite ←   Check Context
                        ↓
                 Summarize/Direct
                        ↓
                    Generate
        ```
        """)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.rerun()
        
        if st.button("🔄 Reset LLM Counter", use_container_width=True):
            import run_chat
            run_chat.llm_call_count = 0
            st.session_state.session_llm_calls = 0
            st.rerun()
    
    # Main content area
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            if "llm_calls" in message:
                st.caption(f"ℹ️ LLM calls for this query: {message['llm_calls']}")
            if "iterations" in message and message["iterations"] > 0:
                st.caption(f"🔄 Query rewrite iterations: {message['iterations']}")
    
    # Query input
    query = st.chat_input("Ask about Jharkhand Government policies and schemes...")
    
    if query:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {query}
        </div>
        """, unsafe_allow_html=True)
        
        # Process query
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # Get initial LLM call count
                initial_calls = get_current_llm_count()
                
                # Invoke the compiled graph from run_chat.py
                inputs = {"question": query, "iterations": 0}
                final_state = app.invoke(inputs)
                
                # Get the answer and metrics
                answer = final_state.get("generation", "I couldn't generate an answer.")
                calls_made = get_current_llm_count() - initial_calls
                iterations = final_state.get("iterations", 0)
                num_docs = len(final_state.get("documents", []))
                
                # Add assistant message to chat
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "llm_calls": calls_made,
                    "iterations": iterations,
                    "num_docs": num_docs
                })
                st.session_state.total_queries += 1
                
                # Display assistant message
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.caption(f"📊 LLM calls: {calls_made}")
                with metric_col2:
                    st.caption(f"📄 Documents used: {num_docs}")
                with metric_col3:
                    if iterations > 0:
                        st.caption(f"🔄 Rewrites: {iterations}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)  # This will show the full traceback
    
    # Show helpful message if no chat history
    if len(st.session_state.chat_history) == 0:
        st.info("👋 Welcome! Ask me anything about Jharkhand Government policies, schemes, and programs.")
        
        # Show example queries
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - What are the eligibility criteria for [scheme name]?
            - Explain the implementation process of...
            - What benefits are provided under...
            - What are the key objectives of [policy name]?
            - How do I apply for...
            - What documents are required for...
            - What is the budget allocation for...
            """)

if __name__ == "__main__":
    main()