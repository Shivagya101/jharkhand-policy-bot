import sys
from pathlib import Path

# --- Ensure current directory is in sys.path ---
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# --- Import app from run_chat.py ---
try:
    from run_chat import app
except ImportError as e:
    print(f"❌ Error importing 'app' from run_chat.py: {e}")
    print("Make sure 'run_chat.py' is in the same folder as this script.")
    sys.exit(1)

# --- Draw LangGraph Workflow ---
print("\n--- LangGraph Workflow ASCII Diagram ---")
try:
    graph = app.get_graph()
    ascii_graph = graph.draw_ascii()  # native ASCII diagram
    print(ascii_graph)
    print("--------------------------------------")
except Exception as e:
    print(f"\n❌ FAILED to generate ASCII graph. Error: {e}")
