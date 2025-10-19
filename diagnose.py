"""
Diagnostic script to check why the LangGraph agent isn't initializing
"""
import os
import sys

print("=" * 70)
print("LANGGRAPH AGENT DIAGNOSTIC")
print("=" * 70)

# Check 1: Environment variables
print("\n[1] Checking Environment Variables...")
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")

print(f"   OPENAI_API_KEY: {'✓ Set' if openai_key else '✗ NOT SET'}")
print(f"   TAVILY_API_KEY: {'✓ Set' if tavily_key else '✗ NOT SET (optional)'}")
print(f"   COHERE_API_KEY: {'✓ Set' if cohere_key else '✗ NOT SET (optional)'}")

# Check 2: Data files
print("\n[2] Checking Data Files...")
csv_path = "book_research/data/space_exploration_books.csv"
pdf_path = "book_research/data/thief_of_sorrows.pdf"

print(f"   CSV file: {'✓ Found' if os.path.exists(csv_path) else '✗ NOT FOUND: ' + csv_path}")
print(f"   PDF file: {'✓ Found' if os.path.exists(pdf_path) else '✗ NOT FOUND: ' + pdf_path}")

# Check 3: Required imports
print("\n[3] Checking Required Dependencies...")

dependencies = [
    ("langchain", "langchain"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_community", "langchain-community"),
    ("langgraph", "langgraph"),
    ("langchain_cohere", "langchain-cohere"),
    ("qdrant_client", "qdrant-client"),
    ("pandas", "pandas"),
    ("fitz", "pymupdf"),
]

missing = []
for module, package in dependencies:
    try:
        __import__(module)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")
        missing.append(package)

# Check 4: Agent components
print("\n[4] Checking Book Agent Components...")
try:
    from book_research.configuration import Configuration
    print("   ✓ Configuration")
except ImportError as e:
    print(f"   ✗ Configuration - {e}")

try:
    from book_research.state import AgentState
    print("   ✓ AgentState")
except ImportError as e:
    print(f"   ✗ AgentState - {e}")

try:
    from book_research.book_agent import build_book_agent
    print("   ✓ build_book_agent")
except ImportError as e:
    print(f"   ✗ build_book_agent - {e}")

# Check 5: Try to initialize agent
print("\n[5] Testing Agent Initialization...")
if missing:
    print(f"   ✗ Cannot test - missing dependencies: {', '.join(missing)}")
else:
    try:
        print("   Attempting to import and initialize...")
        from book_research.configuration import Configuration
        from book_research.state import AgentState
        from book_research.book_agent import build_book_agent

        # Try to create retrievers
        print("   Checking retriever initialization...")

        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Qdrant
        from langchain_core.documents import Document

        if not openai_key:
            print("   ✗ Cannot initialize - OPENAI_API_KEY not set")
        else:
            print("   ✓ All components can be imported")
            print("   NOTE: Full initialization requires running server")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if missing:
    print(f"\n❌ MISSING DEPENDENCIES: {', '.join(missing)}")
    print("\nTo fix, run:")
    print(f"   pip install {' '.join(missing)}")
elif not openai_key:
    print("\n❌ OPENAI_API_KEY not set in .env file")
    print("\nTo fix, check .env file format:")
    print("   OPENAI_API_KEY=sk-...")
elif not os.path.exists(csv_path) or not os.path.exists(pdf_path):
    print("\n❌ DATA FILES MISSING")
    print(f"\nExpected locations:")
    print(f"   {csv_path}")
    print(f"   {pdf_path}")
else:
    print("\n✅ All prerequisites look good!")
    print("\nIf agent still not working, check server startup logs for errors.")

print("\n" + "=" * 70)
