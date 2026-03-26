import os
import time
import textwrap
from dotenv import load_dotenv
from langsmith import Client

# 1. LOAD ENV & CONFIGURE TRACING *BEFORE* OTHER IMPORTS
load_dotenv()

# Map LangSmith keys to LangChain keys
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# SAFETY CHECK: Only set Endpoint if it actually exists in .env (avoids "None" errors)
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

# ---- NOW IMPORT LANGCHAIN MODULES ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hr-policy-gesci-index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4-turbo"  # or gpt-3.5-turbo

def get_retriever():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def build_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=CHAT_MODEL,
        temperature=0.1
    )

def answer_question(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""
    Answer based on the context below:
    Context: {context}
    Question: {query}
    """
    # This invocation triggers the trace
    response = llm.invoke(prompt)
    return response.content

def verify_langsmith():
    """Checks if the trace landed."""
    print("\n--- Verifying LangSmith Trace ---")
    project_name = os.environ.get("LANGCHAIN_PROJECT")
    client = Client()
    
    # 1. Create Project if missing
    try:
        client.read_project(project_name=project_name)
    except:
        client.create_project(project_name=project_name)
        print(f"Created project: {project_name}")

    # 2. Wait loop to allow background upload to finish
    print("Waiting for trace to upload...", end="", flush=True)
    for _ in range(5): # Wait up to 10 seconds
        time.sleep(2)
        print(".", end="", flush=True)
        runs = list(client.list_runs(project_name=project_name, limit=1))
        if runs:
            print("\n SUCCESS! Trace found.")
            print(f"View Run Here: https://smith.langchain.com/o/{client.read_project(project_name=project_name).\
                tenant_id}/projects/p/{client.read_project(project_name=project_name).id}/r/{runs[0].id}")
            return
    
    print("\n Still no traces found. Double check your API Key and Project Name in .env")

def chat():
    print(f"Chat Ready (Project: {os.getenv('LANGCHAIN_PROJECT')})")
    retriever = get_retriever()
    llm = build_llm()

    # Ask ONE automated question to force a trace for testing
    print("\nRunning automated test query...")
    answer_question("What is this policy about?", retriever, llm)
    print("Test query complete.")

    # Run verification immediately
    verify_langsmith()

    # Enter interactive mode
    while True:
        query = input("\nYour Question (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}: break
        try:
            ans = answer_question(query, retriever, llm)
            print("\nAnswer:", textwrap.fill(ans, 100))
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    chat()