"""
memory_4_vectorstore.py
------------------------
Memory type 4: VectorStoreRetrieverMemory

Behaviour : Each turn is embedded and stored in a vector database (Chroma).
            At query time, the top-k most semantically relevant past turns
            are retrieved and injected into the prompt — not all turns.
Token cost: Low per query — only top-k relevant turns enter the prompt,
            regardless of how many total turns exist.
Best for  : Long-lived personal AI assistants, research copilots,
            sessions spanning hundreds or thousands of turns.
Avoid when: Simple short-session apps where the setup cost is not justified.

Change vs. Type 1 (ChatMessageHistory)
---------------------------------------
REMOVE: from langchain_community.chat_message_histories import ChatMessageHistory
REMOVE: def make_history(): return ChatMessageHistory()
ADD   : from langchain_openai import OpenAIEmbeddings
ADD   : from langchain_community.vectorstores import Chroma
ADD   : from langchain.memory import VectorStoreRetrieverMemory
ADD   : def make_history(): builds Chroma + retriever + VectorStoreRetrieverMemory
KEEP  : everything in scaffold.py — unchanged.

Extra install:
    pip install chromadb
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from scaffold import build_bot, run_repl

# ── How many semantically relevant turns to inject per query ──────────────────
TOP_K = 3


# ── The ONLY thing that changes between memory types ──────────────────────────
def make_history():
    """
    Returns VectorStoreRetrieverMemory backed by an in-memory Chroma store.

    For production, replace Chroma() with a persistent store:
        Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    Or swap to any LangChain-compatible vector store:
        Pinecone, FAISS, Weaviate, Qdrant, etc.
    """
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        collection_name="chat_memory",   # keeps sessions isolated
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return VectorStoreRetrieverMemory(retriever=retriever)


if __name__ == "__main__":
    print("=" * 60)
    print("  Memory Type 4 — VectorStoreRetrieverMemory")
    print(f"  Injects top-{TOP_K} semantically relevant past turns.")
    print("  Scales to 1000s of turns with flat token cost.")
    print("=" * 60)
    print("\nNote: First run embeds each message — expect a small delay per turn.\n")

    bot, cfg = build_bot(make_history)

    # Demo: seed facts spread across many turns, then query selectively
    demo_inputs = [
        "My name is Priya.",
        "I work at a fintech startup called Fintella.",
        "My favourite programming language is Python.",
        "I enjoy hiking and photography on weekends.",
        "I am building a RAG system using LangChain.",
        "My team uses FastAPI for our backend services.",
        "What is my name and where do I work?",          # retrieves turns 1 & 2
        "What hobby did I mention?",                      # retrieves turn 4
        "What tech stack am I using?",                   # retrieves turns 5 & 6
    ]

    print(f"[Demo — {len(demo_inputs)} turns with selective semantic retrieval]\n")
    for msg in demo_inputs:
        print(f"You: {msg}")
        print("Bot: ", end="", flush=True)
        for chunk in bot.stream({"input": msg}, config=cfg):
            print(chunk, end="", flush=True)
        print("\n")

    print("[Switching to interactive mode]\n")
    run_repl(bot, cfg)
