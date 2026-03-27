"""
chatbot.py
----------
Complete chatbot with all 4 memory types.
Switch between them with a CLI flag — no code changes needed.

Usage:
    python chatbot.py                        # default: buffer window (k=10)
    python chatbot.py --memory buffer        # Type 1: full buffer
    python chatbot.py --memory window        # Type 2: buffer window
    python chatbot.py --memory summary       # Type 3: summary buffer
    python chatbot.py --memory vector        # Type 4: vector store
    python chatbot.py --memory window --k 5  # Type 2 with custom window size
    python chatbot.py --session alice        # run as a different session
"""

import argparse
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MEMORY FACTORIES  — only these functions differ per type
# ══════════════════════════════════════════════════════════════════════════════

def make_buffer_history():
    """Type 1 — ChatMessageHistory: keeps every message, forever."""
    from langchain_community.chat_message_histories import ChatMessageHistory
    return ChatMessageHistory()


def make_window_history():
    """
    Type 2 — Buffer Window: plain ChatMessageHistory.
    The window is applied in get_session_history(), not here.
    """
    from langchain_community.chat_message_histories import ChatMessageHistory
    return ChatMessageHistory()


def make_summary_history(max_token_limit: int = 300):
    """Type 3 — Summary Buffer: compresses old turns when token limit exceeded."""
    from langchain.memory import ConversationSummaryBufferMemory
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        max_token_limit=max_token_limit,
        return_messages=True,
    )


def make_vector_history(top_k: int = 3):
    """Type 4 — VectorStore: embeds turns, retrieves top-k relevant ones."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.memory import VectorStoreRetrieverMemory
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        collection_name="chatbot_memory",
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return VectorStoreRetrieverMemory(retriever=retriever)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SHARED SCAFFOLD  — identical regardless of memory type
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise — answer in 2–3 sentences."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

LLM    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
PARSER = StrOutputParser()


def build_bot(memory_type: str, session_id: str, window_k: int = 10):
    """
    Build the bot for the chosen memory type.

    Args:
        memory_type : "buffer" | "window" | "summary" | "vector"
        session_id  : identifies the conversation history store
        window_k    : only used when memory_type == "window"

    Returns:
        (bot, cfg)
    """
    store = {}

    # ── pick the factory ──────────────────────────────────────────────────────
    if memory_type == "buffer":
        factory = make_buffer_history
        window = None

    elif memory_type == "window":
        factory = make_window_history
        window = window_k

    elif memory_type == "summary":
        factory = lambda: make_summary_history(max_token_limit=300)
        window = None

    elif memory_type == "vector":
        factory = make_vector_history
        window = None

    else:
        raise ValueError(f"Unknown memory type: {memory_type!r}")

    # ── session history getter ────────────────────────────────────────────────
    def get_session_history(sid: str):
        if sid not in store:
            store[sid] = factory()
        hist = store[sid]
        # Apply window trimming only for Type 2
        if window is not None and hasattr(hist, "messages"):
            hist.messages = hist.messages[-window:]
        return hist

    chain = PROMPT | LLM | PARSER
    bot   = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = {"configurable": {"session_id": session_id}}
    return bot, cfg


# ══════════════════════════════════════════════════════════════════════════════
# 3.  REPL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_repl(bot, cfg):
    print("\nChatbot ready. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        print("Bot: ", end="", flush=True)
        for chunk in bot.stream({"input": user_input}, config=cfg):
            print(chunk, end="", flush=True)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_LABELS = {
    "buffer":  "ChatMessageHistory (full buffer — no pruning)",
    "window":  "Buffer Window (last k turns)",
    "summary": "ConversationSummaryBufferMemory (compress old turns)",
    "vector":  "VectorStoreRetrieverMemory (semantic retrieval)",
}

def main():
    parser = argparse.ArgumentParser(
        description="LangChain memory-type chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot.py                        # buffer window, k=10 (default)
  python chatbot.py --memory buffer        # full buffer — never forgets
  python chatbot.py --memory window --k 5  # window of 5 messages
  python chatbot.py --memory summary       # summarises old turns
  python chatbot.py --memory vector        # semantic vector retrieval
  python chatbot.py --memory window --session alice  # named session
        """,
    )
    parser.add_argument(
        "--memory", "-m",
        choices=["buffer", "window", "summary", "vector"],
        default="window",
        help="Memory strategy to use (default: window)",
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=10,
        help="Window size for --memory window (default: 10)",
    )
    parser.add_argument(
        "--session", "-s",
        default="user-1",
        help="Session ID — different IDs get isolated histories (default: user-1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  LangChain Memory Demo")
    print(f"  Memory type : {MEMORY_LABELS[args.memory]}")
    if args.memory == "window":
        print(f"  Window size : k={args.k}")
    print(f"  Session ID  : {args.session}")
    print("=" * 60)

    bot, cfg = build_bot(args.memory, session_id=args.session, window_k=args.k)
    run_repl(bot, cfg)


if __name__ == "__main__":
    main()
