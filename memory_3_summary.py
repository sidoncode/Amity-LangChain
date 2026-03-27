"""
memory_3_summary.py
--------------------
Memory type 3: ConversationSummaryBufferMemory

Behaviour : Keeps recent messages verbatim.
            When the buffer exceeds max_token_limit, older messages are
            compressed into a single summary message by the LLM.
            The summary replaces the raw old messages in the prompt.
Token cost: Medium — recent turns are raw, older turns are summarised.
            Costs one extra LLM call whenever summarisation triggers.
Best for  : Long support sessions, interview assistants, coaching bots.
            Sessions where gist of old context matters but exact wording does not.
Avoid when: Latency or cost is tightly constrained — every summarisation
            adds an extra round-trip to the LLM.

Change vs. Type 1 (ChatMessageHistory)
---------------------------------------
REMOVE: from langchain_community.chat_message_histories import ChatMessageHistory
REMOVE: def make_history(): return ChatMessageHistory()
ADD   : from langchain.memory import ConversationSummaryBufferMemory
ADD   : summariser_llm = ChatOpenAI(...)
ADD   : def make_history(): return ConversationSummaryBufferMemory(llm=..., max_token_limit=...)
KEEP  : everything in scaffold.py — unchanged.
"""

from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from scaffold import build_bot, run_repl


# ── Separate LLM used only for summarisation ──────────────────────────────────
# You can use a cheaper/faster model here since summarisation is low-stakes.
_SUMMARISER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tune: when the buffer exceeds this many tokens, old messages are summarised.
MAX_TOKEN_LIMIT = 300


# ── The ONLY thing that changes between memory types ──────────────────────────
def make_history():
    """
    Returns ConversationSummaryBufferMemory.
    - Keeps recent messages verbatim (up to max_token_limit).
    - Automatically summarises older messages when limit is exceeded.
    - return_messages=True is required for MessagesPlaceholder injection.
    """
    return ConversationSummaryBufferMemory(
        llm=_SUMMARISER_LLM,
        max_token_limit=MAX_TOKEN_LIMIT,
        return_messages=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  Memory Type 3 — ConversationSummaryBufferMemory")
    print(f"  Summarises when buffer exceeds {MAX_TOKEN_LIMIT} tokens.")
    print("  Watch for a [SUMMARY] line appearing in the history.")
    print("=" * 60)

    bot, cfg = build_bot(make_history)

    demo_inputs = [
        "My name is Priya. I am a machine learning engineer at a fintech startup.",
        "I have been building a document Q&A system using LangChain and ChromaDB.",
        "My biggest challenge is making the retrieval step accurate for long PDFs.",
        "I also want to add streaming so users see tokens as they are generated.",
        "What have I told you about myself so far?",  # tests whether summary was formed
    ]

    print(f"\n[Demo — {len(demo_inputs)} turns. The model will summarise when buffer fills.]\n")
    for i, msg in enumerate(demo_inputs, 1):
        print(f"You: {msg}")
        print("Bot: ", end="", flush=True)
        for chunk in bot.stream({"input": msg}, config=cfg):
            print(chunk, end="", flush=True)
        print()
        # Peek at what is currently stored (summary + recent messages)
        session = bot.get_session_history("user-1") if hasattr(bot, "get_session_history") else None
        print(f"     [Turn {i} complete]\n")

    print("\n[Switching to interactive mode]\n")
    run_repl(bot, cfg)
