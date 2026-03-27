"""
memory_1_buffer.py
------------------
Memory type 1: ChatMessageHistory (Buffer)

Behaviour : Keeps the ENTIRE conversation history, forever.
Token cost: Grows linearly with every turn.
Best for  : Short demos, prototypes, sessions under ~20 turns.
Avoid when: Running a production chatbot with long conversations
            — you will eventually hit ContextLengthExceeded.

Change vs. baseline
-------------------
ADD  : from langchain_community.chat_message_histories import ChatMessageHistory
ADD  : def make_history(): return ChatMessageHistory()
KEEP : everything in scaffold.py — unchanged.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from scaffold import build_bot, run_repl


# ── The ONLY thing that changes between memory types ──────────────────────────
def make_history():
    """Returns a plain in-memory chat history — stores every message."""
    return ChatMessageHistory()


# ── Wire up and run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Memory Type 1 — ChatMessageHistory (Buffer)")
    print("  Stores every message. No pruning. No summarisation.")
    print("=" * 60)

    bot, cfg = build_bot(make_history)

    # Demo: show that it remembers everything across turns
    demo_inputs = [
        "My name is Priya and I am a machine learning engineer.",
        "I am currently building a RAG system in Python.",
        "What is my name and what am I working on?",   # should recall both facts
    ]
    print("\n[Demo mode — running 3 preset turns]\n")
    for msg in demo_inputs:
        print(f"You: {msg}")
        print("Bot: ", end="", flush=True)
        for chunk in bot.stream({"input": msg}, config=cfg):
            print(chunk, end="", flush=True)
        print()

    print("\n[Switching to interactive mode]\n")
    run_repl(bot, cfg)
