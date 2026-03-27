"""
memory_2_buffer_window.py
--------------------------
Memory type 2: Buffer Window (last k turns only)

Behaviour : Keeps only the most recent K messages in context.
            Older messages are silently dropped before each invoke.
Token cost: Fixed — bounded by K regardless of session length.
Best for  : Production chatbots, customer support bots, tutors.
            Most chat apps only need the last 10–20 turns of context.
Avoid when: The user's early statements are critical later in the session
            (e.g. the user mentions their account number on turn 1 and
            references it on turn 50 — it will be gone).

Change vs. Type 1 (ChatMessageHistory)
---------------------------------------
KEEP : make_history() — same ChatMessageHistory returned.
CHANGE: get_session_history() — add ONE line: hist.messages = hist.messages[-K:]
REMOVE: the plain `return store[session_id]` in get_session_history.
KEEP : everything in scaffold.py — unchanged.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── Tune this to control how many messages stay in context ────────────────────
WINDOW_K = 6   # keeps last 6 messages (3 human + 3 assistant turns)


# ── Shared prompt / LLM / parser (identical to scaffold.py) ──────────────────
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise — answer in 2–3 sentences."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
_LLM    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_PARSER = StrOutputParser()


# ── Custom get_session_history — the one meaningful change ───────────────────
store = {}

def make_history():
    """Same as Type 1 — plain ChatMessageHistory."""
    return ChatMessageHistory()

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = make_history()
    hist = store[session_id]
    # ▼ THE WINDOW — trim to last K messages before every invoke
    hist.messages = hist.messages[-WINDOW_K:]
    return hist


# ── Wire up ───────────────────────────────────────────────────────────────────
def build_bot():
    chain = _PROMPT | _LLM | _PARSER
    bot   = RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = {"configurable": {"session_id": "user-1"}}
    return bot, cfg


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Memory Type 2 — Buffer Window  (k={WINDOW_K} messages)")
    print("  Older turns are dropped. Token cost stays flat.")
    print("=" * 60)

    bot, cfg = build_bot()

    # Demo: intentionally overflow the window to show forgetting
    demo_inputs = [
        "My name is Priya.",                           # turn 1 — will be forgotten
        "I love hiking on weekends.",                  # turn 2 — will be forgotten
        "I am a machine learning engineer.",           # turn 3 — will be forgotten
        "I work with Python every day.",               # turn 4 — kept
        "My favourite framework is LangChain.",        # turn 5 — kept
        "What is my name?",                            # turn 6 — 'Priya' was in turn 1 — gone!
    ]

    print(f"\n[Demo — {len(demo_inputs)} turns, window={WINDOW_K}. Watch early facts disappear.]\n")
    for msg in demo_inputs:
        print(f"You: {msg}")
        print("Bot: ", end="", flush=True)
        for chunk in bot.stream({"input": msg}, config=cfg):
            print(chunk, end="", flush=True)
        print()
        msgs_in_context = min(len(store.get("user-1", ChatMessageHistory()).messages), WINDOW_K)
        print(f"     [messages currently in context window: {msgs_in_context} / {WINDOW_K}]\n")

    print("\n[Switching to interactive mode]\n")
    from scaffold import run_repl
    run_repl(bot, cfg)
