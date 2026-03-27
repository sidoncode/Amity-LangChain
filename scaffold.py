"""
scaffold.py
-----------
Shared LangChain chain scaffold used by every memory type.

Design principle:
  - build_bot(make_history_fn) is the only public function.
  - Pass a factory that returns any ChatMessageHistory-compatible object.
  - The prompt, chain, and RunnableWithMessageHistory never change.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# ── Shared prompt (identical for all memory types) ────────────────────────────
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise — answer in 2–3 sentences."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# ── Shared LLM ────────────────────────────────────────────────────────────────
_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Shared output parser ──────────────────────────────────────────────────────
_PARSER = StrOutputParser()


def build_bot(make_history_fn):
    """
    Build a stateful chatbot from any memory factory.

    Args:
        make_history_fn: Callable[[], BaseChatMessageHistory]
            Factory that returns a fresh history object per session.

    Returns:
        (bot, cfg) tuple ready to invoke.
    """
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = make_history_fn()
        return store[session_id]

    chain = _PROMPT | _LLM | _PARSER

    bot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = {"configurable": {"session_id": "user-1"}}
    return bot, cfg


def run_repl(bot, cfg):
    """
    Simple REPL loop — type 'quit' or 'exit' to stop.
    Streams responses token-by-token.
    """
    print("\nChatbot ready. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        print("Bot: ", end="", flush=True)
        for chunk in bot.stream({"input": user_input}, config=cfg):
            print(chunk, end="", flush=True)
        print()
