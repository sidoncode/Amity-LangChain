# LangChain Memory Types — Complete Demo

> One scaffold. Four memory strategies. Minimal code changes between each.

A hands-on Python project that demonstrates all four LangChain conversational memory types side-by-side. Every file shares the same prompt, chain, and `RunnableWithMessageHistory` — only the `make_history()` factory changes.

---

## What is LangChain?

LangChain is an open-source framework for building applications powered by **large language models (LLMs)**. It provides composable building blocks — prompts, models, parsers, memory, retrieval, and tools — so you can connect an LLM to your data and logic without writing hundreds of lines of boilerplate.

```
Your Question → Prompt Template → LLM → Output Parser → Answer
```

---


## Memory types — use-case table

| Memory type | Session length | Token cost | Best use case | Avoid when |
|---|---|---|---|---|
| `ChatMessageHistory` | < 20 turns | Unbounded | Demos, prototypes, short Q&A | Production bots with long conversations |
| Buffer Window `k=10` | 20 – 200 turns | Fixed (by k) | Production chatbots, support bots, tutors | Early context is critical to recall later |
| `SummaryBufferMemory` | 200 – 1 000 turns | Medium + 1 extra LLM call | Long support chats, coaching, interview assistants | Latency or cost is tightly constrained |
| `VectorStoreRetrieverMemory` | 1 000+ turns | Low per query | Personal AI, research copilots, long-lived assistants | Simple apps where setup complexity is not justified |

---

<img width="739" height="226" alt="image" src="https://github.com/user-attachments/assets/6d30063a-3046-4c22-abb2-5ed979e9fd27" />

---



### Core concepts

| Concept | What it does |
|---|---|
| **Prompt Templates** | Parameterised prompts — fill `{variables}` at runtime |
| **Chat Models** | Wrappers around OpenAI, Anthropic, Gemini, Ollama, etc. |
| **Output Parsers** | Convert `AIMessage` → `str`, `dict`, Pydantic model |
| **LCEL (pipe `\|`)** | Chain any two Runnables: `prompt \| llm \| parser` |
| **Memory** | Inject conversation history into every prompt automatically |
| **Retrieval (RAG)** | Pull relevant docs from a vector store into the prompt |
| **Agents & Tools** | Let the LLM decide which tools to call and act autonomously |

### Why use LangChain instead of raw API calls?

```python
# Without LangChain — you manage everything manually
messages = []
def chat(user_msg):
    messages.append({"role": "user", "content": user_msg})
    res = client.chat.completions.create(model="gpt-4o", messages=messages)
    reply = res.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

# With LangChain — four lines, same behaviour
chain = prompt | llm | StrOutputParser()
bot   = RunnableWithMessageHistory(chain, get_history, ...)
```

LangChain handles history management, streaming, retries, output parsing, and tool routing. You write the logic, not the glue.

---

## Project structure

```
langchain-memory-demo/
│
├── chatbot.py                  ← Complete chatbot — all 4 types via --memory flag
├── scaffold.py                 ← Shared scaffold used by individual type files
│
├── memory_1_buffer.py          ← Type 1: ChatMessageHistory (full buffer)
├── memory_2_buffer_window.py   ← Type 2: Buffer Window (last k turns)
├── memory_3_summary.py         ← Type 3: ConversationSummaryBufferMemory
├── memory_4_vectorstore.py     ← Type 4: VectorStoreRetrieverMemory
│
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Clone and enter the directory

```bash
git clone https://github.com/your-username/langchain-memory-demo.git
cd langchain-memory-demo
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Type 4 only** — also install Chroma:
> ```bash
> pip install chromadb
> ```

### 4. Set your API key

```bash
cp .env.example .env
# Open .env and paste your OpenAI API key
```

Or export directly:

```bash
export OPENAI_API_KEY="sk-..."
```

### 5. Run the chatbot

```bash
# Default (Buffer Window, k=10) — best starting point
python chatbot.py

# Choose a memory type with --memory
python chatbot.py --memory buffer       # Type 1 — full history
python chatbot.py --memory window       # Type 2 — last k turns (default)
python chatbot.py --memory summary      # Type 3 — summarise old turns
python chatbot.py --memory vector       # Type 4 — semantic retrieval

# Customise the window size (Type 2 only)
python chatbot.py --memory window --k 5

# Use a named session (isolated history per user)
python chatbot.py --memory window --session alice
python chatbot.py --memory window --session bob
```

### 6. Run individual type files (with built-in demos)

Each file runs a preset demo first, then drops into interactive mode:

```bash
python memory_1_buffer.py          # demos full recall across 3 turns
python memory_2_buffer_window.py   # demos forgetting after window overflows
python memory_3_summary.py         # demos summarisation kicking in
python memory_4_vectorstore.py     # demos semantic retrieval from 9 turns
```

---

## The one-swap design

Every memory type shares **identical** prompt, chain, and invocation code. The only thing that changes is `make_history()`:

```python
# ── Shared scaffold — never changes ───────────────────────────────────
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = make_history()   # ← only this line differs
    return store[session_id]

chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
bot   = RunnableWithMessageHistory(chain, get_session_history,
          input_messages_key="input", history_messages_key="history")
cfg   = {"configurable": {"session_id": "user-1"}}

bot.invoke({"input": "Hello!"}, config=cfg)
```

---

## Memory types — what changes

### Type 1 — `ChatMessageHistory` (Buffer)

```python
# ADD
from langchain_community.chat_message_histories import ChatMessageHistory

def make_history():
    return ChatMessageHistory()   # stores every message, forever
```

**Lines to remove vs baseline:** none — this is the baseline.

---

### Type 2 — Buffer Window (last k turns)

```python
# KEEP make_history() the same as Type 1
def make_history():
    return ChatMessageHistory()

# CHANGE get_session_history() — add ONE line:
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = make_history()
    hist = store[session_id]
    hist.messages = hist.messages[-10:]   # ← the window (only new line)
    return hist
```

**Lines removed vs Type 1:** the plain `return store[session_id]` is replaced by the 2-line window block.

---

### Type 3 — `ConversationSummaryBufferMemory`

```python
# REMOVE
# from langchain_community.chat_message_histories import ChatMessageHistory
# def make_history(): return ChatMessageHistory()

# ADD
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

summariser_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def make_history():
    return ConversationSummaryBufferMemory(
        llm=summariser_llm,
        max_token_limit=300,   # summarise when buffer exceeds 300 tokens
        return_messages=True,
    )
```

**Lines removed vs Type 1:** the `ChatMessageHistory` import and its `make_history()`.

---

### Type 4 — `VectorStoreRetrieverMemory`

```python
# REMOVE
# from langchain_community.chat_message_histories import ChatMessageHistory
# def make_history(): return ChatMessageHistory()

# ADD
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory

def make_history():
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    return VectorStoreRetrieverMemory(retriever=retriever)

# Extra install: pip install chromadb
```

**Lines removed vs Type 1:** same as Type 3.

---

## Decision guide

```
Session < 20 turns?
  └─ Yes → ChatMessageHistory (Buffer)

Session 20 – 200 turns?
  └─ Yes → Buffer Window (k=10–20)  ← best default for production

Session 200+ turns, gist of history is enough?
  └─ Yes → ConversationSummaryBufferMemory

Session 200+ turns, semantic precision matters?
  └─ Yes → VectorStoreRetrieverMemory

Multiple users?
  └─ Any type — key sessions by user ID in get_session_history()

Cost is primary constraint?
  └─ Buffer Window — zero extra LLM calls

Latency is primary constraint?
  └─ Buffer Window — no embedding or summarisation overhead
```

> **Golden rule:** Start with Buffer Window (`k=10`). Add complexity only when you hit a real limit — context overflow → switch to Summary; semantic precision needed → switch to VectorStore. You only change `make_history()`.

---

## Optional: LangSmith tracing

LangSmith gives you a visual trace of every prompt, response, and chain step — invaluable for debugging.

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=langchain-memory-demo
OPENAI_API_KEY = "sk - .... "
```

Sign up free at [smith.langchain.com](https://smith.langchain.com).

---

## Further reading

- [LangChain docs](https://python.langchain.com)
- [LangSmith — tracing & observability](https://smith.langchain.com)
- [LangGraph — stateful multi-agent flows](https://langchain-ai.github.io/langgraph/)
- [LCEL reference](https://python.langchain.com/docs/expression_language/)

---

## License

MIT
