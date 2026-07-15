# Multi-Agent Assistant

A multi-agent system where specialized agents collaborate to answer a task. A **planner** decomposes the request, a **researcher** gathers information using tools, and a **writer** composes the final answer — coordinated by an orchestrator that owns the shared state.

Built with **Python** and **FastAPI**, with a framework-agnostic orchestration loop that mirrors the patterns behind LangGraph and CrewAI.

---

## Why

A single prompt asked to do everything is hard to steer and harder to debug — when the output is wrong, there's no way to tell which part of the reasoning broke. Splitting the work across agents with narrow responsibilities makes each step inspectable and gives failures an address. Tools extend the same idea to action: rather than generating a plausible-looking number, the researcher calls a calculator and returns a real one.

---

## How it works

```
USER TASK ──► Planner ──► Researcher ──► Writer ──► ANSWER
                 │            │             │
             decomposes    selects and   synthesizes
             the task      runs a tool   the findings
```

Each agent receives the prior step's output through a shared state dict held by the orchestrator. Agents stay stateless and reusable; the workflow lives in one place.

**Tool calling.** The researcher is prompted with the available tools and asked to reply in a `tool_name | argument` format. The orchestrator parses that response, resolves the tool from a registry, and executes it:

```
Task:      "What is 24 * (8 + 5)?"
Decision:  calculator | 24 * (8 + 5)
Result:    312
```

The model decides, the system executes. That reason-then-act loop is the ReAct pattern.

---

## Project structure

```
multi-agent-assistant/
├── app/
│   ├── tools.py         # tool implementations + registry
│   ├── agents.py        # Planner, Researcher, Writer
│   ├── orchestrator.py  # coordination and shared state
│   ├── llm.py           # swappable LLM interface
│   └── main.py          # FastAPI server (POST /run)
├── demo.py              # CLI demo, runs without an API key
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# CLI demo — shows the full planner → researcher → writer flow
python demo.py

# Or run the API
uvicorn app.main:app --reload   # http://127.0.0.1:8000/docs
```

The demo runs offline against a deterministic stub LLM so the orchestration is observable without an API key. For real reasoning, install `openai`, set `OPENAI_API_KEY`, and point `default_llm` at `openai_llm` in `app/llm.py`.

---

## Adding a tool

Tools are plain functions registered by name:

```python
# app/tools.py
def reverse_text(text: str) -> str:
    return text[::-1]

TOOLS["reverse_text"] = Tool(
    "reverse_text",
    "Reverses the characters in the given text.",
    reverse_text,
)
```

The registry is what the researcher's prompt is built from, so a newly registered tool becomes available to the agent immediately — no changes to the agent or orchestrator.

---

## Design decisions

**Agents as roles, not classes of magic.** Each agent is an LLM plus a role-specific prompt and a single method. The interesting behaviour comes from composition, not from any individual agent being clever.

**The registry as the tool interface.** Tools are looked up by name at execution time, which keeps the agent decoupled from the tool implementations and makes the toolset extensible from one file.

**Orchestration separated from agents.** The orchestrator owns execution order and shared state. Agents don't know about each other, so the workflow can be rearranged without touching them.

**A stub LLM alongside the real one.** The `(prompt) -> str` interface means a deterministic stub can stand in for a live model, which makes the orchestration runnable offline and testable without spending tokens.

---

## Roadmap

- Loop the researcher over multiple tool calls until it has sufficient information (full ReAct loop)
- Add memory so agents retain context across steps in longer tasks
- Expose a document retriever as a tool, enabling grounded research over a corpus
- Validate tool arguments before execution
- Port the same flow to LangGraph as an explicit state graph
