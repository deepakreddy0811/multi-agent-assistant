# Multi-Agent Assistant

A small multi-agent system where specialized agents collaborate to answer a
task. A **planner** breaks the task into steps, a **researcher** gathers
information using tools, and a **writer** composes the final answer.

Built with **Python** and a clean, framework-agnostic orchestration loop that
mirrors how **LangGraph** and **CrewAI** work.

---

## What it does

```
USER TASK ──► Planner ──► Researcher ──► Writer ──► ANSWER
                │            │             │
            makes a      calls a tool   writes the
             plan        (calculator,   final answer
                          lookup, …)
```

Each agent has one focused job. The orchestrator owns the shared state and
passes each agent's output to the next. Tools let the researcher *act*, not
just talk — that's the "tool calling / function calling" piece.

---

## Project structure

```
multi-agent-assistant/
├── app/
│   ├── tools.py         # the tools agents can call + a registry
│   ├── agents.py        # Planner, Researcher, Writer
│   ├── orchestrator.py  # coordinates the agents, owns shared state
│   ├── llm.py           # pluggable LLM (fake for offline demo, or OpenAI)
│   └── main.py          # FastAPI server (POST /run)
├── demo.py              # run the full flow from the CLI, no API key
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run the offline demo — watch all three agents work + a real tool call
python demo.py

# Or run the web API
uvicorn app.main:app --reload   # then open http://127.0.0.1:8000/docs
```

The demo runs offline using a deterministic **fake LLM** so you can see the
orchestration without any API key. To use real reasoning, install `openai`,
set `OPENAI_API_KEY`, and change `default_llm` to `openai_llm` in `app/llm.py`.

---

## How it works (the parts you should be able to explain)

**Why multiple agents instead of one big prompt?**
Splitting work across focused agents makes each step more reliable and far
easier to debug. If the answer is wrong you can see *which* agent failed —
the planner, the researcher, or the writer. One giant prompt is a black box.

**What is a "tool" and why do agents need them?**
A tool is a function the agent can call (calculator, lookup, web search, a
database query). LLMs can't do math reliably or access live/private data on
their own. Tools let the agent take real actions and ground its work in facts.

**How does the researcher decide which tool to use?**
It prompts the LLM with the list of available tools and asks it to respond in
a `tool_name | argument` format. The orchestrator parses that, looks the tool
up in the registry, and runs it. This "reason, then act" pattern is the core
idea behind **ReAct** and is how LangChain/CrewAI agents call tools.

**What is the orchestrator's job?**
It owns the shared `state` dict and the order of execution. Agents stay simple
and reusable; the *workflow* lives in one place. LangGraph models this as a
graph of nodes; CrewAI models it as a crew of agents with assigned tasks — same
idea, different vocabulary.

**Where does this map to the frameworks on my resume?**
- Agent = a CrewAI Agent / a LangGraph node.
- Tool registry = LangChain tools / CrewAI tools.
- Orchestrator = a LangGraph StateGraph / a CrewAI Crew.
Building it from scratch first means you understand what those frameworks do
*for* you — which is exactly what a good interviewer wants to hear.

---

## Things I'd improve next (good interview talking points)

- Add a **loop**: let the researcher call multiple tools until it has enough
  info, instead of exactly one (true ReAct / agent loop).
- Add **memory** so agents remember earlier steps across a longer task.
- Plug the RAG project in as a `retrieve` tool, so the researcher can answer
  questions over real documents.
- Add **guardrails / validation** on tool arguments before execution.
- Rebuild it on **LangGraph** to show the same flow as an explicit state graph.
