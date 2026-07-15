"""
orchestrator.py — Coordinates the agents.

This is the "crew" / "graph" layer. It owns the shared state and decides the
order agents run in. Keeping orchestration separate from the agents themselves
is a clean design: agents stay simple and reusable, and the workflow lives in
one place you can reason about.

Flow:
    Planner.plan(task)          -> steps
    Researcher.research(task)   -> findings (may call a tool)
    Writer.write(task, findings)-> final answer

Each step's output is stored in `state` and passed to the next agent. That
hand-off of state between steps is the essence of agent orchestration
(LangGraph models it as a graph; CrewAI as a crew of agents with tasks).
"""

from typing import Callable

from app.agents import Planner, Researcher, Writer


class Orchestrator:
    def __init__(self, llm: Callable[[str], str]):
        self.planner = Planner(role="planner", llm=llm)
        self.researcher = Researcher(role="researcher", llm=llm)
        self.writer = Writer(role="writer", llm=llm)

    def run(self, task: str, verbose: bool = True) -> dict:
        state: dict = {"task": task}

        # Step 1: plan
        state["plan"] = self.planner.plan(task)
        if verbose:
            print("PLANNER produced a plan:")
            for i, step in enumerate(state["plan"], 1):
                print(f"   {i}. {step}")
            print()

        # Step 2: research (this is where a tool may be called)
        state["findings"] = self.researcher.research(task)
        if verbose:
            print(f"RESEARCHER findings:\n   {state['findings']}\n")

        # Step 3: write the final answer
        state["answer"] = self.writer.write(task, state["findings"])
        if verbose:
            print(f"WRITER final answer:\n   {state['answer']}\n")

        return state
