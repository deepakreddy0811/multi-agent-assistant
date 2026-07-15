"""
agents.py — The specialized agents.

Each agent has:
  - a ROLE (what it's good at)
  - a single method that takes some input and returns output by prompting the LLM

This is the core idea behind multi-agent systems: instead of one giant prompt
doing everything, you split the work across focused agents. Each one has a
narrow job, which makes behaviour more reliable and easier to debug — the same
philosophy CrewAI and LangGraph are built on.

The three agents here form a classic pipeline:
    Planner   -> breaks the task into steps
    Researcher-> gathers information, using tools when needed
    Writer    -> turns the findings into a final answer
"""

from dataclasses import dataclass
from typing import Callable

from app.tools import TOOLS, tool_descriptions


@dataclass
class Agent:
    role: str
    llm: Callable[[str], str]


class Planner(Agent):
    """Breaks a user task into a short, ordered list of steps."""

    def plan(self, task: str) -> list[str]:
        prompt = (
            f"You are a planner. Create a plan (numbered steps) to answer "
            f"this task.\n\nTask: {task}"
        )
        raw = self.llm(prompt)
        # parse "1. ... \n 2. ..." into a clean list of step strings
        steps = [
            line.split(".", 1)[1].strip()
            for line in raw.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        return steps or [raw.strip()]


class Researcher(Agent):
    """
    Gathers information. Decides whether to call a tool, and if so which one.
    This is the ReAct-style 'reason then act' step that powers tool-using agents.
    """

    def research(self, task: str) -> str:
        prompt = (
            "You are a researcher with access to these tools:\n"
            f"{tool_descriptions()}\n\n"
            "Choose ONE tool to help answer the task. Respond in the format:\n"
            "tool_name | argument\n\n"
            f"Task: {task}"
        )
        decision = self.llm(prompt)

        # parse "tool_name | argument"
        if "|" in decision:
            name, _, arg = decision.partition("|")
            name, arg = name.strip(), arg.strip()
            tool = TOOLS.get(name)
            if tool:
                result = tool.run(arg)
                return f"Used tool '{name}' with '{arg}' -> {result}"
        return f"No tool used. Raw response: {decision}"


class Writer(Agent):
    """Synthesizes the findings into a clear final answer for the user."""

    def write(self, task: str, findings: str) -> str:
        prompt = (
            "You are a writer. Write a clear final answer to the user's task "
            "using the findings.\n\n"
            f"Task: {task}\n\nFindings: {findings}"
        )
        return self.llm(prompt)
