"""
llm.py — Pluggable LLM wrapper.

Same idea as in the RAG project: one small place to swap model providers.

We ship a `fake_llm` so the whole multi-agent system runs offline with no API
key. The fake LLM is deterministic and just smart enough to demonstrate the
orchestration flow (planning, choosing a tool, writing a final answer). Swap in
`openai_llm` and everything downstream behaves the same way.
"""

import os
import re


def openai_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Real LLM call. Requires: pip install openai; export OPENAI_API_KEY=..."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def fake_llm(prompt: str) -> str:
    """
    A deterministic stand-in for a real model. It reads simple cues in the
    prompt and returns structured responses, so the agent loop is fully
    demonstrable offline. This is ONLY for the demo — real reasoning comes
    from a real model via openai_llm.
    """
    p = prompt.lower()

    # Always reason about the TASK only, never the tool-description examples
    # that also appear in the prompt. The task is the text after "Task:".
    task = prompt.split("Task:")[-1].strip() if "Task:" in prompt else prompt
    task_has_math = bool(re.search(r"[0-9].*[+\-*/].*[0-9]", task))

    # The PLANNER asks the model to break a task into steps.
    if "create a plan" in p:
        if task_has_math:
            return "1. Use the calculator to compute the expression.\n2. Write the final answer."
        return "1. Look up the relevant fact.\n2. Write the final answer."

    # The RESEARCHER decides which tool to call and with what argument.
    if "choose one tool" in p:
        if task_has_math:
            # pull the arithmetic expression out of the task
            expr = re.search(r"[-0-9+*/(). ]{3,}", task)
            return f"calculator | {expr.group().strip() if expr else '2+2'}"
        for term in ("python", "faiss", "rag", "transformer"):
            if term in task.lower():
                return f"knowledge_lookup | {term}"
        return "knowledge_lookup | python"

    # The WRITER composes a final answer from the gathered findings.
    if "write a clear final answer" in p:
        findings = prompt.split("Findings:")[-1].strip()
        return f"Based on the research: {findings}"

    return "OK."


# Swap this to openai_llm to use a real model everywhere.
default_llm = fake_llm
