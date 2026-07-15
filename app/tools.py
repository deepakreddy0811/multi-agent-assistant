"""
tools.py — Tools the agents can call.

A "tool" is just a function with a name and a description. Giving agents tools
is what lets them DO things (look up data, calculate) instead of only talking.
This is the heart of "tool calling" / "function calling" on your resume.

We keep a registry (a dict) mapping tool name -> function, so the orchestrator
can look a tool up by name and run it. That's exactly how frameworks like
LangChain and CrewAI expose tools under the hood.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def run(self, arg: str) -> str:
        return self.func(arg)


# ----- individual tool implementations -------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 2)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only basic arithmetic is allowed."
    try:
        # eval is safe here because we whitelisted the characters above.
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# A tiny built-in "knowledge base" so the demo works fully offline.
# In a real system this would be a web search tool or a RAG retriever
# (you can literally plug in Project 1 here — a great thing to mention).
_KB = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "faiss": "FAISS is a library by Meta for fast vector similarity search.",
    "rag": "RAG (Retrieval-Augmented Generation) grounds LLM answers in retrieved documents.",
    "transformer": "The Transformer is a neural network architecture introduced in 2017 in 'Attention Is All You Need'.",
}


def knowledge_lookup(query: str) -> str:
    """Look up a term in the local knowledge base."""
    key = query.strip().lower()
    for term, fact in _KB.items():
        if term in key:
            return fact
    return f"No entry found for '{query}'."


def word_count(text: str) -> str:
    """Count the words in a piece of text."""
    return str(len(text.split()))


# ----- the registry --------------------------------------------------------

TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        "calculator",
        "Evaluates basic arithmetic, e.g. '12 * (3 + 4)'.",
        calculator,
    ),
    "knowledge_lookup": Tool(
        "knowledge_lookup",
        "Looks up a fact about a term (python, faiss, rag, transformer).",
        knowledge_lookup,
    ),
    "word_count": Tool(
        "word_count",
        "Counts the number of words in the given text.",
        word_count,
    ),
}


def tool_descriptions() -> str:
    """A formatted list of tools, used to tell an agent what it can use."""
    return "\n".join(f"- {t.name}: {t.description}" for t in TOOLS.values())
