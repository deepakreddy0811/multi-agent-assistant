"""
demo.py — Run the multi-agent assistant from the command line. No API key needed.

It uses the deterministic fake LLM (see app/llm.py) so you can watch the full
planner -> researcher -> writer flow, including a real tool call, completely
offline. Swap default_llm to openai_llm in app/llm.py for real reasoning.

Run:
    python demo.py
"""

from app.llm import default_llm
from app.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator(llm=default_llm)

    tasks = [
        "What is 24 * (8 + 5)?",          # routes to the calculator tool
        "Tell me what FAISS is.",          # routes to the knowledge_lookup tool
    ]

    for task in tasks:
        print("=" * 70)
        print(f"USER TASK: {task}\n")
        orchestrator.run(task, verbose=True)


if __name__ == "__main__":
    main()
