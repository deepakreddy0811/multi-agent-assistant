"""
main.py — FastAPI server exposing the multi-agent assistant.

Run:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs and POST a task to /run.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.llm import default_llm
from app.orchestrator import Orchestrator

app = FastAPI(title="Multi-Agent Assistant")
orchestrator = Orchestrator(llm=default_llm)


class Task(BaseModel):
    task: str


@app.post("/run")
async def run(payload: Task):
    """Run the planner -> researcher -> writer pipeline on a task."""
    state = orchestrator.run(payload.task, verbose=False)
    return {
        "task": state["task"],
        "plan": state["plan"],
        "findings": state["findings"],
        "answer": state["answer"],
    }


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}
