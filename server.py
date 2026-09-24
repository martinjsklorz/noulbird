"""noulbird: game server and inference endpoint.

Loads a Laya classifier once at startup and serves both the game page and the
inference API from a single origin, so the browser needs no CORS handling or
endpoint configuration.

Endpoints:
    GET  /             The game page.
    GET  /v1/health    Readiness probe; reports whether the model is loaded.
    POST /v1/systemone Answers questions about a described state.

Usage:
    pip install -r requirements.txt
    python server.py [--port 8786] [--host 127.0.0.1] [--model ID]

Then open http://localhost:8786
"""

from __future__ import annotations

import argparse
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import laya_mlx as laya
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

DEFAULT_MODEL = "aac6fef/laya-mlx"
DEFAULT_PORT = 8786
DEFAULT_HOST = "127.0.0.1"

# Populated during the lifespan startup hook and shared across requests. The
# model is loaded once because initialisation costs seconds, while a single
# inference takes tens of milliseconds.
AGENT: Any = None

# Set from the command line before the app starts.
MODEL_ID = DEFAULT_MODEL
PORT = DEFAULT_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before serving and release it on shutdown."""
    global AGENT
    print(f"\n  Loading model: {MODEL_ID}")
    started = time.perf_counter()
    AGENT = laya.load(MODEL_ID, dtype="float16")
    print(f"  Loaded in {time.perf_counter() - started:.1f}s")
    print(f"  Ready at http://localhost:{PORT}\n")
    yield
    AGENT = None


app = FastAPI(
    title="noulbird",
    description="Serves the game and a small classifier that flies one bird.",
    lifespan=lifespan,
)


def normalize_criteria(question_type: str, criteria: Any) -> Any:
    """Accept noul criteria as either a mapping or a [true, false] pair."""
    if question_type == "noul" and isinstance(criteria, list):
        return {
            "true": criteria[0] if criteria else "",
            "false": criteria[1] if len(criteria) > 1 else "",
        }
    return criteria


def format_answer(question_type: str, raw: dict) -> dict:
    """Normalise one raw model answer into a stable response shape.

    Field names vary between question types and model versions, so each type is
    projected onto a predictable set of keys the client can rely on.
    """
    if question_type == "choice":
        probabilities = raw.get("probabilities", {})
        fallback = max(probabilities, key=probabilities.get) if probabilities else ""
        return {
            "choice": raw.get("choice", fallback),
            "confidence": raw.get("confidence", 0.0),
            "probabilities": probabilities,
        }

    if question_type == "score":
        return {
            "score": raw.get("score", 0.0),
            "confidence": raw.get("confidence", 0.0),
            "probabilities": raw.get("probabilities", {}),
        }

    if question_type == "noul":
        # A noul is the probability that the question is true. Some model
        # versions report it directly; others only expose class probabilities.
        value = raw.get("noul")
        if value is None:
            probabilities = raw.get("probabilities", {})
            value = probabilities.get("true", probabilities.get(True, 0.5))
        return {
            "noul": round(float(value), 4),
            "confidence": raw.get("confidence", 0.0),
        }

    return raw


@app.post("/v1/systemone")
async def system_one(payload: dict) -> dict:
    """Answer one or more questions about a described state.

    Expects a JSON body of:

        {
          "state": "The bird is below the safe gap. ...",
          "questions": {
            "below_gap": {
              "type": "noul",
              "instructions": "Is the bird below the safe gap?",
              "criteria": {"true": "...", "false": "..."}
            }
          }
        }

    Several questions may be asked in one call; they are evaluated together
    against the same state, which is cheaper than separate round trips.
    """
    if AGENT is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    state = payload.get("state", "")
    raw_questions = payload.get("questions", {})
    if not state:
        raise HTTPException(status_code=400, detail="'state' is required")
    if not raw_questions:
        raise HTTPException(status_code=400, detail="'questions' is required")

    question_types = {}
    questions = {}
    for name, question in raw_questions.items():
        question_type = question.get("type", "choice")
        question_types[name] = question_type
        questions[name] = {
            "type": question_type,
            "instructions": question.get("instructions", ""),
            "criteria": normalize_criteria(question_type, question.get("criteria", [])),
        }

    started = time.perf_counter()
    try:
        result = AGENT.predict(state, questions)
    except Exception as exc:  # surfaced to the client rather than a bare 500
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "answers": {
            name: format_answer(question_types.get(name, "choice"), raw)
            for name, raw in result.get("answers", {}).items()
        },
        "usage": result.get("usage", {}),
        "model": "laya-mlx",
        "latency_ms": round(elapsed_ms, 1),
    }


@app.get("/v1/health")
async def health() -> dict:
    """Report whether the model has finished loading."""
    return {
        "status": "ok" if AGENT is not None else "loading",
        "model": MODEL_ID,
        "backend": "laya-mlx",
    }


@app.get("/")
async def index() -> FileResponse:
    """Serve the game page."""
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


def main() -> None:
    """Parse arguments and run the server."""
    global MODEL_ID, PORT

    parser = argparse.ArgumentParser(
        description="noulbird - race a local Laya classifier at Flappy Bird."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Bind address. Defaults to localhost; use 0.0.0.0 to expose on the network.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier to load.")
    args = parser.parse_args()

    MODEL_ID = args.model
    PORT = args.port

    import uvicorn

    print("\n  noulbird")
    print(f"  Model: {MODEL_ID}")
    print(f"  http://localhost:{PORT}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
