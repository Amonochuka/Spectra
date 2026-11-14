"""Spectra Unified FastAPI Mission Science Engine.

Specification Anchor: SPECIFICATION.md §4.2, §10.4.8
Unified FastAPI service running on port 7860 orchestrating science pipelines,
zero-auth observer sessions, and binary tensor streaming.
"""

import asyncio
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from science.similarity import similarity_search

# Setup uvloop event loop policy for high performance on Unix
if sys.platform != "win32":
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

app = FastAPI(
    title="Spectra Mission Science Engine",
    description="Unified FastAPI Backend for SPHEREx Time-Domain & Hyperspectral Exploration",
    version="1.0.0",
)

# CORS Middleware (SPECIFICATION.md §10.4.8)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Origin",
        "Content-Type",
        "Accept",
        "Authorization",
        "X-Observer-ID",
        "X-Observer-Key",
    ],
    expose_headers=["X-Observer-ID", "X-Cache"],
    max_age=600,
)


@app.get("/api/health/live")
async def health_live() -> JSONResponse:
    """Process liveness check for container supervisor.

    Specification Anchor: SPECIFICATION.md §4.2, §6.1
    """
    return JSONResponse(status_code=200, content={"status": "live"})


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Dual-action keep-alive check.

    Specification Anchor: SPECIFICATION.md §4.2, §10.4.8
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "spectra-backend",
            "version": "1.0.0",
        },
    )


@app.post("/api/subjects/similar")
async def subject_similarity(payload: dict) -> dict:
    """Compute in-memory cosine similarity matches against the pre-loaded spectral library."""
    try:
        query = payload.get("query")
        if query is None:
            for key in ("spectrum", "flux", "values", "vector", "spectrum_array"):
                if key in payload:
                    query = payload[key]
                    break
        if query is None:
            raise KeyError("No spectral vector was provided.")
        result = similarity_search(query, k=int(payload.get("k", 5)))
        return {
            "matches": result["matches"],
            "results": result["matches"],
            "top_matches": result["matches"],
            "status": "ok",
            "count": len(result["matches"]),
        }
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
