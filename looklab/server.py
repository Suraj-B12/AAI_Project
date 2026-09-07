"""FastAPI over the compiled graph.

Two things here are deliberate and load-bearing:

**Endpoints are plain ``def``, not ``async def``.** ``SqliteSaver`` implements
only the synchronous checkpointer protocol, so ``await graph.ainvoke(...)``
raises ``NotImplementedError`` on the very first request. A plain ``def``
endpoint is run by Starlette in a threadpool, which is also exactly why the
connection is opened with ``check_same_thread=False``.

**Writes to one thread are serialised by a per-thread lock.** Firing eight
concurrent invokes at a single ``thread_id`` silently collapses them to one
surviving update -- verified, no exception raised, the other seven turns are
just gone. A double-clicked send button is enough to trigger it, so the fix
belongs on the server rather than in the UI.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import context as ctx
from . import kb as kb_module
from .graph import BRANCH_MAP, build_graph, mermaid, new_turn_input
from .llm import get_narrator
from .persistence import make_checkpointer, make_store

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MAX_IMAGE_BYTES = 12 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Seed two contrasting demo threads if the database is empty.

    Free hosts have ephemeral disks and sleep after inactivity, so a grader
    opening the link would otherwise land on a blank app -- and blank panes
    demonstrate none of T2, T3 or T4. Set LOOKLAB_SEED=0 to disable.

    A lifespan handler rather than ``@app.on_event("startup")``, which FastAPI
    deprecated.
    """
    if (os.getenv("LOOKLAB_SEED") or "1").lower() not in {"0", "false", "no", "off"}:
        try:
            from .seed import seed

            seed(GRAPH)
        except Exception:  # never let seeding stop the server from starting
            pass
    yield


app = FastAPI(
    title="LookLab",
    version="1.0",
    description="Colour-grading assistant",
    lifespan=lifespan,
)

# Built once at import. Tests never touch this -- they call build_graph() with
# their own tmp_path-backed checkpointer and store.
CHECKPOINTER = make_checkpointer()
STORE = make_store()
GRAPH = build_graph(CHECKPOINTER, STORE)

_thread_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _lock_for(thread_id: str) -> threading.Lock:
    with _locks_guard:
        return _thread_locks[thread_id]


def _decode_image(b64: str | None, label: str) -> bytes | None:
    if not b64:
        return None
    payload = b64.split(",", 1)[1] if b64.startswith("data:") else b64
    # Strip the whitespace a wrapped data URL may carry, then decode with
    # validate=True. With validate=False, base64 silently DISCARDS characters
    # outside the alphabet, so "!!!!" decodes to b"" and a corrupt upload is
    # indistinguishable from no upload at all -- a 200 with a wrong answer
    # instead of a 400.
    payload = "".join(payload.split())
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, f"{label} image is not valid base64: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{label} image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)}MB")
    if not raw:
        return None
    return raw


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field("suraj", min_length=1, max_length=64)
    thread_id: str = Field("default", min_length=1, max_length=64)
    budget: int = Field(ctx.DEFAULT_BUDGET, ge=ctx.MIN_BUDGET, le=100_000)
    reference_b64: str | None = None
    current_b64: str | None = None


def _state_payload(thread_id: str) -> dict[str, Any]:
    """Read a thread's state. An unknown thread must not 500 or blank the UI."""
    snapshot = GRAPH.get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values if snapshot else {}
    if not values:
        return {
            "thread_id": thread_id,
            "exists": False,
            "messages": [],
            "recipe": [],
            "images": {},
            "intent": None,
            "branch": None,
            "telemetry": {},
            "matches": [],
        }
    images = {
        role: sig
        for role, sig in (values.get("images") or {}).items()
        if not role.startswith("_") and isinstance(sig, dict)
    }
    return {
        "thread_id": thread_id,
        "exists": True,
        "messages": [
            {"role": _role_of(m), "content": _content_of(m)}
            for m in (values.get("messages") or [])
        ],
        "recipe": values.get("recipe") or [],
        "images": images,
        "intent": values.get("intent"),
        "branch": values.get("branch"),
        "telemetry": values.get("telemetry") or {},
        "matches": values.get("matches") or [],
        "user_id": values.get("user_id"),
    }


def _role_of(message: Any) -> str:
    kind = getattr(message, "type", "") or ""
    return {"human": "user", "ai": "assistant", "system": "system"}.get(kind, kind or "assistant")


def _content_of(message: Any) -> str:
    from .context import _text_of

    return _text_of(message)


@app.get("/health")
def health() -> dict[str, Any]:
    narrator = get_narrator()
    meta = kb_module.load_kb().get("meta", {})
    return {
        "ok": True,
        "narrator": getattr(narrator, "name", type(narrator).__name__),
        "deterministic": bool(getattr(narrator, "is_deterministic", False)),
        "token_backend": ctx.TOKEN_BACKEND,
        "looks": len(kb_module.looks()),
        "checkpointer": type(CHECKPOINTER).__name__,
        "store": type(STORE).__name__,
        "plates": {
            "source": meta.get("source"),
            "kb": len(meta.get("kb_plates", []) or meta.get("plates", [])),
            "heldout": len(meta.get("heldout_plates", [])),
        },
        "knowledge_terms": len(_knowledge().get("terms", [])),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    """Narrator status, including per-key Gemini health when configured.

    Keys are never returned -- only their last eight characters, which is
    enough to tell two apart in a log and not enough to use one.
    """
    narrator = get_narrator()
    payload: dict[str, Any] = {
        "narrator": getattr(narrator, "name", type(narrator).__name__),
        "deterministic": bool(getattr(narrator, "is_deterministic", False)),
        "rewrites": getattr(narrator, "rewrites", None),
        "fallbacks": getattr(narrator, "fallbacks", None),
    }
    try:
        from .gemini import get_pool

        pool = get_pool()
        if pool is not None:
            payload["gemini"] = {
                "model": pool.active_model,
                "keys_total": len(pool),
                "keys_usable": pool.usable,
                "health": pool.health(),
            }
    except Exception as exc:
        payload["gemini_error"] = f"{type(exc).__name__}: {exc}"
    return payload


@lru_cache(maxsize=1)
def _knowledge() -> dict[str, Any]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.json")
    if not os.path.exists(path):
        return {"meta": {}, "terms": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"meta": {}, "terms": []}


@app.get("/knowledge")
def knowledge(q: str | None = None) -> dict[str, Any]:
    """Cited colour-science glossary. ``?q=`` filters by substring."""
    data = _knowledge()
    terms = data.get("terms", [])
    if q:
        needle = q.lower()
        terms = [
            t
            for t in terms
            if needle in t["term"].lower()
            or needle in t.get("summary", "").lower()
            or needle in t.get("why_it_matters", "").lower()
        ]
    return {"meta": data.get("meta", {}), "terms": terms}


@app.get("/plates")
def plates() -> dict[str, Any]:
    """Base-plate provenance and attribution, so licensing is inspectable."""
    from .plates import manifest, provenance

    return {"provenance": provenance(), "plates": manifest().get("plates", [])}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """One conversational turn. Sync on purpose -- see the module docstring."""
    reference = _decode_image(req.reference_b64, "reference")
    current = _decode_image(req.current_b64, "current")

    payload = new_turn_input(
        req.message,
        user_id=req.user_id,
        budget=req.budget,
        reference_bytes=reference,
        current_bytes=current,
    )
    config = {"configurable": {"thread_id": req.thread_id}}

    with _lock_for(req.thread_id):
        try:
            result = GRAPH.invoke(payload, config)
        except Exception as exc:  # never let one bad turn take the server down
            raise HTTPException(500, f"graph invocation failed: {type(exc).__name__}: {exc}")

    messages = result.get("messages") or []
    reply = _content_of(messages[-1]) if messages else ""
    return {
        "reply": reply,
        "intent": result.get("intent"),
        "branch": result.get("branch"),
        "branch_node": BRANCH_MAP.get(result.get("intent") or "chat"),
        "recipe": result.get("recipe") or [],
        "telemetry": result.get("telemetry") or {},
        "profile": result.get("profile") or {},
        "matches": result.get("matches") or [],
        "thread_id": req.thread_id,
        "user_id": req.user_id,
    }


@app.get("/threads")
def threads(user_id: str = "suraj") -> dict[str, Any]:
    """Every thread the checkpointer knows about, newest first."""
    seen: dict[str, dict[str, Any]] = {}
    try:
        for snapshot in GRAPH.get_state_history({"configurable": {}}, limit=None) or []:
            tid = (snapshot.config or {}).get("configurable", {}).get("thread_id")
            if tid and tid not in seen:
                seen[tid] = {"thread_id": tid}
    except Exception:
        # get_state_history needs a thread_id on some versions; fall back to
        # listing the checkpointer directly.
        try:
            for item in CHECKPOINTER.list(None):
                tid = (item.config or {}).get("configurable", {}).get("thread_id")
                if not tid or tid in seen:
                    continue
                values = item.checkpoint.get("channel_values", {}) if item.checkpoint else {}
                if user_id and values.get("user_id") and values.get("user_id") != user_id:
                    continue
                seen[tid] = {
                    "thread_id": tid,
                    "user_id": values.get("user_id"),
                    "messages": len(values.get("messages") or []),
                    "recipe": len(values.get("recipe") or []),
                }
        except Exception:
            pass
    return {"threads": list(seen.values())}


@app.get("/state/{thread_id}")
def state(thread_id: str) -> dict[str, Any]:
    return _state_payload(thread_id)


@app.get("/profile/{user_id}")
def profile(user_id: str) -> dict[str, Any]:
    from .memory import describe_profile, load_profile

    data = load_profile(STORE, user_id)
    return {"user_id": user_id, "profile": data, "summary": describe_profile(data)}


@app.get("/graph", response_class=PlainTextResponse)
def graph_text() -> str:
    """Mermaid source, generated by the compiled graph object itself."""
    return mermaid(GRAPH)


@app.get("/looks")
def looks() -> dict[str, Any]:
    kb = kb_module.load_kb()
    return {
        "meta": kb.get("meta", {}),
        "looks": [
            {
                "id": lk["id"],
                "label": lk["label"],
                "notes": lk.get("notes", ""),
                "sliders": lk.get("sliders", {}),
            }
            for lk in kb.get("looks", [])
        ],
    }


@app.get("/")
def index() -> Any:
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        return PlainTextResponse("front end not built; API is at /docs", status_code=200)
    return FileResponse(path)
