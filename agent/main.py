"""
main.py ― FastAPI + LangChain proxy compatible with Chatbot-UI
────────────────────────────────────────────────────────────────
▪ Exposes OpenAI-style endpoints:
    •  GET  /v1/models
    •  POST /v1/chat/completions      (stream & non-stream)

▪ Passes full message context on every call.
▪ Uses LangChain’s ChatOllama backend (easily swappable later).
▪ Ready for future memory / tool calling (see TODOs below).
"""

import os
import json
import time
import uuid
import logging
from typing import List, Dict, Generator

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://llm:11434")
DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "mistral")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("bishop-proxy")

# ── LangChain imports (community wrapper keeps Pydantic v2 happy) ─
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_URL)

# ── FastAPI setup ─────────────────────────────────────────────
app = FastAPI(title="Bishop Chat Proxy")

# Allow browser UIs from any origin (tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model preload on startup ──────────────────────────────────
@app.on_event("startup")
def preload_model() -> None:
    try:
        logger.info("Pre-pulling model '%s' from Ollama …", DEFAULT_MODEL)
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": DEFAULT_MODEL},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        for ln in resp.iter_lines():
            if b"success" in ln:
                logger.info("Model ready.")
                break
    except Exception as exc:
        logger.warning("Model preload failed: %s", exc)

# ── Pydantic request/response schemas ────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., min_length=1)

class ChatCompletionRequest(BaseModel):
    model: str | dict | None = None
    messages: List[ChatMessage]
    stream: bool = True
    
    class Config:
        extra = "ignore"  # ignore extra fields in request
        
# ---------------------------------------------------------------------
# Helper to resolve UI-supplied model into a string we actually use
# ---------------------------------------------------------------------
def resolve_model(model_field) -> str:
    """
    • If UI sends a string, return it
    • If UI sends an object, use its 'id'
    • If it's anything else, fall back to DEFAULT_MODEL
    """
    # if isinstance(model_field, str) and model_field.strip():
    #     return model_field
    # if isinstance(model_field, dict) and "id" in model_field:
    #     return str(model_field["id"])
    return DEFAULT_MODEL

# ── Helper: convert request.messages → LangChain messages list ─
ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}
def to_lc_messages(msgs: List[ChatMessage]):
    lc: List = []
    for m in msgs:
        cls = ROLE_MAP.get(m.role, HumanMessage)
        lc.append(cls(content=m.content))
    return lc

# ── GET /v1/models  (minimal) ─────────────────────────────────
@app.get("/v1/models")
async def list_models():
    """
    OpenAI spec:  https://platform.openai.com/docs/api-reference/models/list
    Chatbot-UI expects at least: id, object, created, owned_by
    """
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,            # e.g. "mistral"
                "object": "model",
                "created": int(time.time()),    # any int is fine
                "owned_by": "local",
                "permission": [],               # optional but harmless
            }
        ],
    }


# ── POST /v1/chat/completions ─────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint:
      • streams Server-Sent Events if req.stream=True
      • otherwise returns one-shot JSON
    Full context is forwarded every call.
    """
    lc_msgs = to_lc_messages(req.messages)
    model_name = resolve_model(req.model)
    created = int(time.time())
    completion_id = str(uuid.uuid4())

    # ── STREAMING mode ────────────────────────────────────────
    if req.stream:
        def sse_stream() -> Generator[str, None, None]:
            for chunk in llm.stream(lc_msgs, model=model_name):
                # chunk is ChatGenerationChunk → chunk.message.content / .text
                delta_text = getattr(chunk, "content", "") or chunk.text
                if delta_text:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "delta": {"content": delta_text},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            # end‐of‐stream marker
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    # ── NON-STREAMING mode ────────────────────────────────────
    result = llm.invoke(lc_msgs, model=model_name)  # returns AIMessage
    answer = result.content if hasattr(result, "content") else str(result)

    response_body: Dict = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
            "index": 0,
        }],
        "usage": {},   # Ollama's community wrapper doesn't expose usage yet
    }
    return JSONResponse(response_body)

# ──────────────────────────────────────────────────────────────
# TODO (future):
#   • swap llm for an AgentExecutor with tools & memory
#   • plug in vector store retrievers (FAISS, Chroma, etc.)
#   • add auth & Supabase persistence if desired
