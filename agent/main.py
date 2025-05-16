import os
import json
import logging
import requests

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama

# ——— Logging setup ————————————————————————————————————————————————————————
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("chatbot-proxy")

# ——— Configuration ——————————————————————————————————————————————————————
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://llm:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral")

# ——— Preload model on startup —————————————————————————————————————————————
def preload_model():
    try:
        logger.info(f"[startup] Pulling model '{DEFAULT_MODEL}' from Ollama…")
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": DEFAULT_MODEL},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            text = line.decode(errors="ignore")
            if "success" in text:
                logger.info(f"[startup] Model '{DEFAULT_MODEL}' ready.")
                break
            logger.debug(f"[startup] {text}")
    except Exception as e:
        logger.error(f"[startup] Model preload failed: {e}")

# ——— LLM client ————————————————————————————————————————————————————————
llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_URL)

# ——— Pydantic request schemas —————————————————————————————————————————————
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = True

# ——— FastAPI app ————————————————————————————————————————————————————————
app = FastAPI(title="Bishop Chatbot-UI Proxy")

@app.on_event("startup")
def on_startup():
    preload_model()

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": DEFAULT_MODEL, "object": "model"}
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model = request.model or DEFAULT_MODEL
    user_msg = request.messages[-1].content

    if request.stream:
        def event_stream():
            try:
                reply = llm.invoke(user_msg).content
                chunk = {"choices":[{"delta":{"content":reply}}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("LLM stream error: %s", e)
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        try:
            reply = llm.invoke(user_msg).content
        except Exception as e:
            logger.error("LLM error: %s", e)
            raise
        return JSONResponse({
            "choices": [{
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop","index": 0,
            }],
            "usage": {},
        })
