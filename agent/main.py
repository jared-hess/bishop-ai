from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bishop-agent")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://llm:11434")
MODEL = os.getenv("LLM_MODEL", "mistral")

class AskRequest(BaseModel):
    query: str

@app.on_event("startup")
def preload_model():
    logger.info(f"Preloading model '{MODEL}' from Ollama...")
    try:
        response = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": MODEL}, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            decoded = line.decode(errors="ignore")
            if "success" in decoded:
                logger.info(f"Model '{MODEL}' pulled successfully.")
                break
            logger.debug(decoded)
    except requests.RequestException as e:
        logger.error(f"Failed to preload model '{MODEL}': {e}")

@app.post("/ask")
def ask(request: AskRequest):
    logger.info(f"Received query: {request.query}")
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": request.query, "stream": False},
        )
        logger.debug(f"Ollama response status: {response.status_code}")
        logger.debug(f"Ollama response body: {response.text}")
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "[No response received]")
        logger.info("Successfully generated response from LLM.")
        return {"answer": answer}
    except requests.RequestException as e:
        logger.error(f"Failed to query Ollama: {e}")
        return {"error": "Failed to generate response from LLM."}
