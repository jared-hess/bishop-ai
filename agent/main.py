import os
import logging
import requests
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("bishop-agent")

# Env config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://llm:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# Optional: preload model on startup
def preload_model():
    try:
        logger.info(f"Preloading model '{LLM_MODEL}' from Ollama...")
        response = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": LLM_MODEL}, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            decoded = line.decode(errors="ignore")
            if "success" in decoded:
                logger.info(f"Model '{LLM_MODEL}' pulled successfully.")
                break
            logger.debug(decoded)
    except Exception as e:
        logger.error(f"Model preload failed: {e}")
        
app = FastAPI(title="Bishop")

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: str

llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL)

chain = RunnableLambda(lambda inputs: {
    "output": llm.invoke(inputs["input"]).content  # extract the string
})
# chain = chain.with_types(input_type=Input, output_type=Output)

add_routes(app, chain, path="/")


# Preload model on app start
@app.on_event("startup")
def startup_event():
    preload_model()
