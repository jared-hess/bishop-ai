# Bishop

**Bishop** is a modular, self-hosted AI assistant built around voice control, memory retrieval, display output, and thermal printing. It runs on local LLMs and modular Flask-based services, orchestrated by LangChain.

> Inspired by sci-fi assistants like J.A.R.V.I.S. and Bishop from *Alien*, but built for real-world daily utility and full user control.

---

## 🧱 Architecture

- **LangChain Agent**: Orchestrates tools, memory, reasoning  
- **LLM Inference**: Local (Ollama, llama.cpp, etc.)  
- **Display Server**: Multi-zone HTML rendering  
- **Voice Server**: Text-to-speech (XTTS or Piper)  
- **Printer Server** *(optional)*: Thermal printer endpoint  
- **Retriever**: FAISS/Chroma + Obsidian vault parsing  

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/bishop.git
cd bishop
cp .env.template .env
docker-compose up --build
```

---

## 🧹 Components

| Folder       | Description                          |
|--------------|--------------------------------------|
| `agent/`     | LangChain logic + tool definitions   |
| `llm/`       | Ollama container for model inference |
| `retriever/` | Vault loader + vector DB             |
| `display/`   | HTML zone rendering server           |
| `printer/`   | Thermal printer server (optional)    |
| `voice/`     | TTS + optional STT endpoints         |

---

## 🛠️ Environment Variables

See `.env.template` for required config.

---

## 📆 Model Support

- Optimized for 7B quantized models via Ollama (`mistral`, `deepseek`, etc.)
- Whisper + Piper recommended for offline speech I/O

---

## 🧠 Status

This is an early-stage prototype. Features are modular and isolated in Docker containers for easy development, replacement, or scaling.

PRs and suggestions welcome!
