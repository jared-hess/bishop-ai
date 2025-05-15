# Bishop

**Bishop** is a modular, self-hosted AI assistant built around voice control, memory retrieval, display output, and thermal printing. It runs on local LLMs and modular Flask-based services, orchestrated by LangChain.


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
