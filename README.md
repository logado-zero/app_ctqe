# CTQE Chat

A conversational RAG chatbot for comparing conversational retrieval strategies. You ask follow-up questions in a chat UI. The backend retrieves passages from the [CORAL](https://huggingface.co/datasets/ariya2357/CORAL) corpus and streams an answer from OpenAI `gpt-3.5-turbo`.

Each chat session uses one of four retrieval modes:

| Mode | How the retrieval query is built |
|------|----------------------------------|
| `default` | The current user message only |
| `rewrite query` | An LLM rewrites the follow-up into a standalone query |
| `embedding history` | Recent turns are joined to the message with `[SEP]` and embedded together |
| `ctqe` | The [CTQE](https://github.com/logado-zero/CTQE) model turns the message and its history into a query embedding |

**Stack:** React 19 + Vite + MUI (frontend), FastAPI (backend), `sentence-transformers/all-mpnet-base-v2` with a FAISS cosine index over the first 10k CORAL passages, and LangChain + OpenAI.

## Quickstart

Complete [Setup & Installation](#setup--installation) first, then run the backend and frontend in two terminals:

```bash
# Terminal 1: API at http://127.0.0.1:8000
python server.py
```

```bash
# Terminal 2: UI at http://localhost:5173
npm run dev
```

Open http://localhost:5173, choose a retrieval mode and a history length, and start chatting.

Each session's retrieval results are logged to `src/model/logs/`.

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI API key
- Optional: a CUDA GPU (embeddings fall back to CPU when none is available)

### 1. Clone the repo and the CTQE model

```bash
git clone <this-repo-url> app_ctqe
cd app_ctqe
git clone https://github.com/logado-zero/CTQE.git src/model/CTQE
```

### 2. Install the Python dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the CORAL dataset

```bash
cd src/model/CTQE
pip install huggingface_hub
python data/download_CORAL.py
cd ../../..
```

This creates `src/model/CTQE/data/raw/dataset/passage_corpus.json`.

### 4. Add the CTQE model files

The trained model files are not included in either repository. Place them at these paths:

```
src/model/CTQE/model/checkpoints/merge_emb_mpnet_complete2.pt
src/model/CTQE/data/bertopic/BerTopic_corpus_mpnet_v2no_emb
```

### 5. Add your OpenAI API key

Create a file named `.api_key` in the project root that contains only your key:

```bash
echo "sk-..." > .api_key
```

### 6. Install the frontend dependencies

```bash
npm install
```

> **Note:** On the first start, the server embeds 10,000 passages to build the FAISS index, which can take several minutes on a CPU. The index is then cached at `src/model/CTQE/data/faiss/mpnet_10000`, so later starts are fast.

## Project structure

```
server.py            FastAPI backend (/session/start, /chat)
src/App.jsx          Chat UI
src/model/CTQE/      CTQE model, data, and checkpoints
src/model/logs/      Per-session retrieval logs
```
