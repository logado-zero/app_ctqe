from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4

from transformers import AutoModel, AutoTokenizer
from langchain_core.output_parsers import StrOutputParser
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm

from src.model.CTQE.model.applied_cqte import CTQEConversationApp

app = FastAPI()
# Allow cross-origin requests from frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

OPENAI_API_KEY = open(".api_key").read().strip()
CORPUS_PATH = Path("src") / "model" / "CTQE" / "data" / "raw" / "dataset" / "passage_corpus.json"
CHECKPOINT_PATH = Path("src") / "model" / "CTQE" / "model" / "checkpoints" / "merge_emb_mpnet_complete2.pt"
BERTOPIC_PATH = Path("src") / "model" / "CTQE" / "data" / "bertopic" / "BerTopic_corpus_mpnet_v2no_emb"
RETRIEVAL_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_HISTORY_LENGTH = 3
DEFAULT_RAG_MODE = "default"
TOP_K_RETRIEVAL = 4
CORPUS_DOC_LIMIT = 10000
EMBED_BATCH_SIZE = 64
LIMITED_CORPUS_CACHE = Path("src") / "model" / "CTQE" / "data" / "raw" / "dataset" / "passage_corpus_10k.json"
FAISS_CHECKPOINT_PATH = Path("src") / "model" / "CTQE" / "data" / "faiss" / f"mpnet_{CORPUS_DOC_LIMIT}"
SESSION_LOG_DIR = Path("src") / "model" / "logs"

# Simple in-memory session store
sessions = {}
_vectorstore = None
_ctqe_apps: dict[int, CTQEConversationApp] = {}
_retrieval_tokenizer = None
_retrieval_model = None


class SessionStartRequest(BaseModel):
    rag_mode: str = DEFAULT_RAG_MODE
    history_length: int = DEFAULT_HISTORY_LENGTH

class SessionStartResponse(BaseModel):
    session_id: str
    content: str
    rag_mode: str
    history_length: int
    started_at: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    rag_mode: str = DEFAULT_RAG_MODE
    history_length: int = DEFAULT_HISTORY_LENGTH


def _document_to_retrieval_record(doc, score: float | None, rank: int) -> dict:
    return {
        "rank": rank,
        "score": score,
        "score_type": "cosine",
        "passage": doc.page_content,
        "metadata": doc.metadata,
    }


def _session_log_stem(rag_mode: str, session_id: str) -> str:
    mode_name = _normalize_mode(rag_mode).replace(" ", "_")
    return f"{mode_name}_{session_id}"


def _append_session_log(session_id: str, rag_mode: str, log_entry: dict) -> Path:
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SESSION_LOG_DIR / _session_log_stem(rag_mode, session_id)
    entries = _load_session_log_entries(log_path)
    entries.append(log_entry)
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return log_path


def _load_session_log_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []

    with open(log_path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()

    if not content:
        return []

    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        entries: list[dict] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
        return entries

    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        return [loaded]
    return []


def _load_corpus_texts(corpus_path: Path, limit: int | None = None) -> list[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file at {corpus_path}")

    try:
        with open(corpus_path, "r", encoding="utf-8") as handle:
            corpus = json.load(handle)
    except json.JSONDecodeError:
        with open(corpus_path, "r", encoding="utf-8") as handle:
            texts: list[str] = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("ref_string") or item.get("text") or item.get("content") or ""
                else:
                    text = str(item)
                if text:
                    texts.append(text)
                    if limit is not None and len(texts) >= limit:
                        break
            return texts if limit is None else texts[:limit]

    texts: list[str] = []
    for item in corpus[:limit] if limit is not None else corpus:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(item.get("ref_string") or item.get("text") or item.get("content") or "")
        else:
            texts.append(str(item))
    return [text for text in texts if text]


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class _DeviceTransformerEmbeddings:
    def __init__(self, model_name: str):
        global _retrieval_tokenizer, _retrieval_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _retrieval_tokenizer is None:
            _retrieval_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _retrieval_model is None:
            _retrieval_model = AutoModel.from_pretrained(model_name).to(self.device)
            _retrieval_model.eval()
        self.tokenizer = _retrieval_tokenizer
        self.model = _retrieval_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Building FAISS", total=max(1, (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE)):
            batch_texts = texts[start:start + EMBED_BATCH_SIZE]
            token_ids = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(**token_ids)
            batch_embeddings = _mean_pooling(output, token_ids["attention_mask"])
            embeddings.extend(batch_embeddings.detach().cpu().tolist())
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
    
    def get_hidden_size(self) -> int:
        return self.model.config.hidden_size

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        embeddings = _DeviceTransformerEmbeddings(RETRIEVAL_MODEL_NAME)
        len_embed = embeddings.get_hidden_size()
        index = faiss.IndexFlatIP(len_embed)
        # Ensure limited corpus cache exists before building or rebuilding the FAISS index
        limited_cache = _ensure_limited_corpus_file()

        if FAISS_CHECKPOINT_PATH.exists():
            try:
                _vectorstore = FAISS.load_local(
                    str(FAISS_CHECKPOINT_PATH),
                    embeddings,
                    allow_dangerous_deserialization=True,
                    normalize_L2=True,
                    distance_strategy = DistanceStrategy.COSINE
                )
            except Exception as e:
                print(f"Failed to load FAISS index from checkpoint: {e}. Rebuilding index...")
                corpus_texts = _load_corpus_texts(CORPUS_PATH, limit=CORPUS_DOC_LIMIT)
                _vectorstore = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                    normalize_L2=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )
                _vectorstore.add_embeddings(zip(corpus_texts, embeddings.embed_documents(corpus_texts)))
                # _vectorstore = FAISS.from_texts(
                #     corpus_texts,
                #     embedding=embeddings,
                #     index=index,
                #     normalize_L2=True,
                #     distance_strategy=DistanceStrategy.COSINE,
                # )
                FAISS_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                _vectorstore.save_local(str(FAISS_CHECKPOINT_PATH))
        else:
            corpus_texts = _load_corpus_texts(CORPUS_PATH, limit=CORPUS_DOC_LIMIT)
            _vectorstore = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                    normalize_L2=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )
            _vectorstore.add_embeddings(zip(corpus_texts, embeddings.embed_documents(corpus_texts)))
            # _vectorstore = FAISS.from_texts(
            #     corpus_texts,
            #     embedding=embeddings,
            #     index=index,
            #     normalize_L2=True,
            #     distance_strategy=DistanceStrategy.COSINE,
            # )
            FAISS_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _vectorstore.save_local(str(FAISS_CHECKPOINT_PATH))
    return _vectorstore


@app.on_event("startup")
async def _warm_vectorstore() -> None:
    _get_vectorstore()


@app.on_event("startup")
async def _warm_ctqe() -> None:
    _get_ctqe_app(DEFAULT_HISTORY_LENGTH)


def _get_rewrite_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)


def _get_answer_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY, streaming=True)


def _get_ctqe_app(history_length: int) -> CTQEConversationApp:
    if history_length not in _ctqe_apps:
        _ctqe_apps[history_length] = CTQEConversationApp(
            model_name=RETRIEVAL_MODEL_NAME,
            checkpoint_path=CHECKPOINT_PATH,
            corpus_path=CORPUS_PATH,
            use_bertopic=True,
            bertopic_path=BERTOPIC_PATH,
            history_num=history_length,
        )
    return _ctqe_apps[history_length]


def _ensure_limited_corpus_file() -> Path:
    if LIMITED_CORPUS_CACHE.exists():
        return LIMITED_CORPUS_CACHE

    LIMITED_CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    limited_texts = _load_corpus_texts(CORPUS_PATH, limit=CORPUS_DOC_LIMIT)
    # Write as newline-delimited JSON (one document per line)
    with open(LIMITED_CORPUS_CACHE, "w", encoding="utf-8") as handle:
        for i,doc in enumerate(limited_texts):
            handle.write(json.dumps({"ref_id": i, "ref_string": doc}, ensure_ascii=False))
            handle.write("\n")
    return LIMITED_CORPUS_CACHE


def _normalize_mode(mode: str | None) -> str:
    normalized = (mode or DEFAULT_RAG_MODE).strip().lower()
    if normalized in {"rewrite query", "rewrite_query", "rewrite"}:
        return "rewrite query"
    if normalized == "ctqe":
        return "ctqe"
    if normalized in {
        "embedding history",
        "embedding_history",
        "embeddinghistory",
        "embed history",
        "embed_history",
        "embedhistory",
    }:
        return "embedding history"
    return "default"


def _session_history_pairs(session: dict) -> list[tuple[str, str]]:
    history = session.get("history", [])
    return [(item.get("question", ""), item.get("response", "")) for item in history if item.get("question")]


def _history_to_text(history_pairs: list[tuple[str, str]]) -> str:
    if not history_pairs:
        return ""
    return "\n\n".join([f"User: {question}\nAssistant: {response}" for question, response in history_pairs])


def _rewrite_query(user_msg: str, history_pairs: list[tuple[str, str]]) -> str:
    llm = _get_rewrite_llm()
    prompt = ChatPromptTemplate.from_template(
        "You rewrite a follow-up question into a standalone search query.\n"
        "Conversation history:\n{history}\n\n"
        "Follow-up question:\n{question}\n\n"
        "Return only the rewritten query."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"history": _history_to_text(history_pairs) or "None", "question": user_msg}).strip()

def _concat_history_to_query(user_msg: str, history_pairs: list[tuple[str, str]]) -> str:
    """Concatenate recent conversation history to the user message for retrieval, separated by [SEP] tokens."""

    if not history_pairs:
        return user_msg
    history_text = "[SEP]".join([f" {q} [SEP] {a}" for q, a in history_pairs])
    return f"{history_text} [SEP] {user_msg}"

def _search_with_scores(vectorstore, query: str | list[float], *, by_vector: bool, k: int):
    if by_vector:
        for method_name in ("similarity_search_with_score_by_vector", "similarity_search_by_vector_with_relevance_scores"):
            method = getattr(vectorstore, method_name, None)
            if callable(method):
                return method(query, k=k)
        docs = vectorstore.similarity_search_by_vector(query, k=k)
        return [(doc, None) for doc in docs]

    for method_name in ("similarity_search_with_score", "similarity_search_with_relevance_scores"):
        method = getattr(vectorstore, method_name, None)
        if callable(method):
            return method(query, k=k)
    docs = vectorstore.similarity_search(query, k=k)
    return [(doc, None) for doc in docs]


def _retrieve_documents(session: dict, user_msg: str, rag_mode: str, history_length: int):
    retrieval_started = perf_counter()
    vectorstore = _get_vectorstore()
    session_history = _session_history_pairs(session)[-history_length:]
    mode = _normalize_mode(rag_mode)

    if mode == "rewrite query":
        retrieval_query = _rewrite_query(user_msg, session_history)
        scored_docs = _search_with_scores(vectorstore, retrieval_query, by_vector=False, k=TOP_K_RETRIEVAL)
        return scored_docs, retrieval_query, mode, round((perf_counter() - retrieval_started) * 1000, 2)

    if mode == "ctqe":
        ctqe_app = _get_ctqe_app(history_length)
        transformed_embedding = ctqe_app.run(user_msg, session_history, normalize_output=True)
        scored_docs = _search_with_scores(
            vectorstore,
            transformed_embedding.detach().cpu().numpy().tolist(),
            by_vector=True,
            k=TOP_K_RETRIEVAL,
        )
        return scored_docs, user_msg, mode, round((perf_counter() - retrieval_started) * 1000, 2)
    
    if mode == "embedding history":
        concat_query = _concat_history_to_query(user_msg, session_history)
        scored_docs = _search_with_scores(vectorstore, concat_query, by_vector=False, k=TOP_K_RETRIEVAL)
        return scored_docs, concat_query, mode, round((perf_counter() - retrieval_started) * 1000, 2)

    scored_docs = _search_with_scores(vectorstore, user_msg, by_vector=False, k=TOP_K_RETRIEVAL)
    return scored_docs, user_msg, mode, round((perf_counter() - retrieval_started) * 1000, 2)


def _build_answer_prompt(
    context: str, question: str, history_pairs: list[tuple[str, str]] | None = None
) -> str:
    history_text = _history_to_text(history_pairs) if history_pairs else ""
    return (
        "You are an AI assistant for HCMUT University. Use the provided context to answer the question. "
        "If the context does not contain the answer, say you do not know.\n\n"
        f"Conversation history:\n{history_text or 'None'}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _log_session_retrieval(
    session: dict,
    *,
    session_id: str,
    user_msg: str,
    rag_mode: str,
    retrieval_query: str,
    scored_docs: list[tuple],
    retrieval_time_ms: float,
    started_at: datetime,
) -> None:
    retrieval_records = [
        _document_to_retrieval_record(doc, float(score) if score is not None else None, rank + 1)
        for rank, (doc, score) in enumerate(scored_docs)
    ]
    log_entry = {
        "session_id": session_id,
        "timestamp": started_at.isoformat(),
        "user_message": user_msg,
        "rag_mode": rag_mode,
        "retrieval_query": retrieval_query,
        "retrieval_time_ms": retrieval_time_ms,
        "results": retrieval_records,
    }
    session.setdefault("retrieval_logs", []).append(log_entry)
    session["last_retrieval_log"] = log_entry
    session["log_path"] = str(_append_session_log(session_id, rag_mode, log_entry))
    

@app.post("/session/start", response_model=SessionStartResponse)
async def start_session(req: SessionStartRequest):
    session_id = str(uuid4())
    rag_mode = _normalize_mode(req.rag_mode)
    history_length = max(1, req.history_length)
    sessions[session_id] = {
        "rag_mode": rag_mode,
        "history_length": history_length,
        "history": [],
        "retrieval_logs": [],
        "log_path": str(SESSION_LOG_DIR / _session_log_stem(rag_mode, session_id)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if rag_mode == "ctqe":
        sessions[session_id]["ctqe_app"] = _get_ctqe_app(history_length)
    initial_msg = "Assistant is connected. How can I help you today?"
    return {
        "session_id": session_id,
        "content": initial_msg,
        "rag_mode": rag_mode,
        "history_length": history_length,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/chat", response_class=StreamingResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    user_msg = req.message
    if session_id not in sessions:
        return StreamingResponse(
            iter(["data: Invalid session_id\n\n"]), 
            media_type="text/event-stream"
        )
    session = sessions[session_id]
    rag_mode = _normalize_mode(req.rag_mode or session.get("rag_mode"))
    history_length = max(1, req.history_length or session.get("history_length", DEFAULT_HISTORY_LENGTH))
    scored_docs, retrieval_query, retrieval_mode, retrieval_time_ms = _retrieve_documents(
        session,
        user_msg,
        rag_mode,
        history_length,
    )
    docs = [doc for doc, _score in scored_docs]
    context = "\n\n".join(doc.page_content for doc in docs[:TOP_K_RETRIEVAL])
    answer_llm = _get_answer_llm()
    # include recent session history in the answer prompt
    history_pairs = _session_history_pairs(session)[-history_length:]
    full_prompt = _build_answer_prompt(context, user_msg, history_pairs)
    print(f"Full prompt:\n{full_prompt}\n{'-'*50}")

    started_at = datetime.now(timezone.utc)
    request_started = perf_counter()
    
    async def generate():
        full_answer = ""
        try:
            async for token in answer_llm.astream(full_prompt):
                chunk = token.content if hasattr(token, "content") else str(token)
                if not chunk:
                    continue
                full_answer += chunk
                yield chunk
        finally:
            _log_session_retrieval(
                session,
                session_id=session_id,
                user_msg=user_msg,
                rag_mode=retrieval_mode,
                retrieval_query=retrieval_query,
                scored_docs=scored_docs,
                retrieval_time_ms=retrieval_time_ms,
                started_at=started_at,
            )
            session.setdefault("history", []).append({"question": user_msg, "response": full_answer})
            session["rag_mode"] = rag_mode
            session["history_length"] = history_length
            session["last_retrieval_query"] = retrieval_query
            session["last_retrieval_mode"] = retrieval_mode
            session["last_retrieval_time_ms"] = retrieval_time_ms
            session["last_latency_ms"] = round((perf_counter() - request_started) * 1000, 2)
            session["last_response_started_at"] = started_at.isoformat()
            print(f"Full answer: {full_answer}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Server-Time": started_at.isoformat(),
            "X-RAG-Mode": rag_mode,
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
