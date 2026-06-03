from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()
# Allow cross-origin requests from frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

OPENAI_API_KEY = open(".api_key").read().strip()

# Simple in-memory session store
sessions = {}

class SessionStartResponse(BaseModel):
    session_id: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

def create_rag_chain():
    # Build a retrieval-augmented chain (mock example)
    context_doc = "HCMUT University is a leading technical university in Vietnam."
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts([context_doc], embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0.0, api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant for HCMUT University. "
        "Based on the following context, answer the question:\n\n{context}\n\nQuestion: {question}"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever, llm

@app.post("/session/start", response_model=SessionStartResponse)
async def start_session():
    session_id = str(uuid4())
    chain, retriever, llm = create_rag_chain()
    sessions[session_id] = {"chain": chain, "retriever": retriever, "llm": llm}
    # Optional initial greeting
    initial_msg = "Assistant is connected. How can I help you today?"
    return {"session_id": session_id, "content": initial_msg}

@app.post("/chat", response_class=StreamingResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    user_msg = req.message
    if session_id not in sessions:
        return StreamingResponse(
            iter(["data: Invalid session_id\n\n"]), 
            media_type="text/event-stream"
        )
    # Retrieve LangChain components
    retriever = sessions[session_id]["retriever"]
    llm = sessions[session_id]["llm"]
    
    # Get relevant context (simple RAG)
    docs = retriever.get_relevant_documents(user_msg)
    context = docs[0].page_content if docs else ""
    full_prompt = f"You are an AI assistant for HCMUT University. " \
                  f"Give you some context, collect the information to answer the question:\n\n" \
                  f"Context:\n {context}\n\nQuestion: {user_msg} \n\nAnswer:"
    
    async def generate():
        full_answer = ""
        # Stream tokens from the model
        async for token in llm.astream(full_prompt):
            full_answer += token
            yield token
        print(f"Full answer: {full_answer}")
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
