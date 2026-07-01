import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from ingest import run_ingest
from rag import chat
from database import count_chunks

load_dotenv()

app = FastAPI(title="Kenya Civic Info Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []


@app.get("/health")
async def health():
    try:
        return {"status": "ok", "chunks_in_db": count_chunks()}
    except Exception as e:
        return {"status": "degraded", "detail": str(e), "chunks_in_db": 0}


@app.post("/ingest")
async def ingest(x_api_key: str = Header(...)):
    if x_api_key != os.environ["INGEST_API_KEY"]:
        raise HTTPException(status_code=401, detail="Invalid API key")
    count = await run_ingest()
    return {"status": "ok", "chunks_ingested": count}


@app.post("/chat")
async def chat_endpoint(body: ChatRequest):
    result = await chat(body.message, body.conversation_history)
    return result
