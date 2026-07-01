import os
import re
from openai import OpenAI
from anthropic import Anthropic
from database import similarity_search
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a civic education assistant for Kenya. Answer questions about Kenyan governance, "
    "the Constitution, voting rights, and county government using ONLY the provided context. "
    "Always cite which document your answer comes from (e.g., 'According to the Kenya Constitution 2010, Article 38...' "
    "or 'According to IEBC voter registration guidelines...'). "
    "If the answer is not in the context, say so clearly and suggest the user contact the relevant government body. "
    "Keep answers concise, clear, and accessible to a young Kenyan with a secondary school education."
)

ARTICLE_RE = re.compile(r"Article\s+\d+[A-Z]?", re.IGNORECASE)


def _build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )


async def chat(message: str, conversation_history: list[dict]) -> dict:
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    query_embedding = openai.embeddings.create(
        model="text-embedding-3-small", input=message
    ).data[0].embedding

    chunks = similarity_search(query_embedding, top_k=5)
    context = _build_context(chunks)
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    messages = [
        *conversation_history,
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"},
    ]

    response = anthropic.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text
    articles_cited = list(dict.fromkeys(ARTICLE_RE.findall(answer)))

    return {"answer": answer, "sources": sources, "articles_cited": articles_cited}
