import os
import httpx
import fitz  # PyMuPDF
from openai import OpenAI
from database import setup_table, insert_chunks, clear_chunks
from dotenv import load_dotenv

load_dotenv()

CONSTITUTION_URL = "https://www.parliament.go.ke/sites/default/files/2017-05/The_Constitution_of_Kenya_2010.pdf"
CHUNK_SIZE = 500      # tokens (approx chars / 4)
CHUNK_OVERLAP = 50

def _openai():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

IEBC_TEXT = """
Voter Registration in Kenya — IEBC Guidelines

Voter registration is open to Kenyan citizens aged 18 and above.
Required documents: Original National ID card or valid Kenyan Passport.
Registration is done at IEBC constituency offices or during mobile registration drives.
After registration, you receive a voter's card which you must carry on election day.
You can check your registration status at elections.iebc.or.ke.
Registration deadlines are announced by IEBC at least 60 days before an election.
You can transfer your registration to a new constituency if you have moved.
""".strip()

COUNTY_TEXT = """
Kenya County Government — Civic Information

Kenya has 47 counties each headed by an elected Governor.
Each county has a County Assembly led by a Speaker with elected Ward Representatives (MCAs).
County governments handle: health services, early childhood education, agriculture, county roads, trade licensing.
National government handles: defence, foreign affairs, national security, immigration, universities.
County budgets are published by the Controller of Budget at Kenya.
Citizens have the right to attend county assembly proceedings and public participation forums.
Ward representatives (MCAs) are the closest elected officials to citizens at ward level.
""".strip()


def _char_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end].strip())
        start += chunk_chars - overlap_chars
    return [c for c in chunks if len(c) > 50]


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def run_ingest() -> int:
    setup_table()
    clear_chunks()

    # Download and chunk the Constitution PDF
    _openai()  # validate key exists before doing work
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(CONSTITUTION_URL)
        resp.raise_for_status()
        pdf_bytes = resp.content

    constitution_text = _extract_pdf_text(pdf_bytes)
    chunk_chars = CHUNK_SIZE * 4
    overlap_chars = CHUNK_OVERLAP * 4

    all_chunks: list[dict] = []

    for i, chunk in enumerate(_char_chunks(constitution_text, chunk_chars, overlap_chars)):
        all_chunks.append({"content": chunk, "source": "Kenya Constitution 2010", "chunk_index": i})

    # Add static knowledge base texts
    for source, text in [
        ("IEBC Voter Registration Guidelines", IEBC_TEXT),
        ("Kenya County Government Information", COUNTY_TEXT),
    ]:
        for i, chunk in enumerate(_char_chunks(text, chunk_chars, overlap_chars)):
            all_chunks.append({"content": chunk, "source": source, "chunk_index": i})

    # Embed in batches of 100
    client = _openai()
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings = _embed_batch(client, [c["content"] for c in batch])
        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb

    insert_chunks(all_chunks)
    return len(all_chunks)
