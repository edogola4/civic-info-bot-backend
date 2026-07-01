import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()


def _raw_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg2.connect(url)


def get_conn():
    """Returns a connection with the vector type registered."""
    conn = _raw_conn()
    register_vector(conn)
    return conn


def setup_table():
    # Step 1: create extension using a plain connection (vector type not registered yet)
    with _raw_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    # Step 2: now the extension exists, safe to use vector-aware connection
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS civic_knowledge (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS civic_knowledge_embedding_idx
                ON civic_knowledge USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        conn.commit()


def insert_chunks(chunks: list[dict]):
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO civic_knowledge (content, embedding, source, chunk_index)
                VALUES %s
                """,
                [(c["content"], c["embedding"], c["source"], c["chunk_index"]) for c in chunks],
            )
        conn.commit()


def similarity_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source
                FROM civic_knowledge
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, top_k),
            )
            rows = cur.fetchall()
    return [{"content": r[0], "source": r[1]} for r in rows]


def count_chunks() -> int:
    with _raw_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM civic_knowledge;")
            return cur.fetchone()[0]


def clear_chunks():
    with _raw_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM civic_knowledge;")
        conn.commit()
