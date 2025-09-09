#!/usr/bin/env python3
# dont forget to init with `./code_search_cli.py init`

import os
import click
import psycopg2
import openai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

openai.api_key = os.getenv("OPENAI_API_KEY")
DB_CONN = os.getenv("DATABASE_URL")  # e.g. postgres://user:pass@host:25060/dbname
CHUNK_SIZE = 50  # ~50 lines per chunk


def get_conn():
    return psycopg2.connect(DB_CONN)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS code_chunks (
        id SERIAL PRIMARY KEY,
        file_path TEXT,
        chunk_id INT,
        content TEXT,
        embedding VECTOR(1536)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()


def embed_text(text):
    resp = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return resp['data'][0]['embedding']


def ingest_file(file_path):
    conn = get_conn()
    cur = conn.cursor()

    with open(file_path, "r", errors="ignore") as f:
        lines = f.readlines()

    for i in range(0, len(lines), CHUNK_SIZE):
        chunk = "".join(lines[i:i+CHUNK_SIZE]).strip()
        if not chunk:
            continue
        embedding = embed_text(chunk)
        cur.execute(
            "INSERT INTO code_chunks (file_path, chunk_id, content, embedding) VALUES (%s, %s, %s, %s)",
            (file_path, i // CHUNK_SIZE, chunk, embedding)
        )

    conn.commit()
    cur.close()
    conn.close()


def ingest_dir(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".php", ".js", ".ts", ".py")):
                path = os.path.join(root, file)
                console.print(f"[cyan]Ingesting[/cyan] {path}")
                ingest_file(path)


def search(query, k=5):
    conn = get_conn()
    cur = conn.cursor()
    q_embed = embed_text(query)
    cur.execute(
        "SELECT file_path, content FROM code_chunks ORDER BY embedding <-> %s LIMIT %s",
        (q_embed, k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def ask_llm(question, context):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a code reviewer. Follow the repo’s coding style."},
            {"role": "user", "content": f"Question: {question}\nContext:\n{context}"}
        ]
    )
    return resp['choices'][0]['message']['content']


@click.group()
def cli():
    """📚 Code Search + Analysis CLI (pgvector + OpenAI)"""
    pass


@cli.command()
def init():
    """Initialize database schema"""
    init_db()
    console.print("[green]Database initialized ✅[/green]")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def ingest(path):
    """Ingest code (file or directory) into DB"""
    if os.path.isdir(path):
        ingest_dir(path)
    else:
        ingest_file(path)
    console.print("[green]Ingestion complete ✅[/green]")


@cli.command()
@click.argument("query")
@click.option("-k", default=5, help="Number of results")
def search_cmd(query, k):
    """Search codebase with embeddings"""
    results = search(query, k)
    for r in results:
        console.print(Panel(r[1][:500] + "...", title=f"[yellow]{r[0]}[/yellow]", expand=False))


@cli.command()
@click.argument("question")
@click.option("-k", default=5, help="Number of context snippets")
def analyze(question, k):
    """Ask LLM about codebase with context"""
    results = search(question, k)
    context = "\n\n".join([f"File: {r[0]}\n{r[1]}" for r in results])
    answer = ask_llm(question, context)
    console.print(Panel(Markdown(answer), title="[magenta]LLM Answer[/magenta]"))


if __name__ == "__main__":
    cli()
