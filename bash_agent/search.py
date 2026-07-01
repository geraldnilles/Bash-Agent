#!/usr/bin/env python3
"""
Semantic search tool that indexes files and finds similar documents using embeddings.
"""
import os
import sys
import json
import argparse
import hashlib
import fnmatch
import numpy as np
import requests

# Import configuration from config.py
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bash_agent import config
from bash_agent import llm

# Always force embedding to use OpenRouter

# Constants
EMBEDDINGS_DB = os.path.abspath(".bash_agent_tmp/embeddings.json")
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
RERANK_MODEL = "cohere/rerank-4-pro"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
SEARCH_DISABLED_FLAG = os.path.abspath(".bash_agent_tmp/search_disabled")


def get_ignore_patterns():
    """Get ignore patterns from .gitignore and hardcoded exclusions."""
    ignore_patterns = {".git", ".bash_agent_tmp", "__pycache__"}
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_patterns.add(line)
    return ignore_patterns


def get_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_all_files(root_dir):
    """Walk directory and return list of files respecting ignore patterns."""
    ignore_patterns = get_ignore_patterns()
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in ignore_patterns)]
        for filename in filenames:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, root_dir)
            if not any(fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(filename, p) for p in ignore_patterns):
                files.append(rel_path)
    return files


def load_embeddings_db():
    """Load the embeddings database from JSON file."""
    if os.path.exists(EMBEDDINGS_DB):
        with open(EMBEDDINGS_DB, "r") as f:
            return json.load(f)
    return {}


def save_embeddings_db(db):
    """Save the embeddings database to JSON file."""
    os.makedirs(os.path.dirname(EMBEDDINGS_DB), exist_ok=True)
    with open(EMBEDDINGS_DB, "w") as f:
        json.dump(db, f, indent=2)


def get_file_content(file_path, root_dir, max_chars=100000):
    """Read file content for embedding. Skip binary files."""
    try:
        full_path = os.path.join(root_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Truncate very long files to save API costs
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n... [truncated {len(content) - max_chars} characters]"
            return content
    except (UnicodeDecodeError, PermissionError, FileNotFoundError):
        return None


def fetch_embedding(client, texts):
    """Fetch embedding utilizing the dynamic adapter layer."""
    response = llm.create_embedding(model=EMBEDDING_MODEL, input_texts=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))



def rerank_documents(client, query, documents, top_n):
    """Rerank documents using OpenRouter's rerank API. Returns list of (index, score) sorted by relevance."""
    if not documents:
        return []
    
    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set. Skipping reranking.", file=sys.stderr)
        return [(i, 0.0) for i in range(len(documents))]
    
    url = "https://openrouter.ai/api/v1/rerank"
    payload = {
        "documents": documents,
        "model": RERANK_MODEL,
        "query": query,
        "top_n": min(top_n, len(documents))
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Parse reranker response: expected format has "results" with "index" and "relevance_score"
        if "results" in result:
            scored = [(r["index"], r.get("relevance_score", 0.0)) for r in result["results"]]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        else:
            print(f"Unexpected rerank response format: {result}", file=sys.stderr)
            return [(i, 0.0) for i in range(len(documents))]
    except Exception as e:
        print(f"Reranking failed: {e}. Falling back to embedding similarity.", file=sys.stderr)
        return [(i, 0.0) for i in range(len(documents))]

def main():
    parser = argparse.ArgumentParser(description="Semantic search tool for the codebase")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("-n", "--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug/informational output")
    args = parser.parse_args()

    def debug_print(*a, **kw):
        """Print only when --debug flag is enabled."""
        if args.debug:
            print(*a, **kw)

    root_dir = os.getcwd()

    # Check if search has been disabled for this directory
    if os.path.exists(SEARCH_DISABLED_FLAG):
        print("The user has disabled search for this directory. Use other commands to navigate.", file=sys.stderr)
        sys.exit(1)
    client = llm.get_llm_client()
    
    debug_print(f"Indexing files in {root_dir}...", file=sys.stderr)
    
    # Load existing embeddings database
    db = load_embeddings_db()
    
    # Get all files in the directory
    all_files = get_all_files(root_dir)
    debug_print(f"Found {len(all_files)} files to consider.", file=sys.stderr)
    
    # Remove stale entries from database (files that no longer exist)
    stale_keys = []
    for db_file_path in list(db.keys()):
        full_path = os.path.join(root_dir, db_file_path)
        if not os.path.exists(full_path):
            stale_keys.append(db_file_path)
    if stale_keys:
        for key in stale_keys:
            del db[key]
        debug_print(f"Cleaned up {len(stale_keys)} stale entries from embedding database.", file=sys.stderr)
    
    # Check which files need re-embedding
    needs_embedding = []
    for file_path in all_files:
        try:
            current_hash = get_file_content(file_path, root_dir, max_chars=1)  # Just check existence
            if current_hash is None:
                continue
            full_path = os.path.join(root_dir, file_path)
            current_hash = get_file_hash(full_path)
        except (FileNotFoundError, PermissionError):
            continue
        
        if file_path not in db or db[file_path].get("hash") != current_hash:
            needs_embedding.append(file_path)
    
    debug_print(f"Files needing re-embedding: {len(needs_embedding)}", file=sys.stderr)
    
    # Fetch embeddings for files that need it (batched, 10 at a time)
    for batch_start in range(0, len(needs_embedding), 10):
        batch_end = min(batch_start + 10, len(needs_embedding))
        batch = needs_embedding[batch_start:batch_end]
        debug_print(f"Embedding batch {batch_start+1}-{batch_end}/{len(needs_embedding)}", file=sys.stderr)
        
        # Collect content for this batch
        batch_contents = []
        batch_file_paths = []
        for file_path in batch:
            content = get_file_content(file_path, root_dir)
            if content is not None:
                batch_contents.append(content)
                batch_file_paths.append(file_path)
        
        if not batch_contents:
            continue
        
        try:
            embeddings = fetch_embedding(client, batch_contents)
            for file_path, embedding in zip(batch_file_paths, embeddings):
                full_path = os.path.join(root_dir, file_path)
                file_hash = get_file_hash(full_path)
                db[file_path] = {"hash": file_hash, "embedding": embedding}
        except Exception as e:
            print(f"Error embedding batch: {e}", file=sys.stderr)
    
    # Save updated database
    if needs_embedding:
        save_embeddings_db(db)
        debug_print(f"Saved embeddings database to {EMBEDDINGS_DB}", file=sys.stderr)
    
    # Get query embedding
    debug_print(f"Embedding query: {args.query[:50]}...", file=sys.stderr)
    try:
        query_embedding = fetch_embedding(client, [args.query])[0]
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate similarities
    results = []
    for file_path, data in db.items():
        if "embedding" in data:
            sim = cosine_similarity(query_embedding, data["embedding"])
            results.append((file_path, sim))
    
    # Sort by similarity to get initial candidate pool
    results.sort(key=lambda x: x[1], reverse=True)
    # Use a larger pool for reranking (5x top_k, min 20)
    rerank_pool_size = max(args.top_k * 5, min(20, len(results)))
    candidate_pool = results[:rerank_pool_size]
    
    # Fetch file contents for reranking
    candidate_paths = [fp for fp, _ in candidate_pool]
    debug_print(f"Fetching contents for {len(candidate_paths)} candidates for reranking...", file=sys.stderr)
    candidate_documents = []
    for fp in candidate_paths:
        content = get_file_content(fp, root_dir, max_chars=4000)
        if content is None:
            content = ""
        candidate_documents.append(content)
    
    # Rerank candidates
    debug_print("Reranking candidates...", file=sys.stderr)
    reranked = rerank_documents(client, args.query, candidate_documents, args.top_k)
    
    # Build final results from reranked order
    top_results = []
    for idx, score in reranked:
        if idx < len(candidate_pool):
            file_path, sim = candidate_pool[idx]
            top_results.append((file_path, score))
    
    # Fall back to embedding similarity if reranking returned nothing
    if not top_results:
        top_results = [(fp, sim) for fp, sim in results[:args.top_k]]
    
    # Output results
    for file_path, score in top_results:
        print(f"{score:.4f}\t{file_path}")


if __name__ == "__main__":
    main()
