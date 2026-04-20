import os
from config import settings
from typing import List
import httpx
import time
from fastapi import HTTPException
import asyncio

CHAT_URL = "https://api.openai.com/v1/chat/completions"
EMBED_URL = "https://api.openai.com/v1/embeddings"
# Embedding configuration (tweak in config if desired)
EMBED_BATCH_SIZE = getattr(settings, "EMBED_BATCH_SIZE", 12)
EMBED_TIMEOUT_SECS = getattr(settings, "EMBED_TIMEOUT_SECS", 120)
EMBED_MAX_RETRIES = getattr(settings, "EMBED_MAX_RETRIES", 3)
EMBED_BACKOFF_BASE = getattr(settings, "EMBED_BACKOFF_BASE", 1.8)
MAX_TEXT_LENGTH_FOR_EMBED = getattr(settings, "MAX_TEXT_LENGTH_FOR_EMBED", 8000)

# Storage paths (local). Change to S3/presigned URLs if needed.
UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# ---------- Embedding & LLM (robust) ----------
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Robust embedding call:
      - Splits into smaller batches (EMBED_BATCH_SIZE)
      - Retries transient failures with exponential backoff
      - Uses EMBED_TIMEOUT_SECS per request
    Returns embeddings in same order as `texts`.
    """
    if not texts:
        return []

    url = EMBED_URL
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # split into batches preserving order
    batches = [texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECS) as client:
        for batch_idx, batch_texts in enumerate(batches):
            payload = {"model": settings.EMBEDDING_MODEL, "input": batch_texts}
            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    emb_batch = [d["embedding"] for d in data["data"]]
                    if len(emb_batch) != len(batch_texts):
                        raise HTTPException(status_code=502, detail="Embedding response length mismatch")
                    results.extend(emb_batch)
                    break
                except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                    # network/transient errors -> retry
                    if attempt >= EMBED_MAX_RETRIES:
                        raise HTTPException(
                            status_code=504,
                            detail=f"Embedding request timed out after {EMBED_MAX_RETRIES} attempts (batch {batch_idx}). Last error: {str(exc)}",
                        )
                    backoff = EMBED_BACKOFF_BASE ** (attempt - 1)
                    jitter = (0.1 * backoff) * (0.5 - (time.time() % 1))
                    await asyncio.sleep(backoff + jitter)
                    continue
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    text = ""
                    try:
                        text = exc.response.text
                    except Exception:
                        pass
                    # Retry on 5xx
                    if 500 <= status < 600 and attempt < EMBED_MAX_RETRIES:
                        await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail=f"Embedding service returned status {status}: {text[:200]}",
                    )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Embedding failed: {str(exc)}")

    return results


async def embed_single(text: str) -> List[float]:
    embs = await embed_texts([text])
    return embs[0]
