# backend/main.py  —  CS Tutor Platform (Company Secretary)
import io
import os
import unicodedata
import asyncio
import time
import shutil
from datetime import datetime, timedelta
from typing import Any, List, Optional, Dict
import re
import tempfile
import secrets
import traceback

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Header,
    Form,
    APIRouter,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import httpx
import pinecone
from pypdf import PdfReader
from typing import Literal

from config import settings
from cs_text_normalizer import expand_cs_abbreviations
from ingestion.enhanced_upload_service import process_pdf_enhanced
from email_service import send_admin_signup_notification, send_password_reset_otp
from payment_router import router as payment_router
from s3_service import upload_pdf_to_s3, delete_pdf_from_s3, is_s3_configured


# ============================================================
# APP + CORS
# ============================================================

app = FastAPI(title="CS Tutor — AI-Powered Company Secretary Exam Prep")

_raw_origin    = settings.FRONTEND_ORIGIN.strip()
_allow_origins = ["*"] if _raw_origin == "*" else [
    o.strip() for o in _raw_origin.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins      = _allow_origins,
    allow_origin_regex = r"https://.*\.vercel\.app",
    allow_credentials  = True,
    allow_methods      = ["*"],
    allow_headers      = ["*"],
    expose_headers     = ["*"],
)


# ============================================================
# GLOBAL ERROR MIDDLEWARE  —  exposes real crash in logs
# ============================================================

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print("GLOBAL SERVER ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"},
        )


# ── Routers must be included AFTER middleware ────────────────────────────────
app.include_router(payment_router, prefix="/payments", tags=["payments"])

CHAT_URL  = "https://api.openai.com/v1/chat/completions"
EMBED_URL = "https://api.openai.com/v1/embeddings"

# Gemini Flash endpoint — cheap async summarisation
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)


# ============================================================
# DB + EXTERNAL CLIENTS
# ============================================================

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"], default="pbkdf2_sha256", deprecated="auto"
)

mongo_client         = AsyncIOMotorClient(settings.MONGO_URI)
db                   = mongo_client[settings.MONGO_DB]
users_collection     = db["users"]
docs_collection      = db["documents"]
dashboard_collection = db["cs_dashboard"]

pinecone_client = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
index           = pinecone_client.Index(settings.PINECONE_INDEX)

JWT_EXP_MINUTES           = 60 * 24
EMBED_BATCH_SIZE          = getattr(settings, "EMBED_BATCH_SIZE",          12)
EMBED_TIMEOUT_SECS        = getattr(settings, "EMBED_TIMEOUT_SECS",        120)
EMBED_MAX_RETRIES         = getattr(settings, "EMBED_MAX_RETRIES",         3)
EMBED_BACKOFF_BASE        = getattr(settings, "EMBED_BACKOFF_BASE",        1.8)
MAX_TEXT_LENGTH_FOR_EMBED = getattr(settings, "MAX_TEXT_LENGTH_FOR_EMBED", 8000)

# Local uploads folder — fallback when S3 is not configured
UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

# ── Conversation memory constants ─────────────────────────────────────────────
MAX_TURNS       = 10   # Q+A pairs kept in the sliding window per user
SUMMARIZE_EVERY = 5    # trigger Gemini summarisation every N new turns


# ============================================================
# STARTUP VALIDATION
# ============================================================

@app.on_event("startup")
async def validate_config():
    if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is missing or invalid in .env")
    if not getattr(settings, "LLM_MODEL", ""):
        raise RuntimeError("LLM_MODEL is not set in .env")
    gemini_key = getattr(settings, "GEMINI_API_KEY", "")
    if not gemini_key:
        print("[startup] WARNING: GEMINI_API_KEY not set — conversation summarisation disabled")
    print(f"[startup] Platform        = CS Tutor (Company Secretary)")
    print(f"[startup] DB              = {settings.MONGO_DB}")
    print(f"[startup] Pinecone index  = {settings.PINECONE_INDEX}")
    print(f"[startup] LLM_MODEL       = {settings.LLM_MODEL}")
    print(f"[startup] EMBEDDING_MODEL = {settings.EMBEDDING_MODEL}")
    print(f"[startup] Gemini summary  = {'enabled' if gemini_key else 'disabled (set GEMINI_API_KEY to enable)'}")


# ============================================================
# PYDANTIC MODELS
# ============================================================

class UserCreate(BaseModel):
    email:      str
    password:   str
    name:       str
    phone:      str
    cs_level:   str                    # CSEET | Executive | Professional
    cs_attempt: str
    role:       str = "student"
    plan:       Optional[str] = "free"
    payment_id: Optional[str] = None

class UserLogin(BaseModel):
    email:    str
    password: str

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"

class UserOut(BaseModel):
    email:               str
    role:                str
    plan:                Optional[str] = "free"
    subscription_status: Optional[str] = "free"

class ChatResponse(BaseModel):
    answer:  str
    sources: List[dict]

class ChatMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    mode:    Optional[str] = "qa"   # "qa" | "discussion"

class UploadResponse(BaseModel):
    success:            bool
    message:            str
    filename:           str
    storage_backend:    str
    processing_status:  str

class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp:   str

class ResetPasswordRequest(BaseModel):
    email:        str
    otp:          str
    new_password: str


# ============================================================
# AUTH HELPERS
# ============================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXP_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGO)

async def get_user_by_email(email: str):
    return await users_collection.find_one({"email": email})

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGO])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_current_admin(user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ============================================================
# HELPERS  (upload + chat shared)
# ============================================================

def _save_local(content: bytes, safe_filename: str) -> str:
    """Write bytes to UPLOAD_ROOT and return the URL path."""
    with open(os.path.join(UPLOAD_ROOT, safe_filename), "wb") as f:
        f.write(content)
    base_url = getattr(settings, "BASE_URL", "")
    return f"{base_url}/uploads/{safe_filename}" if base_url else f"/uploads/{safe_filename}"


def _save_local_from_path(src_path: str, safe_filename: str) -> str:
    """Copy a file from src_path to UPLOAD_ROOT and return the URL path."""
    dest_path = os.path.join(UPLOAD_ROOT, safe_filename)
    shutil.copy(src_path, dest_path)
    base_url = getattr(settings, "BASE_URL", "")
    return f"{base_url}/uploads/{safe_filename}" if base_url else f"/uploads/{safe_filename}"


def _safe_unlink(path: Optional[str]) -> None:
    if path:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


# ============================================================
# EMBEDDING + LLM  (OpenAI)
# ============================================================

async def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    batches: List[List[str]] = [
        texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)
    ]
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECS) as client:
        for batch_idx, batch_texts in enumerate(batches):
            payload = {"model": settings.EMBEDDING_MODEL, "input": batch_texts}
            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = await client.post(EMBED_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    emb_batch = [d["embedding"] for d in resp.json()["data"]]
                    if len(emb_batch) != len(batch_texts):
                        raise HTTPException(status_code=502, detail="Embedding length mismatch")
                    results.extend(emb_batch)
                    break
                except (httpx.ReadTimeout, httpx.WriteTimeout,
                        httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                    if attempt >= EMBED_MAX_RETRIES:
                        raise HTTPException(
                            status_code=504,
                            detail=f"Embedding timeout (batch {batch_idx}): {exc}"
                        )
                    await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
                except httpx.HTTPStatusError as exc:
                    sc = exc.response.status_code
                    et = ""
                    try: et = exc.response.text
                    except Exception: pass
                    if 500 <= sc < 600 and attempt < EMBED_MAX_RETRIES:
                        await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
                        continue
                    raise HTTPException(status_code=502, detail=f"Embedding error {sc}: {et[:200]}")
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    return results


async def embed_single(text: str) -> List[float]:
    return (await embed_texts([text]))[0]


async def call_llm(messages: List[dict], temperature: float = 0.2) -> str:
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(CHAT_URL, headers=headers, json={
            "model": settings.LLM_MODEL, "messages": messages, "temperature": temperature
        })
        if resp.status_code == 404:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"OpenAI model '{settings.LLM_MODEL}' not found. "
                    "Check LLM_MODEL in .env — valid: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini"
                )
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def call_llm_with_chain(
    *,
    user_question:       str,
    context:             str,
    final_system_prompt: str,
    temperature:         float = 0.2,
) -> str:
    """
    Two-step chain:
      1. Classify query type (DEFINITION | PROCEDURE | CASE_LAW | NUMERICAL | COMPARISON | GENERAL)
      2. Generate final answer with query-type context injected into the system prompt.
    Only the final answer is returned.
    """
    classification_messages = [
        {
            "role":    "system",
            "content": (
                "You are a query analyser for a CS (Company Secretary) exam platform. "
                "Classify the student's query into ONE of these categories:\n"
                "DEFINITION | PROCEDURE | CASE_LAW | NUMERICAL | COMPARISON | GENERAL\n"
                "Respond with ONLY the category label, nothing else."
            ),
        },
        {"role": "user", "content": user_question},
    ]
    try:
        query_type = await call_llm(classification_messages, temperature=0.0)
    except Exception:
        query_type = "GENERAL"

    messages = [
        {
            "role":    "system",
            "content": final_system_prompt + f"\n\nQuery classification: {query_type}",
        },
        {"role": "system", "content": f"CONTEXT:\n{context}"},
        {"role": "user",   "content": user_question},
    ]
    return await call_llm(messages, temperature=temperature)


# ============================================================
# GEMINI FLASH  —  cheap async summarisation
# ============================================================

async def call_gemini_flash(prompt: str) -> str:
    """
    Call Gemini 2.0 Flash for cheap rolling summarisation.
    Returns empty string on any failure — summarisation is non-critical.
    """
    gemini_key = getattr(settings, "GEMINI_API_KEY", "")
    if not gemini_key:
        return ""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.2,
            "maxOutputTokens": 500,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GEMINI_URL,
                params={"key": gemini_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[Gemini] Summarisation failed (non-critical): {e}")
        return ""


# ============================================================
# CONVERSATION MEMORY  —  MongoDB helpers
# ============================================================

async def get_user_memory(user_email: str) -> Dict[str, Any]:
    user = await users_collection.find_one(
        {"email": user_email},
        {"conversation_turns": 1, "conversation_summary": 1, "total_turns": 1},
    )
    if not user:
        return {"turns": [], "summary": "", "total": 0}
    return {
        "turns":   user.get("conversation_turns",  []),
        "summary": user.get("conversation_summary", ""),
        "total":   user.get("total_turns",          0),
    }


async def save_turn_and_maybe_summarize(
    user_email:     str,
    question:       str,
    answer:         str,
    current_memory: Dict[str, Any],
) -> None:
    now = datetime.utcnow().isoformat()

    new_turns = current_memory["turns"] + [
        {"role": "user",      "content": question, "ts": now},
        {"role": "assistant", "content": answer,   "ts": now},
    ]
    new_turns = new_turns[-(MAX_TURNS * 2):]
    new_total = current_memory["total"] + 1

    update: Dict[str, Any] = {
        "conversation_turns": new_turns,
        "total_turns":        new_total,
    }

    if new_total % SUMMARIZE_EVERY == 0:
        turns_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:400]}"
            for t in new_turns
        )
        existing_summary = current_memory["summary"]

        gemini_prompt = (
            "You are a learning tracker for an Indian CS (Company Secretary) student "
            "preparing for ICSI exams.\n\n"
            f"EXISTING SUMMARY:\n{existing_summary or 'None yet — this is the first summary.'}\n\n"
            f"LATEST CONVERSATION TURNS:\n{turns_text}\n\n"
            "Write an updated 3-5 sentence summary covering:\n"
            "1. Which CS topics, Acts, and subjects the student has studied so far\n"
            "2. Their apparent weak areas or topics they asked about repeatedly\n"
            "3. Their preferred language (Hindi / English / Hinglish)\n"
            "4. Any useful patterns (e.g. exam-focused, concept-heavy, needs case laws)\n\n"
            "Rules:\n"
            "- Be concise. Do NOT copy Q&A verbatim — compress into insights.\n"
            "- Merge the existing summary with new observations; do not discard old info.\n"
            "- Write in plain English. No bullet points. Max 5 sentences."
        )

        new_summary = await call_gemini_flash(gemini_prompt)
        if new_summary:
            update["conversation_summary"] = new_summary
            update["summary_updated_at"]   = now
            print(f"[memory] Summary updated for {user_email} (turn #{new_total})")

    try:
        await users_collection.update_one(
            {"email": user_email},
            {"$set": update},
            upsert=False,
        )
    except Exception as e:
        print(f"[memory] Failed to save turn for {user_email}: {e}")


def build_memory_block(memory: Dict[str, Any]) -> str:
    parts: List[str] = []

    if memory["summary"]:
        parts.append(f"STUDENT LEARNING SUMMARY:\n{memory['summary']}")

    if memory["turns"]:
        recent_turns = memory["turns"][-(MAX_TURNS * 2):]
        lines = []
        for t in recent_turns:
            role    = t.get("role", "user").upper()
            content = t.get("content", "")[:500]
            lines.append(f"{role}: {content}")
        parts.append("RECENT CONVERSATION (last turns):\n" + "\n".join(lines))

    if not parts:
        return ""

    return (
        "=== STUDENT MEMORY ===\n"
        + "\n\n".join(parts)
        + "\n=== END MEMORY ===\n\n"
    )


# ============================================================
# QUERY HELPERS  (used only by /chat)
# ============================================================

def enrich_query_for_rag(question: str) -> str:
    q, hints = question.lower(), []
    if any(w in q for w in ["companies act", "director", "board", "agm", "egm", "nclt"]):
        hints.append("company law corporate governance")
    if any(w in q for w in ["sebi", "securities", "listing", "takeover", "insider"]):
        hints.append("securities law capital markets")
    if any(w in q for w in ["winding up", "insolvency", "ibc", "liquidation", "resolution"]):
        hints.append("insolvency bankruptcy code")
    if any(w in q for w in ["gst", "indirect tax", "customs", "excise"]):
        hints.append("indirect tax gst")
    if any(w in q for w in ["income tax", "tds", "direct tax", "section 80"]):
        hints.append("direct tax income tax")
    if any(w in q for w in ["fema", "rbi", "foreign exchange", "fdi", "ecb"]):
        hints.append("fema foreign exchange management")
    if any(w in q for w in ["secretarial", "complian", "mca", "roc", "filing"]):
        hints.append("secretarial compliance")
    if any(w in q for w in ["arbitration", "contract", "tort", "ipc"]):
        hints.append("general law jurisprudence")
    return question + (" " + " ".join(hints) if hints else "")


def detect_subject(question: str) -> Optional[str]:
    q = question.lower()
    if any(w in q for w in ["companies act", "director", "board", "agm", "egm", "nclt", "nclat",
                              "moa", "aoa", "incorporation", "share capital"]):
        return "Company Law"
    if any(w in q for w in ["sebi", "securities", "listing", "takeover", "insider trading",
                              "depository", "mutual fund", "ipo"]):
        return "Securities Law"
    if any(w in q for w in ["ibc", "insolvency", "liquidation", "winding up",
                              "resolution professional", "nclt insolvency"]):
        return "Insolvency Law"
    if any(w in q for w in ["gst", "indirect tax", "customs", "igst", "sgst", "cgst"]):
        return "Indirect Tax"
    if any(w in q for w in ["income tax", "direct tax", "tds", "section 80", "capital gain"]):
        return "Direct Tax"
    if any(w in q for w in ["fema", "foreign exchange", "fdi", "rbi", "ecb", "nri"]):
        return "FEMA"
    if any(w in q for w in ["secretarial", "compliance", "mca", "roc", "annual return", "form mgt"]):
        return "Secretarial Practice"
    if any(w in q for w in ["contract", "arbitration", "specific performance", "tort"]):
        return "General Law"
    return None


async def is_cs_related_question(question: str) -> bool:
    _platform_keywords = [
        "cs tutor", "this app", "this platform", "who made", "who built",
        "who created", "who started", "about this", "founder", "promoter",
        "tell me about", "what is this app",
    ]
    if any(kw in question.lower() for kw in _platform_keywords):
        return True

    system = (
        "You are a domain classifier for an Indian Company Secretary (CS) assistant.\n\n"
        "Answer YES if the question relates to:\n"
        "- ICSI syllabus, Company Law, Securities Law, Insolvency Law, FEMA,\n"
        "  Direct Tax, Indirect Tax/GST, Secretarial Practice, General Law,\n"
        "  Corporate Governance, CS exams (CSEET / Executive / Professional),\n"
        "  or basic commerce/legal concepts studied by CS students.\n"
        "- The CS Tutor app, its features, team, or anything about this platform.\n\n"
        "Answer NO only if it is clearly unrelated (pure science, coding, sports, entertainment).\n\n"
        "Respond with YES or NO only."
    )
    try:
        result = await call_llm([
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ])
        return result.strip().upper().startswith("YES")
    except Exception:
        return True


# ============================================================
# PERSONALISATION
# ============================================================

def build_personalized_layer(user: dict) -> str:
    name  = user.get("name", "Student").split()[0]
    level = user.get("cs_level", "CSEET")
    guidance = {
        "CSEET":        "Explain in simple language with basic concepts and easy examples. Focus on fundamentals.",
        "Executive":    "Explain with clarity and practical understanding. Include relevant sections and examples.",
        "Professional": "Provide detailed, professional-level explanation. Reference Acts, Rules, Case Laws and ICSI exam perspective.",
    }
    return (
        f"PERSONALISATION CONTEXT:\n"
        f"- Student Name: {name}\n"
        f"- CS Level: {level}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Occasionally address the student as {name}\n"
        f"- Teaching style: {guidance.get(level, guidance['CSEET'])}\n"
        f"- Keep tone supportive, professional, and exam-focused\n"
        f"- Simplify complex legal language where necessary without losing accuracy."
    )


# ============================================================
# ADMIN MATERIALS ROUTER
# ============================================================

router = APIRouter(prefix="/admin/materials")


# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND PROCESSING FUNCTION
# Runs AFTER the upload route returns 200 to the client.
# Handles: PDF parsing → embeddings → Pinecone → MongoDB records.
# ──────────────────────────────────────────────────────────────────────────────

async def process_pdf_background(
    temp_file_path:            str,
    file_name:                 str,
    extra_meta:                dict,
    enable_image_descriptions: bool,
    admin_email:               str,
    course:                    str,
    resolved_subject:          str,
    resolved_module:           str,
    resolved_chapter:          str,
    resolved_unit:             str,
    section:                   str,
    custom_heading:            str,
    pdf_url:                   str,
    storage_backend:           str,
    safe_filename:             str,
):
    try:
        print(f"[BACKGROUND] Processing started: {file_name}")

        result = await process_pdf_enhanced(
            file_path                 = temp_file_path,
            file_name                 = file_name,
            extra_meta                = extra_meta,
            enable_image_descriptions = enable_image_descriptions,
            openai_api_key            = settings.OPENAI_API_KEY,
        )

        now = datetime.utcnow()

        doc_record = {
            "filename":        file_name,
            "safe_filename":   safe_filename,
            "course":          course,
            "level":           course,
            "subject":         resolved_subject,
            "chapter":         resolved_chapter,
            "section":         section or "",
            "unit":            resolved_unit,
            "module":          resolved_module,
            "custom_heading":  custom_heading or "",
            "pdf_url":         pdf_url or "",
            "storage_backend": storage_backend,
            "uploaded_by":     admin_email,
            "uploaded_at":     now,
            "total_vectors":   result["total_vectors"],
            "platform":        "cs_tutor",
            "processing_status": "completed",
        }

        inserted = await docs_collection.insert_one(doc_record)

        await dashboard_collection.insert_one({
            "level":              course,
            "subject":            resolved_subject,
            "module":             resolved_module,
            "chapter":            resolved_chapter if resolved_chapter else file_name.replace(".pdf", ""),
            "unit":               resolved_unit,
            "title":              file_name.replace(".pdf", "").replace("_", " "),
            "pdf_url":            pdf_url,
            "video_url":          None,
            "audio_url":          None,
            "simplified_pdf_url": None,
            "processing_status":  "completed",
            "source_doc":         str(inserted.inserted_id),
            "uploaded_by":        admin_email,
            "created_at":         now,
            "platform":           "cs_tutor",
        })

        print(f"[BACKGROUND] Processing completed: {file_name}  |  vectors={result['total_vectors']}")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {file_name}: {str(e)}")
        traceback.print_exc()

    finally:
        # Always clean up the temp file regardless of success/failure
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"[BACKGROUND] Temp file removed: {temp_file_path}")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# UPLOAD ROUTE  —  Render-safe version
#
# Key fixes vs. original:
#   1. Streams file to disk in 1 MB chunks  →  no RAM spike
#   2. Returns 200 immediately              →  no request timeout / fake CORS error
#   3. Heavy work runs in BackgroundTask    →  Render worker stays alive
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/upload_enhanced", response_model=UploadResponse)
async def upload_pdf_enhanced(
    background_tasks: BackgroundTasks,
    file:    UploadFile        = File(...),
    course:  str               = Form(...),
    subject: Optional[str]     = Form(None),
    module:  Optional[str]     = Form(None),
    chapter: Optional[str]     = Form(None),
    unit:    Optional[str]     = Form(None),
    section: Optional[str]     = Form(None),
    custom_heading: Optional[str] = Form(None),
    enable_image_descriptions: bool = Form(True),
    admin = Depends(get_current_admin),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        level            = course.strip()
        resolved_subject = (subject or chapter or "General").strip()
        resolved_module  = (module  or section or "General").strip()
        resolved_chapter = (chapter or "").strip()
        resolved_unit    = (unit    or "").strip()
        safe_filename    = file.filename.replace(" ", "_")

        # ── STEP 1: Stream file to disk in 1 MB chunks — NO full RAM load ────
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

        while True:
            chunk = await file.read(1024 * 1024)   # 1 MB at a time
            if not chunk:
                break
            temp_file.write(chunk)

        temp_file.close()
        temp_file_path = temp_file.name
        print(f"[UPLOAD] File streamed to disk: {temp_file_path}  ({safe_filename})")

        # ── STEP 2: Upload to storage (S3 or local) ───────────────────────────
        s3_ready = is_s3_configured()
        print(f"[UPLOAD] S3 configured={s3_ready}  file={safe_filename}")

        if s3_ready:
            try:
                # Read once for S3 upload — this is a one-time read, not the whole-request read
                with open(temp_file_path, "rb") as f:
                    pdf_bytes = f.read()

                pdf_url         = upload_pdf_to_s3(
                    file_bytes = pdf_bytes,
                    filename   = safe_filename,
                    level      = level,
                    subject    = resolved_subject,
                )
                storage_backend = "s3"
                print(f"[UPLOAD] S3 upload OK -> {pdf_url}")

            except Exception as s3_err:
                print(f"[UPLOAD] S3 failed, falling back to local: {s3_err}")
                pdf_url         = _save_local_from_path(temp_file_path, safe_filename)
                storage_backend = "local_fallback"
        else:
            pdf_url         = _save_local_from_path(temp_file_path, safe_filename)
            storage_backend = "local"
            print(f"[UPLOAD] S3 not configured — stored locally: {pdf_url}")

        now = datetime.utcnow()

        extra_meta = {
            "title":          file.filename,
            "course":         course,
            "level":          level,
            "subject":        resolved_subject,
            "chapter":        resolved_chapter,
            "section":        section        or "",
            "unit":           resolved_unit,
            "module":         resolved_module,
            "custom_heading": custom_heading or "",
            "uploaded_by":    admin["email"],
            "uploaded_at":    now.isoformat(),
            "pdf_url":        pdf_url        or "",
            "platform":       "cs_tutor",
        }

        # ── STEP 3: Kick off background processing and return immediately ─────
        background_tasks.add_task(
            process_pdf_background,
            temp_file_path,
            file.filename,
            extra_meta,
            enable_image_descriptions,
            admin["email"],
            course,
            resolved_subject,
            resolved_module,
            resolved_chapter,
            resolved_unit,
            section or "",
            custom_heading or "",
            pdf_url,
            storage_backend,
            safe_filename,
        )

        return UploadResponse(
            success           = True,
            message           = f"'{file.filename}' uploaded successfully. Processing started in background.",
            filename          = file.filename,
            storage_backend   = storage_backend,
            processing_status = "started",
        )

    except Exception as e:
        traceback.print_exc()
        print(f"[UPLOAD ERROR] {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ----------------------------------------------------------
# DELETE  ->  Pinecone  +  S3/local  +  Mongo docs  +  Mongo dashboard
# ----------------------------------------------------------

@router.delete("/{doc_id}", tags=["Admin Materials"])
async def delete_document(doc_id: str, admin=Depends(get_current_admin)):
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await docs_collection.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    filename        = doc.get("filename",        "")
    pdf_url         = doc.get("pdf_url",         "")
    safe_filename   = doc.get("safe_filename",   filename.replace(" ", "_"))
    storage_backend = doc.get("storage_backend", "local")

    report: Dict[str, Any] = {
        "doc_id":           doc_id,
        "filename":         filename,
        "pinecone_deleted": 0,
        "s3_deleted":       False,
        "local_deleted":    False,
        "mongo_docs":       False,
        "mongo_dashboard":  0,
        "errors":           [],
    }

    try:
        report["pinecone_deleted"] = await _delete_pinecone_by_source(filename)
    except Exception as e:
        report["errors"].append(f"Pinecone: {e}")

    if storage_backend == "s3" and pdf_url:
        try:
            ok = delete_pdf_from_s3(pdf_url)
            report["s3_deleted"] = ok
            if not ok:
                report["errors"].append("S3 delete returned False — file may not exist in bucket")
        except Exception as e:
            report["errors"].append(f"S3: {e}")
    else:
        local_path = os.path.join(UPLOAD_ROOT, safe_filename)
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                report["local_deleted"] = True
        except Exception as e:
            report["errors"].append(f"Local file: {e}")

    del_result           = await docs_collection.delete_one({"_id": obj_id})
    report["mongo_docs"] = del_result.deleted_count > 0

    dash_result               = await dashboard_collection.delete_many({"source_doc": doc_id})
    report["mongo_dashboard"] = dash_result.deleted_count

    return {"message": f"'{filename}' deleted successfully", "report": report}


async def _delete_pinecone_by_source(source_filename: str) -> int:
    """Delete all Pinecone vectors whose metadata.source == source_filename."""
    try:
        index_info = index.describe_index_stats()
        dim = index_info.get("dimension", 3072)
    except Exception:
        dim = 3072

    zero_vec      = [0.0] * dim
    total_deleted = 0

    while True:
        try:
            res = index.query(
                vector           = zero_vec,
                top_k            = 1000,
                include_metadata = False,
                filter           = {"source": {"$eq": source_filename}},
            )
        except Exception:
            break

        matches = res.get("matches") or []
        if not matches:
            break

        index.delete(ids=[m["id"] for m in matches])
        total_deleted += len(matches)

        if len(matches) < 1000:
            break

    return total_deleted


@router.get("/upload_health")
async def upload_service_health():
    from s3_service import debug_s3_config, create_bucket_if_not_exists

    s3_cfg = debug_s3_config()

    health: Dict[str, Any] = {
        "service":  "cs_tutor_upload",
        "platform": "CS Tutor",
        "status":   "operational",
        "storage":  "s3" if is_s3_configured() else "local",
        "s3_config": s3_cfg,
        "features": {
            "aws_s3":               is_s3_configured(),
            "docling_parser":       True,
            "pdfplumber_tables":    True,
            "enhanced_chunking":    True,
            "pinecone_delete":      True,
            "conversation_memory":  True,
            "gemini_summarisation": bool(getattr(settings, "GEMINI_API_KEY", "")),
            "background_processing": True,
        },
    }

    try:
        import docling, pdfplumber, fitz
        health["dependencies"] = "all_available"
    except ImportError as e:
        health["status"]       = "degraded"
        health["dependencies"] = f"missing: {e}"

    if is_s3_configured():
        try:
            bucket_ready = create_bucket_if_not_exists()
            health["s3_connectivity"] = (
                f"bucket '{s3_cfg['AWS_S3_BUCKET']}' ready"
                if bucket_ready else
                "bucket creation failed — check IAM permissions"
            )
        except Exception as e:
            health["s3_connectivity"] = f"error: {e}"

    return health


# ── Router must be included AFTER all routes are defined ─────────────────────
app.include_router(router)


# ============================================================
# AUTH ROUTES
# ============================================================

@app.post("/auth/signup")
async def signup(user: UserCreate):
    if await get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    plan = (user.plan or "free").lower()
    if plan not in ("free", "paid"):
        plan = "free"
    if plan == "paid" and not user.payment_id:
        raise HTTPException(status_code=400, detail="payment_id is required for paid plan")

    now = datetime.utcnow()
    await users_collection.insert_one({
        "email":               user.email,
        "password_hash":       hash_password(user.password),
        "name":                user.name,
        "phone":               user.phone,
        "cs_level":            user.cs_level,
        "cs_attempt":          user.cs_attempt,
        "role":                "student",
        "status":              "approved",
        "plan":                plan,
        "subscription_status": "active" if plan == "paid" else "free",
        "payment_id":          user.payment_id,
        "plan_activated_at":   now if plan == "paid" else None,
        "plan_expires_at": (
            datetime(now.year, now.month + 1 if now.month < 12 else 1, now.day)
            if plan == "paid" else None
        ),
        "created_at": now,
        "conversation_turns":   [],
        "conversation_summary": "",
        "total_turns":          0,
    })
    try:
        send_admin_signup_notification({
            **user.dict(),
            "plan": plan,
            "platform": "CS Tutor",
        })
    except Exception as e:
        print("Signup email to admin failed:", e)
    return {"message": "Signup successful."}


@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    if await get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    plan = (user.plan or "free").lower()
    now  = datetime.utcnow()

    await users_collection.insert_one({
        "email":               user.email,
        "password_hash":       hash_password(user.password),
        "name":                user.name,
        "phone":               user.phone,
        "cs_level":            user.cs_level,
        "cs_attempt":          user.cs_attempt,
        "role":                user.role,
        "status":              "approved",
        "plan":                plan,
        "subscription_status": "active" if plan == "paid" else "free",
        "payment_id":          user.payment_id,
        "created_at":          now,
        "conversation_turns":  [],
        "conversation_summary": "",
        "total_turns":         0,
    })

    try:
        send_admin_signup_notification({
            "name": user.name, "email": user.email, "phone": user.phone,
            "cs_level": user.cs_level, "cs_attempt": user.cs_attempt,
            "plan": plan, "platform": "CS Tutor",
        })
    except Exception as e:
        print(f"[register] Admin notification failed (non-fatal): {e}")

    token = create_access_token({"sub": user.email})
    return Token(access_token=token)


@app.post("/auth/login", response_model=Token)
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)
    if not user or user.get("status") != "approved":
        raise HTTPException(status_code=403, detail="Your account is pending admin approval.")
    pw_hash = user.get("password_hash") or user.get("password", "")
    if not verify_password(data.password, pw_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return Token(access_token=create_access_token({"sub": user["email"]}))


@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(
        email               = user["email"],
        role                = user.get("role", "student"),
        plan                = user.get("plan", "free"),
        subscription_status = user.get("subscription_status", user.get("plan", "free")),
    )


# ============================================================
# PASSWORD RESET (OTP)
# ============================================================

@app.post("/auth/forgot-password")
async def forgot_password(body: ForgotPasswordRequest):
    user = await get_user_by_email(body.email)
    if not user:
        return {"message": "If this email is registered, an OTP has been sent."}
    otp        = str(secrets.randbelow(900000) + 100000)
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    await users_collection.update_one(
        {"email": body.email},
        {"$set": {"reset_otp": otp, "reset_otp_expires": expires_at}},
    )
    try:
        send_password_reset_otp(email=body.email, otp=otp, name=user.get("name", "Student"))
    except Exception as e:
        print("OTP email failed:", e)
        raise HTTPException(status_code=500, detail="Failed to send OTP email.")
    return {"message": "If this email is registered, an OTP has been sent."}


@app.post("/auth/verify-otp")
async def verify_otp(body: VerifyOTPRequest):
    user = await get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid OTP or email.")
    stored_otp     = user.get("reset_otp")
    stored_expires = user.get("reset_otp_expires")
    if not stored_otp or not stored_expires:
        raise HTTPException(status_code=400, detail="No OTP requested. Please request a new one.")
    if datetime.utcnow() > stored_expires:
        raise HTTPException(status_code=400, detail="OTP has expired.")
    if stored_otp != body.otp.strip():
        raise HTTPException(status_code=400, detail="Incorrect OTP.")
    return {"message": "OTP verified."}


@app.post("/auth/reset-password")
async def reset_password(body: ResetPasswordRequest):
    user = await get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid request.")
    stored_otp     = user.get("reset_otp")
    stored_expires = user.get("reset_otp_expires")
    if not stored_otp or not stored_expires:
        raise HTTPException(status_code=400, detail="No OTP requested.")
    if datetime.utcnow() > stored_expires:
        raise HTTPException(status_code=400, detail="OTP has expired.")
    if stored_otp != body.otp.strip():
        raise HTTPException(status_code=400, detail="Incorrect OTP.")
    if len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    await users_collection.update_one(
        {"email": body.email},
        {"$set":   {"password_hash": hash_password(body.new_password)},
         "$unset": {"reset_otp": "", "reset_otp_expires": ""}},
    )
    return {"message": "Password reset successfully. You can now log in."}


# ============================================================
# ADMIN — STUDENT MANAGEMENT
# ============================================================

@app.get("/admin/students")
async def get_all_students(admin=Depends(get_current_admin)):
    students = []
    async for user in users_collection.find({"role": "student"}):
        user["_id"] = str(user["_id"])
        user.pop("password_hash", None)
        user.pop("password", None)
        user.pop("reset_otp", None)
        students.append(user)
    return students


@app.post("/admin/approve/{user_id}")
async def approve_student(user_id: str, admin=Depends(get_current_admin)):
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)}, {"$set": {"status": "approved"}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student approved"}


@app.post("/admin/reject/{user_id}")
async def reject_student(user_id: str, admin=Depends(get_current_admin)):
    await users_collection.delete_one({"_id": ObjectId(user_id)})
    return {"message": "Student rejected"}


# ============================================================
# ADMIN — CONVERSATION MEMORY MANAGEMENT
# ============================================================

@app.get("/admin/students/{user_id}/memory")
async def get_student_memory(user_id: str, admin=Depends(get_current_admin)):
    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    user = await users_collection.find_one(
        {"_id": obj_id},
        {
            "email": 1, "name": 1, "cs_level": 1, "cs_attempt": 1,
            "conversation_turns": 1, "conversation_summary": 1,
            "total_turns": 1, "summary_updated_at": 1,
        },
    )
    if not user:
        raise HTTPException(status_code=404, detail="Student not found")

    user["_id"] = str(user["_id"])
    return user


@app.delete("/admin/students/{user_id}/memory")
async def clear_student_memory(user_id: str, admin=Depends(get_current_admin)):
    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    result = await users_collection.update_one(
        {"_id": obj_id},
        {"$set": {
            "conversation_turns":   [],
            "conversation_summary": "",
            "total_turns":          0,
        }},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Conversation memory cleared successfully."}


# ============================================================
# ADMIN — DOCUMENTS LIST
# ============================================================

@app.get("/admin/documents", tags=["Admin"])
async def list_documents(admin=Depends(get_current_admin)):
    docs = []
    async for doc in docs_collection.find(
        {},
        {"_id": 1, "filename": 1, "level": 1, "subject": 1,
         "uploaded_at": 1, "total_vectors": 1, "pdf_url": 1, "processing_status": 1}
    ):
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return {"documents": docs}


@app.get("/admin/documents/grouped", tags=["Admin Materials"])
async def get_grouped_documents(admin=Depends(get_current_admin)):
    grouped: Dict[str, List[Any]] = {}
    async for doc in docs_collection.find().sort("uploaded_at", -1):
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("uploaded_at"), datetime):
            doc["uploaded_at"] = doc["uploaded_at"].isoformat()
        course = doc.get("course") or doc.get("level") or "Other"
        grouped.setdefault(course, [])
        grouped[course].append(doc)
    return grouped


# ============================================================
# CS DASHBOARD ROUTES
# ============================================================

@app.get("/dashboard/tree")
async def get_dashboard_tree(user=Depends(get_current_user)):
    tree: Dict[str, Any] = {}
    async for doc in dashboard_collection.find().sort("created_at", 1):
        doc["_id"] = str(doc["_id"])
        level   = (doc.get("level")   or "Others").strip()
        subject = (doc.get("subject") or "General").strip()
        module  = (doc.get("module")  or subject).strip()
        chapter = (doc.get("chapter") or doc.get("title", "General")).strip()

        tree.setdefault(level, {})
        tree[level].setdefault(subject, {})
        tree[level][subject].setdefault(module, {})
        tree[level][subject][module].setdefault(chapter, [])

        tree[level][subject][module][chapter].append({
            "_id":                doc["_id"],
            "title":              doc.get("title",              ""),
            "pdf_url":            doc.get("pdf_url",            ""),
            "video_url":          doc.get("video_url")          or None,
            "audio_url":          doc.get("audio_url")          or None,
            "simplified_pdf_url": doc.get("simplified_pdf_url") or None,
            "processing_status":  doc.get("processing_status")  or None,
            "status":             doc.get("processing_status")  or None,
            "chapter":            chapter,
            "unit":               doc.get("unit", ""),
        })
    return tree


@app.get("/dashboard", tags=["Dashboard"])
async def get_dashboard(user=Depends(get_current_user)):
    cs_level = user.get("cs_level", "")
    query: dict = {}
    if cs_level:
        query["level"] = cs_level

    items = []
    async for item in dashboard_collection.find(query):
        item["_id"] = str(item["_id"])
        items.append(item)

    grouped: Dict[str, Any] = {}
    for item in items:
        lvl  = item.get("level",   "Others")
        subj = item.get("subject", "General")
        mod  = item.get("module",  subj)

        grouped.setdefault(lvl, {})
        grouped[lvl].setdefault(subj, {})
        grouped[lvl][subj].setdefault(mod, [])
        grouped[lvl][subj][mod].append(item)

    return {"grouped": grouped, "total": len(items)}


@app.get("/dashboard/item/{item_id}", tags=["Dashboard"])
async def get_dashboard_item(item_id: str, user=Depends(get_current_user)):
    doc = await dashboard_collection.find_one({"_id": ObjectId(item_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Item not found")
    doc["_id"] = str(doc["_id"])
    return doc


@app.post("/dashboard/add")
async def add_dashboard_resource(
    level:     str = Form(...),
    subject:   str = Form(...),
    module:    str = Form(...),
    chapter:   str = Form(...),
    unit:      str = Form(...),
    title:     str = Form(...),
    pdf_url:   str = Form(...),
    video_url: str = Form(""),
    admin=Depends(get_current_admin),
):
    await dashboard_collection.insert_one({
        "level":      level,
        "subject":    subject,
        "module":     module,
        "chapter":    chapter,
        "unit":       unit,
        "title":      title,
        "pdf_url":    pdf_url,
        "video_url":  video_url,
        "platform":   "cs_tutor",
        "created_at": datetime.utcnow(),
    })
    return {"message": "Added successfully"}


# ============================================================
# PDF PROXY — serves S3 PDFs inline
# ============================================================

@app.get("/dashboard/pdf-proxy")
async def pdf_proxy(url: str, user=Depends(get_current_user)):
    if not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS URLs are supported")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch PDF: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch PDF: {str(e)}")

    return StreamingResponse(
        iter([resp.content]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline",
            "Content-Type":        "application/pdf",
            "Cache-Control":       "private, max-age=3600",
            "X-Frame-Options":     "SAMEORIGIN",
        },
    )


@app.get("/dashboard/audio-proxy")
async def audio_proxy(url: str, user=Depends(get_current_user)):
    if not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS URLs are supported")

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch audio: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch audio: {str(e)}")

    return StreamingResponse(
        iter([resp.content]),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment",
            "Content-Type":        "audio/mpeg",
            "Cache-Control":       "private, max-age=3600",
        },
    )


# ============================================================
# CHAT  (RAG + CONVERSATION MEMORY CHAINING)
# ============================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    try:
        req.message = expand_cs_abbreviations(req.message)

        if not await is_cs_related_question(req.message):
            return ChatResponse(
                answer=(
                    "This assistant is designed for Indian CS (Company Secretary) students. "
                    "Please ask a question related to CS topics such as Company Law, Securities Law, "
                    "Insolvency & Bankruptcy Code, FEMA, Secretarial Practice, Direct Tax, GST, "
                    "CS exams (CSEET / Executive / Professional), "
                    "or about this platform."
                ),
                sources=[],
            )

        memory       = await get_user_memory(user["email"])
        memory_block = build_memory_block(memory)

        query_embedding = await embed_single(
            enrich_query_for_rag(req.message)[:MAX_TEXT_LENGTH_FOR_EMBED]
        )
        detected_subj   = detect_subject(req.message)

        q_kwargs: Dict[str, Any] = {
            "vector":           query_embedding,
            "top_k":            20,
            "include_metadata": True,
        }

        pinecone_filter: dict = {}
        cs_level = user.get("cs_level", "")
        if detected_subj:
            pinecone_filter["subject"] = {"$eq": detected_subj}
        if cs_level:
            pinecone_filter["level"] = {"$eq": cs_level}
        if pinecone_filter:
            q_kwargs["filter"] = pinecone_filter

        res     = index.query(**q_kwargs)
        matches = res.get("matches") or []

        if len(matches) < 4 and detected_subj:
            fallback_filter = {"level": {"$eq": cs_level}} if cs_level else {}
            res     = index.query(
                vector=query_embedding, top_k=20, include_metadata=True,
                filter=fallback_filter if fallback_filter else None,
            )
            matches = res.get("matches") or []

        if len(matches) < 4:
            res     = index.query(vector=query_embedding, top_k=20, include_metadata=True)
            matches = res.get("matches") or []

        matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)
        q_len   = len(req.message.split())
        thr     = 0.35 if q_len <= 6 else 0.45 if q_len <= 15 else 0.52
        matches = [m for m in matches if m.get("score", 0) >= thr] or matches[:5]

        personal_context = build_personalized_layer(user)

        if not matches:
            system_prompt = (
                memory_block
                + personal_context + "\n\n"
                "You are a senior Indian Company Secretary (CS) faculty with deep expertise "
                "in ICSI exam preparation (CSEET, Executive, Professional).\n\n"

                "Language rule (MANDATORY):\n"
                "- Reply strictly in the SAME language as the student's question "
                "(English, Hindi, or Hinglish).\n\n"

                "Knowledge & safety rules:\n"
                "- Use your well-established CS / ICSI knowledge.\n"
                "- Do NOT guess exact section numbers, limits, or year-specific amendments "
                "unless you are certain.\n"
                "- If precise data is uncertain, explain the concept without risky figures.\n"
                "- Strictly avoid: guessing section numbers, adding unsupported facts, "
                "presenting assumptions as facts.\n\n"

                "Answer structure (EXAM-ORIENTED):\n"
                "1. Begin with a clear definition or core concept.\n"
                "2. Explain in logical steps using proper CS / ICSI terminology.\n"
                "3. Where relevant, mention the applicable Act, Section, Rule, or Case Law.\n"
                "4. Include ONE short exam-oriented or practical illustration if helpful.\n\n"

                "Exam guidance:\n"
                "- Add ONE short CS exam tip or common mistake to avoid.\n"
                "- Keep the answer concise, structured, and revision-friendly.\n\n"

                "Memory guidance:\n"
                "- If STUDENT MEMORY references topics the student studied before, "
                "connect this answer to their prior learning where relevant.\n"
                "- Do not repeat what the student already knows well per the summary."
            )

            answer = await call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": req.message},
            ])

            asyncio.create_task(
                save_turn_and_maybe_summarize(user["email"], req.message, answer, memory)
            )

            return ChatResponse(
                answer=answer,
                sources=[{
                    "doc_title": "General CS Knowledge (LLM based)",
                    "note":      "No match in uploaded docs",
                }],
            )

        context_blocks: List[str] = []
        sources: List[dict]       = []

        for m in sorted(matches, key=lambda m: m.get("score", 0), reverse=True):
            meta = m.get("metadata", {})
            text = meta.get("text", "")
            if not text:
                continue

            header = []
            if meta.get("doc_title"):  header.append(f"Document: {meta['doc_title']}")
            if meta.get("chapter"):    header.append(f"Chapter: {meta['chapter']}")
            if meta.get("topic"):      header.append(f"Topic: {meta['topic']}")
            if meta.get("page_start"): header.append(f"Page: {meta['page_start']}")
            context_blocks.append(f"{' | '.join(header)}\n{text}")

            sources.append({
                "doc_title":  meta.get("doc_title"),
                "source":     meta.get("source"),
                "page_start": meta.get("page_start"),
                "chapter":    meta.get("chapter"),
                "topic":      meta.get("topic"),
                "type":       meta.get("type", "text"),
            })

        trimmed, total = [], 0
        for b in context_blocks:
            if total + len(b) > 12000:
                break
            trimmed.append(b)
            total += len(b)
        context_str = "\n\n---\n\n".join(trimmed)

        if req.mode == "discussion":
            sys_p = (
                memory_block
                + personal_context + "\n\n"
                "You are an expert Indian CS (Company Secretary) tutor simulating a healthy "
                "academic discussion between two CS students preparing for ICSI exams.\n\n"

                "Language rules:\n"
                "- Reply strictly in the SAME language as the student's question "
                "(English, Hindi, or Hinglish).\n\n"

                "Discussion format rules:\n"
                "- Write the answer as a discussion between 'Student A:' and 'Student B:'.\n"
                "- Alternate clearly between Student A and Student B.\n"
                "- Provide at least 4 to 6 exchanges.\n\n"

                "Content rules (VERY IMPORTANT):\n"
                "- Explain concepts step-by-step in a teaching style.\n"
                "- Keep explanations exam-oriented as per ICSI expectations.\n"
                "- Refer to relevant Acts, Sections, Rules, and Case Laws where applicable.\n"
                "- Use simple intuition first, then technical / legal clarity.\n"
                "- Include 1 short practical or exam-oriented example if relevant.\n"
                "- Add a quick CS exam tip, memory aid, or common mistake to avoid.\n"
                "- Use the provided context as the PRIMARY source. Do NOT add unnecessary "
                "outside knowledge beyond standard CS principles.\n\n"

                "Memory guidance:\n"
                "- Reference the student's past learning from STUDENT MEMORY where it adds value.\n\n"

                f"Context:\n{context_str}"
            )
        else:
            sys_p = (
                memory_block
                + personal_context + "\n\n"
                "You are an expert Indian Company Secretary (CS) tutor preparing students "
                "for ICSI exams (CSEET, Executive, Professional).\n\n"

                "Language rule:\n"
                "- Reply strictly in the SAME language as the student's question "
                "(English, Hindi, or Hinglish).\n\n"

                "Answering style rules:\n"
                "- Answer using the context provided below as the PRIMARY source.\n"
                "- Do NOT add unnecessary outside knowledge beyond standard CS principles.\n"
                "- Keep the explanation clear, concise, and exam-oriented, in detail.\n"
                "- Start with a direct definition or core concept.\n"
                "- Then explain step-by-step, referencing Acts / Sections / Rules / Case Laws "
                "where relevant.\n"
                "- If applicable, cite a leading case law or NCLT / NCLAT / Supreme Court ruling briefly.\n"
                "- Include a short practical or exam-oriented example if helpful.\n"
                "- If tables or figures are present in the context, refer to them explicitly.\n\n"

                "Exam guidance:\n"
                "- Add one very short CS exam tip or a common mistake to avoid.\n"
                "- Avoid unnecessary storytelling or over-explanation.\n\n"

                "Memory guidance:\n"
                "- If STUDENT MEMORY references past topics, briefly connect to prior learning.\n"
                "- Do not repeat what the student already knows well (per the summary).\n\n"

                f"Context:\n{context_str}"
            )

        answer = await call_llm_with_chain(
            user_question       = req.message,
            context             = context_str,
            final_system_prompt = sys_p,
        )

        seen: set                 = set()
        clean_sources: List[dict] = []
        for s in sources:
            key = (s.get("doc_title"), s.get("page_start"))
            if key not in seen:
                seen.add(key)
                clean_sources.append(s)

        asyncio.create_task(
            save_turn_and_maybe_summarize(user["email"], req.message, answer, memory)
        )

        return ChatResponse(answer=answer, sources=clean_sources[:5])

    except Exception as e:
        traceback.print_exc()
        try:
            answer = await call_llm([
                {"role": "system", "content": "You are a helpful Indian CS (Company Secretary) tutor."},
                {"role": "user",   "content": req.message},
            ])
            return ChatResponse(
                answer=answer,
                sources=[{
                    "doc_title": "LLM fallback",
                    "note":      "Answered without document sources due to a system issue",
                }],
            )
        except Exception as inner_e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Chat error: {str(e)} | Fallback error: {str(inner_e)}"
            )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status":   "ok",
        "platform": "CS Tutor",
        "db":       settings.MONGO_DB,
        "index":    settings.PINECONE_INDEX,
    }
