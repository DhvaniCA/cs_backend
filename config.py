from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database — CS platform uses its own DB and Pinecone index
    MONGO_URI: str
    MONGO_DB: str = "cs-vector-db-index"        # ← CS-specific DB

    # Auth
    JWT_SECRET: str
    JWT_ALGO: str = "HS256"

    # Pinecone — use a SEPARATE index from CA
    PINECONE_API_KEY: str
    PINECONE_INDEX: str                          # e.g. "cs-tutor-index"

    # OpenAI
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4.1"
    EMBEDDING_MODEL: str = "text-embedding-3-large"

    # Gemini — cheap async conversation summarisation
    # Get a free key at: https://aistudio.google.com/app/apikey
    # Leave blank to disable (chat still works fine without it)
    GEMINI_API_KEY: str = ""

    # Frontend
    FRONTEND_ORIGIN: str = "*"

    # Email
    EMAIL_HOST: str
    EMAIL_PORT: int = 587
    EMAIL_USERNAME: str
    EMAIL_PASSWORD: str
    EMAIL_FROM: str
    ADMIN_EMAIL: str

    # Razorpay
    RAZORPAY_KEY_ID: str
    RAZORPAY_KEY_SECRET: str

    # ── AWS S3 ────────────────────────────────────────────────────────────────
    # All four fields must be set in .env for S3 to be used.
    # If any are missing the app falls back to local file storage silently.
    AWS_ACCESS_KEY_ID:     Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION:            str           = "ap-south-1"
    AWS_S3_BUCKET:         Optional[str] = None
    # Set to "false" when your bucket has ACLs disabled (new AWS default).
    # When false, 7-day presigned URLs are generated instead.

    # ── Optional tuning ───────────────────────────────────────────────────────
    EMBED_BATCH_SIZE:          int   = 12
    EMBED_TIMEOUT_SECS:        int   = 120
    EMBED_MAX_RETRIES:         int   = 3
    EMBED_BACKOFF_BASE:        float = 1.8
    MAX_TEXT_LENGTH_FOR_EMBED: int   = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
