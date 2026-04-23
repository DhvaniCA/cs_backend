"""
payment_router.py — CS Tutor Platform
--------------------------------------
Razorpay integration using httpx (no extra SDK needed).

Mount in main.py:
    from payment_router import router as payment_router
    app.include_router(payment_router, prefix="/payments", tags=["payments"])

Required .env:
    RAZORPAY_KEY_ID=rzp_test_XXXXXXXXXXXXXXX
    RAZORPAY_KEY_SECRET=your_secret_here
"""

import hmac
import hashlib
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

RAZORPAY_BASE = "https://api.razorpay.com/v1"

# CS exam levels used for order receipts
CS_LEVELS = {"CSEET", "Executive", "Professional"}


# ── Schemas ──────────────────────────────────────────────────────────────────

class CreateOrderRequest(BaseModel):
    amount:   int           # in paise  e.g. 49900 → ₹499
    currency: str = "INR"
    plan:     str           # "CSEET" | "Executive" | "Professional"


class CreateOrderResponse(BaseModel):
    order_id: str
    amount:   int
    currency: str


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id:   str
    razorpay_payment_id: str
    razorpay_signature:  str


# ── Helper ───────────────────────────────────────────────────────────────────

def _rzp_auth() -> tuple:
    key_id     = getattr(settings, "RAZORPAY_KEY_ID",     None)
    key_secret = getattr(settings, "RAZORPAY_KEY_SECRET", None)
    if not key_id or not key_secret:
        raise HTTPException(
            status_code=500,
            detail="RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set in .env",
        )
    return key_id, key_secret


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/create-order", response_model=CreateOrderResponse)
async def create_order(body: CreateOrderRequest):
    """
    Step 1 — create a Razorpay order for a CS Tutor subscription plan.
    Frontend gets the order_id and passes it to the Razorpay checkout modal.
    """
    key_id, key_secret = _rzp_auth()

    # Sanitise plan label used in receipt (max 40 chars for Razorpay)
    safe_plan = body.plan.lower().replace(" ", "_")[:20]

    payload = {
        "amount":   body.amount,
        "currency": body.currency,
        "receipt":  f"cs_{safe_plan}",
        "notes":    {"plan": body.plan, "platform": "cs_tutor"},
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{RAZORPAY_BASE}/orders",
                json=payload,
                auth=(key_id, key_secret),
            )

        if resp.status_code != 200:
            logger.error("Razorpay create-order %s: %s", resp.status_code, resp.text)
            error_detail = resp.json().get("error", {}).get("description", resp.text)
            raise HTTPException(status_code=502, detail=f"Razorpay: {error_detail}")

        data = resp.json()
        return CreateOrderResponse(
            order_id=data["id"],
            amount=data["amount"],
            currency=data["currency"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in create_order")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/verify")
async def verify_payment(body: VerifyPaymentRequest):
    """
    Step 2 — verify HMAC-SHA256 signature returned by Razorpay.
    Call this after the checkout modal fires its success callback.
    Returns {"status": "verified", "payment_id": "..."} on success.
    """
    _, key_secret = _rzp_auth()

    payload_str = f"{body.razorpay_order_id}|{body.razorpay_payment_id}"
    expected = hmac.new(
        key_secret.encode("utf-8"),
        payload_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, body.razorpay_signature):
        raise HTTPException(
            status_code=400,
            detail="Payment signature mismatch — possible tampering.",
        )

    return {"status": "verified", "payment_id": body.razorpay_payment_id}
