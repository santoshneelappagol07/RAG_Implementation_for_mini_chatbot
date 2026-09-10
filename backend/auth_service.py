"""
auth_service.py
---------------
Email OTP authentication with in-memory session management.

Flow:
  1. Client calls POST /auth/send-otp  → OTP emailed to user
  2. Client calls POST /auth/verify-otp → receives a session token
  3. Client passes Authorization: Bearer <token> on all protected routes
  4. Session auto-expires after SESSION_EXPIRY_SECONDS (default 1 hour)

OTP and session data are stored in-memory (no Redis required).
"""

import logging
import random
import smtplib
import time
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    from backend.config import (
        SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL,
        OTP_EXPIRY_SECONDS, SESSION_EXPIRY_SECONDS,
    )
except ImportError:
    from config import (
        SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL,
        OTP_EXPIRY_SECONDS, SESSION_EXPIRY_SECONDS,
    )

logger = logging.getLogger(__name__)

# HTTPBearer scheme — renders the lock (Authorize) button in Swagger UI
_bearer_scheme = HTTPBearer(auto_error=False)

# ── In-memory stores ────────────────────────────────────────────────────
# Key: email → { otp: str, created_at: float }
_otp_store: Dict[str, Dict] = {}

# Key: token (UUID string) → { email: str, created_at: float }
_session_store: Dict[str, Dict] = {}


# ── OTP helpers ─────────────────────────────────────────────────────────

def generate_otp(email: str) -> str:
    """
    Generate a 6-digit OTP for the given email.
    Overwrites any existing OTP for the same email (rate-limit friendly).
    """
    _cleanup_expired_otps()
    otp = f"{random.randint(100000, 999999)}"
    _otp_store[email.lower()] = {
        "otp": otp,
        "created_at": time.time(),
    }
    logger.info("OTP generated for %s (expires in %ds)", email, OTP_EXPIRY_SECONDS)
    return otp


def verify_otp(email: str, otp: str) -> bool:
    """
    Validate the OTP for the given email.
    Returns True if correct and not expired, False otherwise.
    Consumed on successful verification (one-time use).
    """
    _cleanup_expired_otps()
    key = email.lower()
    record = _otp_store.get(key)
    if record is None:
        return False

    elapsed = time.time() - record["created_at"]
    if elapsed > OTP_EXPIRY_SECONDS:
        _otp_store.pop(key, None)
        return False

    if record["otp"] != otp.strip():
        return False

    # Consume the OTP after successful verification
    _otp_store.pop(key, None)
    return True


def _cleanup_expired_otps() -> None:
    """Remove expired OTPs from memory."""
    now = time.time()
    expired = [
        email for email, rec in _otp_store.items()
        if now - rec["created_at"] > OTP_EXPIRY_SECONDS
    ]
    for email in expired:
        del _otp_store[email]


# ── Session helpers ─────────────────────────────────────────────────────

def create_session(email: str) -> str:
    """
    Create a new session token for the authenticated email.
    Returns the UUID token string.
    """
    _cleanup_expired_sessions()
    token = str(uuid.uuid4())
    _session_store[token] = {
        "email": email.lower(),
        "created_at": time.time(),
    }
    logger.info("Session created for %s (expires in %ds)", email, SESSION_EXPIRY_SECONDS)
    return token


def validate_session(token: str) -> Optional[str]:
    """
    Validate a session token.
    Returns the associated email if valid and not expired, else None.
    """
    _cleanup_expired_sessions()
    record = _session_store.get(token)
    if record is None:
        return None

    elapsed = time.time() - record["created_at"]
    if elapsed > SESSION_EXPIRY_SECONDS:
        _session_store.pop(token, None)
        return None

    return record["email"]


def get_session_remaining(token: str) -> Optional[float]:
    """
    Returns the remaining session lifetime in seconds, or None if expired/invalid.
    """
    record = _session_store.get(token)
    if record is None:
        return None
    remaining = SESSION_EXPIRY_SECONDS - (time.time() - record["created_at"])
    return max(0.0, remaining)


def invalidate_session(token: str) -> bool:
    """
    Invalidate (logout) a session token.
    Returns True if the token existed and was removed.
    """
    return _session_store.pop(token, None) is not None


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions from memory."""
    now = time.time()
    expired = [
        token for token, rec in _session_store.items()
        if now - rec["created_at"] > SESSION_EXPIRY_SECONDS
    ]
    for token in expired:
        del _session_store[token]


# ── SMTP email sending ─────────────────────────────────────────────────

def send_otp_email(recipient_email: str, otp: str) -> None:
    """
    Send a formatted OTP email via SMTP/TLS.
    Raises RuntimeError if SMTP credentials are not configured or sending fails.
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        raise RuntimeError(
            "SMTP credentials not configured. Set SMTP_USERNAME and SMTP_PASSWORD in .env"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Your Login OTP: {otp}"
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = recipient_email

    # Plain-text fallback
    text_body = (
        f"Your one-time password (OTP) is: {otp}\n\n"
        f"This OTP is valid for {OTP_EXPIRY_SECONDS // 60} minute(s).\n"
        f"Do not share this code with anyone.\n\n"
        f"If you did not request this, please ignore this email."
    )

    # HTML email body
    html_body = f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; padding: 40px;">
      <div style="max-width: 480px; margin: auto; background: #ffffff; border-radius: 12px;
                  box-shadow: 0 4px 24px rgba(0,0,0,0.08); overflow: hidden;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 32px; text-align: center;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">PDF Chatbot</h1>
          <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0;">Secure Login Verification</p>
        </div>
        <div style="padding: 32px; text-align: center;">
          <p style="color: #555; font-size: 15px; margin-bottom: 24px;">
            Use the following one-time password to complete your login:
          </p>
          <div style="background: #f0f4ff; border: 2px dashed #667eea; border-radius: 8px;
                      padding: 20px; display: inline-block; margin-bottom: 24px;">
            <span style="font-size: 36px; font-weight: 700; letter-spacing: 8px;
                         color: #333; font-family: 'Courier New', monospace;">{otp}</span>
          </div>
          <p style="color: #888; font-size: 13px;">
            This code expires in <strong>{OTP_EXPIRY_SECONDS // 60} minute(s)</strong>.
          </p>
          <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;">
          <p style="color: #aaa; font-size: 12px;">
            If you didn't request this code, you can safely ignore this email.
          </p>
        </div>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, recipient_email, msg.as_string())
        logger.info("OTP email sent to %s", recipient_email)
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed - check SMTP_USERNAME and SMTP_PASSWORD")
        raise RuntimeError("SMTP authentication failed. Check your email credentials.")
    except Exception as e:
        logger.error("Failed to send OTP email to %s: %s", recipient_email, e)
        raise RuntimeError(f"Failed to send OTP email: {str(e)}")


# ── FastAPI Dependency ──────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> str:
    """
    FastAPI dependency: validates the Bearer token from the Authorization header.
    Returns the authenticated user's email address.
    Raises HTTP 401 if the token is missing, invalid, or expired.
    """
    if credentials is None:
        logger.warning("AUTH DENIED: No Authorization header / Bearer token provided")
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Send OTP via /auth/send-otp first.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    logger.info("AUTH CHECK: token=%s..., active_sessions=%d", token[:8], len(_session_store))
    email = validate_session(token)
    if email is None:
        logger.warning("AUTH DENIED: Token not found in session store (store has %d sessions)", len(_session_store))
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please re-authenticate via /auth/send-otp.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("AUTH OK: %s", email)
    return email

