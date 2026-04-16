from __future__ import annotations

from datetime import UTC, datetime, timedelta

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

try:
    import bcrypt as _bcrypt
except Exception:  # pragma: no cover
    _bcrypt = None

# Use a hash that doesn't rely on the bcrypt backend (which may reject long passwords).
# Keep bcrypt verification for legacy hashes if they already exist in the DB.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if hashed_password.startswith(_BCRYPT_PREFIXES):
        if _bcrypt is None:
            return False
        pw = plain_password.encode("utf-8")
        if len(pw) > 72:
            pw = pw[:72]
        return _bcrypt.checkpw(pw, hashed_password.encode("utf-8"))

    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(*, user_id: str) -> str:
    now = datetime.now(UTC)
    exp = now + timedelta(minutes=settings.jwt_access_ttl_minutes)
    payload = {
        "sub": user_id,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def create_refresh_token(*, user_id: str) -> str:
    now = datetime.now(UTC)
    exp = now + timedelta(days=settings.jwt_refresh_ttl_days)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
