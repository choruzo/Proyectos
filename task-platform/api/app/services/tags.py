from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.tag import Tag


def get_or_create_tag(db: Session, name: str) -> Tag:
    normalized = name.strip()
    tag = db.scalar(select(Tag).where(Tag.name == normalized))
    if tag is None:
        tag = Tag(name=normalized)
        db.add(tag)
        db.flush()
    return tag
