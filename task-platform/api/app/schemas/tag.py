from __future__ import annotations

from pydantic import BaseModel


class TagOut(BaseModel):
    id: str
    name: str

    class Config:
        from_attributes = True
