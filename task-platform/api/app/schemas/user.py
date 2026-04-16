from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserMeOut(BaseModel):
    id: str
    email: EmailStr
    is_admin: bool
    is_active: bool

    class Config:
        from_attributes = True


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8)


class AdminUserOut(BaseModel):
    id: str
    email: EmailStr
    is_admin: bool
    is_active: bool

    class Config:
        from_attributes = True


class AdminUserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    is_admin: bool = False


class AdminUserUpdate(BaseModel):
    is_admin: bool | None = None
    is_active: bool | None = None


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=8)


class UserProjectsOut(BaseModel):
    project_ids: list[str]


class UserProjectsUpdate(BaseModel):
    project_ids: list[str]


class ProjectUsersOut(BaseModel):
    user_ids: list[str]


class ProjectUsersUpdate(BaseModel):
    user_ids: list[str]
