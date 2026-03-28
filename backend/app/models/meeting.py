from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field
from uuid import UUID


class Commitment(BaseModel):
    description: str
    owner: str
    due_date: Optional[str] = None


class ActionItem(BaseModel):
    description: str
    owner: str
    due_date: Optional[str] = None
    priority: Literal["high", "medium", "low"]


class StructuredNotes(BaseModel):
    discussed: list[str]
    commitments: list[Commitment]
    action_items: list[ActionItem]
    needs_from_them: list[str]


class MeetingCreate(BaseModel):
    title: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None


class Meeting(BaseModel):
    id: UUID
    title: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    recorded_at: datetime
    audio_file_path: Optional[str] = None
    transcript: Optional[str] = None
    structured_notes: Optional[StructuredNotes] = None
    email_draft: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
