from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal
from app.models.meeting import StructuredNotes
from app.services.claude_compose import compose_email
from app.services.supabase_client import update_meeting_email

router = APIRouter(prefix="/api", tags=["composition"])


class ComposeRequest(BaseModel):
    meeting_id: str
    structured_notes: StructuredNotes
    contact_name: str = Field(..., min_length=1)
    sender_name: str = Field(..., min_length=1)
    tone: Literal["formal", "professional", "friendly"] = "professional"


@router.post("/compose")
async def compose_followup_email(request: ComposeRequest):
    """
    Compose a follow-up email from structured notes using Claude.

    Args:
        request: Meeting ID, structured notes, contact/sender names, tone

    Returns:
        Email draft text
    """
    # Compose email using Claude
    compose_result = compose_email(
        structured_notes=request.structured_notes,
        contact_name=request.contact_name,
        sender_name=request.sender_name,
        tone=request.tone
    )

    if not compose_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email composition failed: {compose_result['error']}"
        )

    email_draft = compose_result["data"]

    # Update meeting in database
    update_result = update_meeting_email(request.meeting_id, email_draft)

    if not update_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database update failed: {update_result['error']}"
        )

    return {
        "meeting_id": request.meeting_id,
        "email_draft": email_draft
    }
