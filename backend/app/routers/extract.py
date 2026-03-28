from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.claude_extraction import extract_structured_notes
from app.services.supabase_client import update_meeting_notes

router = APIRouter(prefix="/api", tags=["extraction"])


class ExtractRequest(BaseModel):
    meeting_id: str
    transcript: str = Field(..., min_length=1)


@router.post("/extract")
async def extract_meeting_notes(request: ExtractRequest):
    """
    Extract structured notes from transcript using Claude.

    Args:
        request: Meeting ID and transcript text

    Returns:
        Structured notes (discussed, commitments, action_items, needs_from_them)
    """
    # Extract structured notes using Claude
    extract_result = extract_structured_notes(request.transcript)

    if not extract_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {extract_result['error']}"
        )

    structured_notes = extract_result["data"]

    # Update meeting in database
    update_result = update_meeting_notes(request.meeting_id, structured_notes)

    if not update_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database update failed: {update_result['error']}"
        )

    return {
        "meeting_id": request.meeting_id,
        "structured_notes": structured_notes.model_dump()
    }
