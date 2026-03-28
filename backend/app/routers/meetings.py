from fastapi import APIRouter, HTTPException, status
from app.models.meeting import MeetingCreate
from app.services.supabase_client import (
    create_meeting,
    get_meeting,
    list_meetings
)

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


@router.get("")
async def get_meetings():
    """
    List all meetings ordered by created_at desc.

    Returns:
        List of meeting summaries
    """
    result = list_meetings()

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return result["data"]


@router.get("/{meeting_id}")
async def get_meeting_by_id(meeting_id: str):
    """
    Get a specific meeting by ID.

    Args:
        meeting_id: UUID of the meeting

    Returns:
        Full meeting data including transcript, notes, email
    """
    result = get_meeting(meeting_id)

    if not result["success"]:
        if "not found" in result["error"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return result["data"]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_meeting(meeting: MeetingCreate):
    """
    Create a new meeting record.

    Args:
        meeting: Meeting creation data

    Returns:
        Created meeting with ID
    """
    result = create_meeting(meeting)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return result["data"]
