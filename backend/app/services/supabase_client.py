import os
from typing import TypedDict, Any
from supabase import create_client, Client
from app.models.meeting import MeetingCreate, StructuredNotes


class Result(TypedDict, total=False):
    success: bool
    data: Any
    error: str | None


def _get_client() -> Client:
    """Get configured Supabase client."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    return create_client(url, key)


def create_meeting(meeting_create: MeetingCreate) -> Result:
    """
    Create a new meeting record in Supabase.

    Args:
        meeting_create: Meeting creation data

    Returns:
        Result with success/data (created meeting) or success/error
    """
    try:
        client = _get_client()

        data = meeting_create.model_dump(exclude_none=True)

        response = client.table("meetings").insert(data).execute()

        if response.error:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {response.error['message']}"
            }

        if not response.data or len(response.data) == 0:
            return {
                "success": False,
                "data": None,
                "error": "No data returned from insert"
            }

        return {
            "success": True,
            "data": response.data[0],
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Supabase error: {str(e)}"
        }


def get_meeting(meeting_id: str) -> Result:
    """
    Get a meeting by ID.

    Args:
        meeting_id: UUID of the meeting

    Returns:
        Result with success/data (meeting) or success/error
    """
    try:
        client = _get_client()

        response = (
            client.table("meetings")
            .select("*")
            .eq("id", meeting_id)
            .single()
            .execute()
        )

        if response.error:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {response.error['message']}"
            }

        if not response.data:
            return {
                "success": False,
                "data": None,
                "error": f"Meeting not found: {meeting_id}"
            }

        return {
            "success": True,
            "data": response.data[0] if isinstance(response.data, list) else response.data,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Supabase error: {str(e)}"
        }


def update_meeting_transcript(meeting_id: str, transcript: str) -> Result:
    """
    Update meeting with transcript.

    Args:
        meeting_id: UUID of the meeting
        transcript: Transcribed text

    Returns:
        Result with success/data or success/error
    """
    try:
        client = _get_client()

        response = (
            client.table("meetings")
            .update({"transcript": transcript})
            .eq("id", meeting_id)
            .execute()
        )

        if response.error:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {response.error['message']}"
            }

        return {
            "success": True,
            "data": response.data[0] if response.data else None,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Supabase error: {str(e)}"
        }


def update_meeting_notes(meeting_id: str, structured_notes: StructuredNotes) -> Result:
    """
    Update meeting with structured notes.

    Args:
        meeting_id: UUID of the meeting
        structured_notes: Extracted meeting notes

    Returns:
        Result with success/data or success/error
    """
    try:
        client = _get_client()

        notes_dict = structured_notes.model_dump()

        response = (
            client.table("meetings")
            .update({"structured_notes": notes_dict})
            .eq("id", meeting_id)
            .execute()
        )

        if response.error:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {response.error['message']}"
            }

        return {
            "success": True,
            "data": response.data[0] if response.data else None,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Supabase error: {str(e)}"
        }


def update_meeting_email(meeting_id: str, email_draft: str) -> Result:
    """
    Update meeting with email draft.

    Args:
        meeting_id: UUID of the meeting
        email_draft: Composed email text

    Returns:
        Result with success/data or success/error
    """
    try:
        client = _get_client()

        response = (
            client.table("meetings")
            .update({"email_draft": email_draft})
            .eq("id", meeting_id)
            .execute()
        )

        if response.error:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {response.error['message']}"
            }

        return {
            "success": True,
            "data": response.data[0] if response.data else None,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Supabase error: {str(e)}"
        }


def list_meetings(limit: int = 50) -> Result:
    """
    List all meetings ordered by created_at desc.

    Args:
        limit: Maximum number of meetings to return

    Returns:
        Result with success/data (list of meetings) or success/error
    """
    try:
        client = _get_client()

        response = (
            client.table("meetings")
            .select("id, title, contact_name, recorded_at, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        if response.error:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {response.error['message']}"
            }

        return {
            "success": True,
            "data": response.data,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Supabase error: {str(e)}"
        }
