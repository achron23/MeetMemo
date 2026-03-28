import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from uuid import uuid4
from app.services.supabase_client import (
    create_meeting,
    get_meeting,
    update_meeting_transcript,
    update_meeting_notes,
    update_meeting_email,
    list_meetings
)
from app.models.meeting import MeetingCreate, StructuredNotes, ActionItem


@pytest.fixture
def mock_supabase():
    with patch("app.services.supabase_client._get_client") as mock:
        yield mock


def test_create_meeting_success(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    meeting_id = uuid4()
    mock_response = Mock()
    mock_response.data = [{
        "id": str(meeting_id),
        "title": "Client Call",
        "contact_name": "John Doe",
        "contact_email": "john@example.com",
        "recorded_at": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat()
    }]
    mock_response.error = None

    mock_client.table.return_value.insert.return_value.execute.return_value = mock_response

    meeting_create = MeetingCreate(
        title="Client Call",
        contact_name="John Doe",
        contact_email="john@example.com"
    )

    result = create_meeting(meeting_create)

    assert result["success"] is True
    assert result["data"]["id"] == str(meeting_id)
    assert result["data"]["title"] == "Client Call"


def test_create_meeting_database_error(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = None
    mock_response.error = {"message": "Connection failed"}

    mock_client.table.return_value.insert.return_value.execute.return_value = mock_response

    meeting_create = MeetingCreate(title="Test")
    result = create_meeting(meeting_create)

    assert result["success"] is False
    assert "Connection failed" in result["error"]


def test_get_meeting_success(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    meeting_id = uuid4()
    mock_response = Mock()
    mock_response.data = [{
        "id": str(meeting_id),
        "title": "Test Meeting",
        "transcript": "Hello world",
        "recorded_at": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat()
    }]
    mock_response.error = None

    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_response

    result = get_meeting(str(meeting_id))

    assert result["success"] is True
    assert result["data"]["id"] == str(meeting_id)
    assert result["data"]["transcript"] == "Hello world"


def test_get_meeting_not_found(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = []
    mock_response.error = None

    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_response

    result = get_meeting(str(uuid4()))

    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_update_meeting_transcript_success(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = [{"id": str(uuid4()), "transcript": "Updated transcript"}]
    mock_response.error = None

    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

    result = update_meeting_transcript(str(uuid4()), "Updated transcript")

    assert result["success"] is True


def test_list_meetings_success(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = [
        {"id": str(uuid4()), "title": "Meeting 1", "recorded_at": datetime.now().isoformat(), "created_at": datetime.now().isoformat()},
        {"id": str(uuid4()), "title": "Meeting 2", "recorded_at": datetime.now().isoformat(), "created_at": datetime.now().isoformat()}
    ]
    mock_response.error = None

    mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    result = list_meetings()

    assert result["success"] is True
    assert len(result["data"]) == 2
