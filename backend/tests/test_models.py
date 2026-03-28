import pytest
from pydantic import ValidationError
from app.models.meeting import StructuredNotes, Commitment, ActionItem


def test_structured_notes_valid():
    data = {
        "discussed": ["Budget planning", "Timeline review"],
        "commitments": [
            {"description": "Send proposal", "owner": "us", "due_date": "2026-04-01"}
        ],
        "action_items": [
            {
                "description": "Review contract",
                "owner": "client",
                "priority": "high"
            }
        ],
        "needs_from_them": ["Signed NDA", "Budget approval"]
    }
    notes = StructuredNotes(**data)
    assert len(notes.discussed) == 2
    assert notes.commitments[0].owner == "us"
    assert notes.action_items[0].priority == "high"


def test_structured_notes_missing_required_fields():
    with pytest.raises(ValidationError) as exc_info:
        StructuredNotes(discussed=[], commitments=[])
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("action_items",) for e in errors)
    assert any(e["loc"] == ("needs_from_them",) for e in errors)


def test_commitment_without_due_date():
    commitment = Commitment(description="Follow up", owner="us")
    assert commitment.due_date is None
    assert commitment.owner == "us"


def test_action_item_invalid_priority():
    with pytest.raises(ValidationError):
        ActionItem(
            description="Test",
            owner="client",
            priority="urgent"  # Invalid - must be high/medium/low
        )


def test_meeting_model_with_all_fields():
    from uuid import uuid4
    from datetime import datetime
    from app.models.meeting import Meeting

    meeting_id = uuid4()
    now = datetime.now()

    structured_notes = StructuredNotes(
        discussed=["Topic 1"],
        commitments=[Commitment(description="Task", owner="us")],
        action_items=[ActionItem(description="Action", owner="client", priority="medium")],
        needs_from_them=["Feedback"]
    )

    meeting = Meeting(
        id=meeting_id,
        title="Client Call",
        contact_name="John Doe",
        contact_email="john@example.com",
        recorded_at=now,
        audio_file_path="/audio/meeting.mp3",
        transcript="Hello world",
        structured_notes=structured_notes,
        email_draft="Dear John...",
        created_at=now
    )

    assert meeting.id == meeting_id
    assert meeting.title == "Client Call"
    assert meeting.structured_notes.discussed[0] == "Topic 1"


def test_meeting_model_minimal_fields():
    from uuid import uuid4
    from datetime import datetime
    from app.models.meeting import Meeting

    meeting = Meeting(
        id=uuid4(),
        recorded_at=datetime.now(),
        created_at=datetime.now()
    )

    assert meeting.title is None
    assert meeting.transcript is None
    assert meeting.structured_notes is None
