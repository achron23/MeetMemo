import pytest
from unittest.mock import Mock, patch
from app.services.claude_compose import compose_email
from app.models.meeting import StructuredNotes, Commitment, ActionItem


@pytest.fixture
def mock_anthropic():
    with patch("app.services.claude_compose.Anthropic") as mock:
        yield mock


@pytest.fixture
def sample_notes():
    return StructuredNotes(
        discussed=["Project timeline", "Budget allocation"],
        commitments=[
            Commitment(description="Send proposal", owner="us", due_date="2026-04-01")
        ],
        action_items=[
            ActionItem(description="Review contract", owner="client", priority="high")
        ],
        needs_from_them=["Signed agreement", "Budget approval"]
    )


def test_compose_email_success(mock_anthropic, sample_notes):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    mock_response = Mock()
    mock_response.content = [Mock(text="Dear John,\n\nThank you for the productive meeting...\n\nBest regards,\nAndre")]
    mock_response.usage.input_tokens = 200
    mock_response.usage.output_tokens = 100
    mock_client.messages.create.return_value = mock_response

    result = compose_email(
        structured_notes=sample_notes,
        contact_name="John Doe",
        sender_name="Andre",
        tone="professional"
    )

    assert result["success"] is True
    assert "Dear John" in result["data"]
    assert "Andre" in result["data"]
    assert len(result["data"]) > 50


def test_compose_email_custom_tone(mock_anthropic, sample_notes):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    mock_response = Mock()
    mock_response.content = [Mock(text="Hey John!\n\nGreat chatting today...")]
    mock_client.messages.create.return_value = mock_response

    result = compose_email(
        structured_notes=sample_notes,
        contact_name="John",
        sender_name="Andre",
        tone="friendly"
    )

    assert result["success"] is True
    assert result["data"] is not None


def test_compose_email_api_error(mock_anthropic, sample_notes):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = Exception("Rate limit exceeded")

    result = compose_email(
        structured_notes=sample_notes,
        contact_name="John",
        sender_name="Andre"
    )

    assert result["success"] is False
    assert "Rate limit" in result["error"]


def test_compose_email_missing_contact_name(sample_notes):
    result = compose_email(
        structured_notes=sample_notes,
        contact_name="",
        sender_name="Andre"
    )

    assert result["success"] is False
    assert "contact_name" in result["error"].lower()
