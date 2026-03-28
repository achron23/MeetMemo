import pytest
from unittest.mock import Mock, patch
from app.services.claude_extraction import extract_structured_notes
from app.models.meeting import StructuredNotes


@pytest.fixture
def mock_anthropic():
    with patch("app.services.claude_extraction.Anthropic") as mock, \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
        yield mock


def test_extract_structured_notes_success(mock_anthropic):
    # Mock Claude response
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    mock_response = Mock()
    mock_response.content = [
        Mock(text='{"discussed": ["Budget"], "commitments": [], "action_items": [{"description": "Review", "owner": "client", "priority": "high"}], "needs_from_them": ["Approval"]}')
    ]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_response

    # Test
    transcript = "We discussed the budget. Client will review."
    result = extract_structured_notes(transcript)

    assert result["success"] is True
    assert isinstance(result["data"], StructuredNotes)
    assert len(result["data"].discussed) == 1
    assert result["data"].discussed[0] == "Budget"
    assert len(result["data"].action_items) == 1


def test_extract_structured_notes_invalid_json(mock_anthropic):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    mock_response = Mock()
    mock_response.content = [Mock(text='invalid json')]
    mock_client.messages.create.return_value = mock_response

    result = extract_structured_notes("transcript")

    assert result["success"] is False
    assert "JSON" in result["error"]


def test_extract_structured_notes_api_error(mock_anthropic):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = Exception("API Error")

    result = extract_structured_notes("transcript")

    assert result["success"] is False
    assert "API Error" in result["error"]


def test_extract_structured_notes_empty_transcript(mock_anthropic):
    result = extract_structured_notes("")

    assert result["success"] is False
    assert "empty" in result["error"].lower()
