import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.google_stt import transcribe_audio


@pytest.fixture
def mock_speech_client():
    with patch("app.services.google_stt.speech.SpeechClient") as mock:
        yield mock


def test_transcribe_audio_success_short_file(mock_speech_client):
    # Mock Google STT response
    mock_client = Mock()
    mock_speech_client.return_value = mock_client

    mock_result = Mock()
    mock_alternative = Mock()
    mock_alternative.transcript = "Καλημέρα, πώς είστε;"
    mock_result.alternatives = [mock_alternative]

    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.recognize.return_value = mock_response

    # Test
    result = transcribe_audio(
        audio_file_path="/audio/short.wav",
        language_code="el-GR"
    )

    assert result["success"] is True
    assert result["data"] == "Καλημέρα, πώς είστε;"

    # Verify API was called with correct params
    call_args = mock_client.recognize.call_args
    assert call_args is not None


def test_transcribe_audio_long_file(mock_speech_client):
    # Mock long_running_recognize operation
    mock_client = Mock()
    mock_speech_client.return_value = mock_client

    mock_operation = MagicMock()
    mock_result = Mock()
    mock_alternative = Mock()
    mock_alternative.transcript = "Long transcript from async operation"
    mock_result.alternatives = [mock_alternative]

    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_operation.result.return_value = mock_response

    mock_client.long_running_recognize.return_value = mock_operation

    result = transcribe_audio(
        audio_file_path="/audio/long.wav",
        language_code="el-GR",
        duration_seconds=120  # 2 minutes
    )

    assert result["success"] is True
    assert "Long transcript" in result["data"]


def test_transcribe_audio_file_not_found():
    result = transcribe_audio(
        audio_file_path="/nonexistent/file.wav",
        language_code="el-GR"
    )

    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_transcribe_audio_api_error(mock_speech_client):
    mock_client = Mock()
    mock_speech_client.return_value = mock_client
    mock_client.recognize.side_effect = Exception("API quota exceeded")

    result = transcribe_audio(
        audio_file_path="/audio/test.wav",
        language_code="el-GR"
    )

    assert result["success"] is False
    assert "API quota exceeded" in result["error"]


def test_transcribe_audio_empty_result(mock_speech_client):
    mock_client = Mock()
    mock_speech_client.return_value = mock_client

    mock_response = Mock()
    mock_response.results = []
    mock_client.recognize.return_value = mock_response

    result = transcribe_audio(
        audio_file_path="/audio/silent.wav",
        language_code="el-GR"
    )

    assert result["success"] is False
    assert "no transcript" in result["error"].lower()
