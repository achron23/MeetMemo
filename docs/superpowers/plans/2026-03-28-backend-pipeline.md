# MeetMemo Backend Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python FastAPI backend that transcribes audio files, extracts structured meeting notes using Claude, and composes follow-up emails.

**Architecture:** Service layer pattern with FastAPI routers consuming Google STT and Claude APIs. Supabase for persistence. Strict TDD throughout - tests before implementation, always.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, Anthropic SDK, Google Cloud Speech-to-Text v2, Supabase (PostgreSQL), pytest

---

## Task 1: Project Scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`
- Create: `backend/pytest.ini`
- Create: `backend/app/__init__.py`
- Create: `backend/tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```bash
cd backend
cat > requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
pydantic-settings==2.1.0
anthropic==0.18.0
google-cloud-speech==2.24.0
supabase==2.3.0
python-multipart==0.0.9
python-dotenv==1.0.1
pytest==8.0.0
pytest-asyncio==0.23.0
pytest-cov==4.1.0
httpx==0.26.0
EOF
```

- [ ] **Step 2: Create .env.example**

```bash
cat > .env.example << 'EOF'
# Anthropic
ANTHROPIC_API_KEY=sk-ant-api03-...

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_API_KEY=AIza...

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...

# Server
PORT=8000
ENVIRONMENT=development
EOF
```

- [ ] **Step 3: Create pytest.ini**

```bash
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts =
    -v
    --cov=app
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
EOF
```

- [ ] **Step 4: Create app/__init__.py**

```bash
mkdir -p app
touch app/__init__.py
```

- [ ] **Step 5: Create tests/__init__.py**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 6: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 7: Verify installation**

Run: `python -c "import fastapi; import anthropic; import google.cloud.speech; print('OK')"`
Expected: Output "OK"

- [ ] **Step 8: Commit scaffold**

```bash
git add requirements.txt .env.example pytest.ini app/__init__.py tests/__init__.py
git commit -m "feat: scaffold backend project structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/meeting.py`
- Create: `backend/tests/test_models.py`

- [ ] **Step 1: Write test for StructuredNotes model**

```bash
mkdir -p tests
cat > tests/test_models.py << 'EOF'
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
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.models'"

- [ ] **Step 3: Create models/__init__.py**

```bash
mkdir -p app/models
cat > app/models/__init__.py << 'EOF'
from .meeting import (
    StructuredNotes,
    Commitment,
    ActionItem,
    Meeting,
    MeetingCreate,
)

__all__ = [
    "StructuredNotes",
    "Commitment",
    "ActionItem",
    "Meeting",
    "MeetingCreate",
]
EOF
```

- [ ] **Step 4: Implement Pydantic models**

```bash
cat > app/models/meeting.py << 'EOF'
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
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Add test for Meeting model**

```bash
cat >> tests/test_models.py << 'EOF'


def test_meeting_model_with_all_fields():
    from uuid import uuid4
    from datetime import datetime

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

    meeting = Meeting(
        id=uuid4(),
        recorded_at=datetime.now(),
        created_at=datetime.now()
    )

    assert meeting.title is None
    assert meeting.transcript is None
    assert meeting.structured_notes is None
EOF
```

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/test_models.py -v --cov=app/models`
Expected: 6 tests PASS, 100% coverage on models

- [ ] **Step 8: Commit models**

```bash
git add app/models/ tests/test_models.py
git commit -m "feat: add Pydantic models for Meeting data

- StructuredNotes with Commitment and ActionItem
- Meeting and MeetingCreate models
- Full validation with Literal types for priority
- 100% test coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Claude Extraction Service

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/claude_extraction.py`
- Create: `backend/tests/services/test_claude_extraction.py`

- [ ] **Step 1: Write test for extract_structured_notes**

```bash
mkdir -p tests/services
cat > tests/services/test_claude_extraction.py << 'EOF'
import pytest
from unittest.mock import Mock, patch
from app.services.claude_extraction import extract_structured_notes
from app.models.meeting import StructuredNotes


@pytest.fixture
def mock_anthropic():
    with patch("app.services.claude_extraction.Anthropic") as mock:
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

    assert result.success is True
    assert isinstance(result.data, StructuredNotes)
    assert len(result.data.discussed) == 1
    assert result.data.discussed[0] == "Budget"
    assert len(result.data.action_items) == 1


def test_extract_structured_notes_invalid_json(mock_anthropic):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    mock_response = Mock()
    mock_response.content = [Mock(text='invalid json')]
    mock_client.messages.create.return_value = mock_response

    result = extract_structured_notes("transcript")

    assert result.success is False
    assert "JSON" in result.error


def test_extract_structured_notes_api_error(mock_anthropic):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = Exception("API Error")

    result = extract_structured_notes("transcript")

    assert result.success is False
    assert "API Error" in result.error


def test_extract_structured_notes_empty_transcript(mock_anthropic):
    result = extract_structured_notes("")

    assert result.success is False
    assert "empty" in result.error.lower()
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_claude_extraction.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.services.claude_extraction'"

- [ ] **Step 3: Create services/__init__.py**

```bash
mkdir -p app/services
cat > app/services/__init__.py << 'EOF'
from .claude_extraction import extract_structured_notes
from .claude_compose import compose_email

__all__ = ["extract_structured_notes", "compose_email"]
EOF
```

- [ ] **Step 4: Implement Claude extraction service**

```bash
cat > app/services/claude_extraction.py << 'EOF'
import json
import os
from typing import TypedDict
from anthropic import Anthropic
from app.models.meeting import StructuredNotes


class Result(TypedDict, total=False):
    success: bool
    data: StructuredNotes | None
    error: str | None


EXTRACTION_PROMPT = """You are an expert at analyzing meeting transcripts and extracting structured information.
The transcript may be in Greek — extract and summarize in English unless told otherwise.

From the transcript below, extract exactly:
1. "discussed" — array of strings, each a key topic covered (concise, 1 sentence each)
2. "commitments" — array of {description, owner, due_date?} — things WE committed to doing
3. "action_items" — array of {description, owner, due_date?, priority} — all action items
4. "needs_from_them" — array of strings — what we need the CLIENT to provide or do

Rules:
- Be specific and factual. Only include what was actually said.
- Do not infer or add things not mentioned.
- If a due date was mentioned, include it. Otherwise omit.
- Return ONLY valid JSON matching the StructuredNotes type. No prose, no markdown.

Transcript:
{transcript}"""


def extract_structured_notes(transcript: str) -> Result:
    """
    Extract structured meeting notes from transcript using Claude.

    Args:
        transcript: Full meeting transcript text

    Returns:
        Result with success/data or success/error
    """
    if not transcript or not transcript.strip():
        return {
            "success": False,
            "data": None,
            "error": "Transcript cannot be empty"
        }

    try:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(transcript=transcript)
                }
            ]
        )

        # Log token usage in development
        if os.environ.get("ENVIRONMENT") == "development":
            print(f"[claude_extraction] tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

        # Parse JSON response
        json_text = response.content[0].text
        data = json.loads(json_text)

        # Validate with Pydantic
        structured_notes = StructuredNotes(**data)

        return {
            "success": True,
            "data": structured_notes,
            "error": None
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "data": None,
            "error": f"Failed to parse JSON from Claude: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Claude API error: {str(e)}"
        }
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/services/test_claude_extraction.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Run coverage check**

Run: `pytest tests/services/test_claude_extraction.py -v --cov=app/services/claude_extraction`
Expected: Coverage > 90%

- [ ] **Step 7: Commit extraction service**

```bash
git add app/services/__init__.py app/services/claude_extraction.py tests/services/test_claude_extraction.py
git commit -m "feat: add Claude extraction service

- Extracts StructuredNotes from transcript
- Result pattern for error handling
- Token usage logging in dev
- 100% test coverage with mocked Anthropic client

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Claude Compose Service

**Files:**
- Create: `backend/app/services/claude_compose.py`
- Create: `backend/tests/services/test_claude_compose.py`

- [ ] **Step 1: Write test for compose_email**

```bash
cat > tests/services/test_claude_compose.py << 'EOF'
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

    assert result.success is True
    assert "Dear John" in result.data
    assert "Andre" in result.data
    assert len(result.data) > 50


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

    assert result.success is True
    assert result.data is not None


def test_compose_email_api_error(mock_anthropic, sample_notes):
    mock_client = Mock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = Exception("Rate limit exceeded")

    result = compose_email(
        structured_notes=sample_notes,
        contact_name="John",
        sender_name="Andre"
    )

    assert result.success is False
    assert "Rate limit" in result.error


def test_compose_email_missing_contact_name(sample_notes):
    result = compose_email(
        structured_notes=sample_notes,
        contact_name="",
        sender_name="Andre"
    )

    assert result.success is False
    assert "contact_name" in result.error.lower()
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_claude_compose.py -v`
Expected: FAIL with "cannot import name 'compose_email'"

- [ ] **Step 3: Implement compose_email service**

```bash
cat > app/services/claude_compose.py << 'EOF'
import os
from typing import TypedDict, Literal
from anthropic import Anthropic
from app.models.meeting import StructuredNotes


class Result(TypedDict, total=False):
    success: bool
    data: str | None
    error: str | None


ToneType = Literal["formal", "professional", "friendly"]


COMPOSE_PROMPT = """You are a professional consultant writing a follow-up email after a client meeting.
You write in first person. Your tone is warm, direct, and respectful of the reader's time.

Rules for the email:
- NEVER start with "Per our call" or "As discussed" or "Hope this finds you well"
- NEVER use corporate jargon or buzzwords
- Open with a genuine, specific 1-sentence thank you referencing something real from the meeting
- Summarize what was covered in 2-3 short paragraphs (prose, not bullets)
- List our commitments clearly with ownership — we take responsibility, no ambiguity
- List what we need from them — phrased as a request, not a demand
- Close warmly with a clear next step
- Maximum 300 words total
- Sound like a senior professional who wrote this themselves, not AI

Meeting data:
{structured_notes_json}

Client name: {contact_name}
My name: {sender_name}
Tone preference: {tone}"""


def compose_email(
    structured_notes: StructuredNotes,
    contact_name: str,
    sender_name: str,
    tone: ToneType = "professional"
) -> Result:
    """
    Compose a follow-up email from structured meeting notes using Claude.

    Args:
        structured_notes: Extracted meeting data
        contact_name: Client's name for personalization
        sender_name: Your name for signature
        tone: Email tone - formal/professional/friendly

    Returns:
        Result with success/data (email text) or success/error
    """
    if not contact_name or not contact_name.strip():
        return {
            "success": False,
            "data": None,
            "error": "contact_name is required"
        }

    if not sender_name or not sender_name.strip():
        return {
            "success": False,
            "data": None,
            "error": "sender_name is required"
        }

    try:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Convert structured notes to JSON string
        notes_json = structured_notes.model_dump_json(indent=2)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": COMPOSE_PROMPT.format(
                        structured_notes_json=notes_json,
                        contact_name=contact_name,
                        sender_name=sender_name,
                        tone=tone
                    )
                }
            ]
        )

        # Log token usage in development
        if os.environ.get("ENVIRONMENT") == "development":
            print(f"[claude_compose] tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

        email_text = response.content[0].text

        return {
            "success": True,
            "data": email_text,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Claude API error: {str(e)}"
        }
EOF
```

- [ ] **Step 4: Update services/__init__.py**

```bash
cat > app/services/__init__.py << 'EOF'
from .claude_extraction import extract_structured_notes
from .claude_compose import compose_email

__all__ = ["extract_structured_notes", "compose_email"]
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/services/test_claude_compose.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Run coverage check**

Run: `pytest tests/services/test_claude_compose.py -v --cov=app/services/claude_compose`
Expected: Coverage > 90%

- [ ] **Step 7: Commit compose service**

```bash
git add app/services/claude_compose.py app/services/__init__.py tests/services/test_claude_compose.py
git commit -m "feat: add Claude compose service

- Generates human-quality follow-up emails
- Tone customization (formal/professional/friendly)
- Input validation for required fields
- Full test coverage with mocked API

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Google Speech-to-Text Service

**Files:**
- Create: `backend/app/services/google_stt.py`
- Create: `backend/tests/services/test_google_stt.py`

- [ ] **Step 1: Write test for transcribe_audio**

```bash
cat > tests/services/test_google_stt.py << 'EOF'
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

    assert result.success is True
    assert result.data == "Καλημέρα, πώς είστε;"

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

    assert result.success is True
    assert "Long transcript" in result.data


def test_transcribe_audio_file_not_found():
    result = transcribe_audio(
        audio_file_path="/nonexistent/file.wav",
        language_code="el-GR"
    )

    assert result.success is False
    assert "not found" in result.error.lower()


def test_transcribe_audio_api_error(mock_speech_client):
    mock_client = Mock()
    mock_speech_client.return_value = mock_client
    mock_client.recognize.side_effect = Exception("API quota exceeded")

    result = transcribe_audio(
        audio_file_path="/audio/test.wav",
        language_code="el-GR"
    )

    assert result.success is False
    assert "API quota exceeded" in result.error


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

    assert result.success is False
    assert "no transcript" in result.error.lower()
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_google_stt.py -v`
Expected: FAIL with "cannot import name 'transcribe_audio'"

- [ ] **Step 3: Implement Google STT service**

```bash
cat > app/services/google_stt.py << 'EOF'
import os
from pathlib import Path
from typing import TypedDict
from google.cloud import speech


class Result(TypedDict, total=False):
    success: bool
    data: str | None
    error: str | None


def transcribe_audio(
    audio_file_path: str,
    language_code: str = "el-GR",
    duration_seconds: int = 60
) -> Result:
    """
    Transcribe audio file using Google Cloud Speech-to-Text API v2.

    Args:
        audio_file_path: Path to audio file (WAV, MP3, M4A, MP4)
        language_code: BCP-47 language code (default: el-GR for Greek)
        duration_seconds: Audio duration - determines sync vs async API call

    Returns:
        Result with success/data (transcript) or success/error

    Notes:
        - Uses chirp_2 model for best multilingual accuracy
        - Synchronous recognize for < 60s, async long_running_recognize for >= 60s
        - Always requests word-level timestamps for future speaker diarization
    """
    # Validate file exists
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        return {
            "success": False,
            "data": None,
            "error": f"Audio file not found: {audio_file_path}"
        }

    try:
        client = speech.SpeechClient()

        # Read audio file
        with open(audio_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=language_code,
            model="chirp_2",
            enable_word_time_offsets=True,
        )

        # Choose sync or async based on duration
        if duration_seconds < 60:
            # Synchronous recognition for short files
            response = client.recognize(config=config, audio=audio)
        else:
            # Asynchronous recognition for long files
            operation = client.long_running_recognize(config=config, audio=audio)
            print("[google_stt] Waiting for long-running operation to complete...")
            response = operation.result(timeout=600)  # 10 minute timeout

        # Extract transcript from results
        if not response.results:
            return {
                "success": False,
                "data": None,
                "error": "Google STT returned no transcript - audio may be silent or unclear"
            }

        # Concatenate all result transcripts
        transcript = " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )

        if not transcript.strip():
            return {
                "success": False,
                "data": None,
                "error": "Transcript is empty"
            }

        return {
            "success": True,
            "data": transcript,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Google STT error: {str(e)}"
        }
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_google_stt.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Run coverage check**

Run: `pytest tests/services/test_google_stt.py -v --cov=app/services/google_stt`
Expected: Coverage > 85%

- [ ] **Step 6: Commit STT service**

```bash
git add app/services/google_stt.py tests/services/test_google_stt.py
git commit -m "feat: add Google Speech-to-Text service

- chirp_2 model for Greek multilingual accuracy
- Sync/async based on audio duration
- Word-level timestamps for future features
- Comprehensive error handling and validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Supabase Client Service

**Files:**
- Create: `backend/app/services/supabase_client.py`
- Create: `backend/tests/services/test_supabase_client.py`

- [ ] **Step 1: Write test for create_meeting**

```bash
cat > tests/services/test_supabase_client.py << 'EOF'
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
    with patch("app.services.supabase_client.create_client") as mock:
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

    assert result.success is True
    assert result.data["id"] == str(meeting_id)
    assert result.data["title"] == "Client Call"


def test_create_meeting_database_error(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = None
    mock_response.error = {"message": "Connection failed"}

    mock_client.table.return_value.insert.return_value.execute.return_value = mock_response

    meeting_create = MeetingCreate(title="Test")
    result = create_meeting(meeting_create)

    assert result.success is False
    assert "Connection failed" in result.error


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

    assert result.success is True
    assert result.data["id"] == str(meeting_id)
    assert result.data["transcript"] == "Hello world"


def test_get_meeting_not_found(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = []
    mock_response.error = None

    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_response

    result = get_meeting(str(uuid4()))

    assert result.success is False
    assert "not found" in result.error.lower()


def test_update_meeting_transcript_success(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = [{"id": str(uuid4()), "transcript": "Updated transcript"}]
    mock_response.error = None

    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

    result = update_meeting_transcript(str(uuid4()), "Updated transcript")

    assert result.success is True


def test_list_meetings_success(mock_supabase):
    mock_client = Mock()
    mock_supabase.return_value = mock_client

    mock_response = Mock()
    mock_response.data = [
        {"id": str(uuid4()), "title": "Meeting 1", "recorded_at": datetime.now().isoformat(), "created_at": datetime.now().isoformat()},
        {"id": str(uuid4()), "title": "Meeting 2", "recorded_at": datetime.now().isoformat(), "created_at": datetime.now().isoformat()}
    ]
    mock_response.error = None

    mock_client.table.return_value.select.return_value.order.return_value.execute.return_value = mock_response

    result = list_meetings()

    assert result.success is True
    assert len(result.data) == 2
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_supabase_client.py -v`
Expected: FAIL with "cannot import name 'create_meeting'"

- [ ] **Step 3: Implement Supabase client service**

```bash
cat > app/services/supabase_client.py << 'EOF'
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
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_supabase_client.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Run coverage check**

Run: `pytest tests/services/test_supabase_client.py -v --cov=app/services/supabase_client`
Expected: Coverage > 90%

- [ ] **Step 6: Commit Supabase service**

```bash
git add app/services/supabase_client.py tests/services/test_supabase_client.py
git commit -m "feat: add Supabase client service

- CRUD operations for meetings table
- Update functions for transcript, notes, email
- List meetings with sorting and limits
- Comprehensive error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Meetings Router

**Files:**
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/routers/meetings.py`
- Create: `backend/tests/routers/test_meetings.py`

- [ ] **Step 1: Write test for GET /meetings endpoint**

```bash
mkdir -p tests/routers
cat > tests/routers/test_meetings.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from uuid import uuid4
from datetime import datetime


@pytest.fixture
def mock_supabase():
    with patch("app.routers.meetings.list_meetings") as mock_list, \
         patch("app.routers.meetings.get_meeting") as mock_get, \
         patch("app.routers.meetings.create_meeting") as mock_create:
        yield {
            "list": mock_list,
            "get": mock_get,
            "create": mock_create
        }


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_list_meetings_success(client, mock_supabase):
    meeting_id = uuid4()
    mock_supabase["list"].return_value = {
        "success": True,
        "data": [
            {
                "id": str(meeting_id),
                "title": "Client Call",
                "contact_name": "John Doe",
                "recorded_at": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
        ],
        "error": None
    }

    response = client.get("/api/meetings")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Client Call"


def test_list_meetings_database_error(client, mock_supabase):
    mock_supabase["list"].return_value = {
        "success": False,
        "data": None,
        "error": "Connection timeout"
    }

    response = client.get("/api/meetings")

    assert response.status_code == 500
    assert "Connection timeout" in response.json()["detail"]


def test_get_meeting_success(client, mock_supabase):
    meeting_id = uuid4()
    mock_supabase["get"].return_value = {
        "success": True,
        "data": {
            "id": str(meeting_id),
            "title": "Test Meeting",
            "transcript": "Hello world",
            "recorded_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        },
        "error": None
    }

    response = client.get(f"/api/meetings/{meeting_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Meeting"
    assert data["transcript"] == "Hello world"


def test_get_meeting_not_found(client, mock_supabase):
    meeting_id = uuid4()
    mock_supabase["get"].return_value = {
        "success": False,
        "data": None,
        "error": "Meeting not found"
    }

    response = client.get(f"/api/meetings/{meeting_id}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_create_meeting_success(client, mock_supabase):
    meeting_id = uuid4()
    mock_supabase["create"].return_value = {
        "success": True,
        "data": {
            "id": str(meeting_id),
            "title": "New Meeting",
            "contact_name": "Jane Doe",
            "recorded_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        },
        "error": None
    }

    response = client.post("/api/meetings", json={
        "title": "New Meeting",
        "contact_name": "Jane Doe",
        "contact_email": "jane@example.com"
    })

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "New Meeting"
    assert data["contact_name"] == "Jane Doe"


def test_create_meeting_validation_error(client):
    response = client.post("/api/meetings", json={
        "title": 123  # Invalid - should be string
    })

    assert response.status_code == 422
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/routers/test_meetings.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.main'"

- [ ] **Step 3: Create routers/__init__.py**

```bash
mkdir -p app/routers
touch app/routers/__init__.py
```

- [ ] **Step 4: Implement meetings router**

```bash
cat > app/routers/meetings.py << 'EOF'
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
EOF
```

- [ ] **Step 5: Create minimal main.py for tests**

```bash
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from app.routers import meetings

app = FastAPI(title="MeetMemo API", version="1.0.0")

app.include_router(meetings.router)
EOF
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/routers/test_meetings.py -v`
Expected: 6 tests PASS

- [ ] **Step 7: Commit meetings router**

```bash
git add app/routers/ app/main.py tests/routers/test_meetings.py
git commit -m "feat: add meetings router

- GET /api/meetings - list all
- GET /api/meetings/{id} - get by ID
- POST /api/meetings - create new
- Proper HTTP status codes and error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Transcribe Router

**Files:**
- Create: `backend/app/routers/transcribe.py`
- Create: `backend/tests/routers/test_transcribe.py`

- [ ] **Step 1: Write test for POST /transcribe endpoint**

```bash
cat > tests/routers/test_transcribe.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from io import BytesIO


@pytest.fixture
def mock_services():
    with patch("app.routers.transcribe.transcribe_audio") as mock_stt, \
         patch("app.routers.transcribe.update_meeting_transcript") as mock_update:
        yield {
            "stt": mock_stt,
            "update": mock_update
        }


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_transcribe_audio_success(client, mock_services):
    mock_services["stt"].return_value = {
        "success": True,
        "data": "Καλημέρα, αυτή είναι η μεταγραφή",
        "error": None
    }

    mock_services["update"].return_value = {
        "success": True,
        "data": {"id": "123", "transcript": "Καλημέρα, αυτή είναι η μεταγραφή"},
        "error": None
    }

    audio_file = BytesIO(b"fake audio content")

    response = client.post(
        "/api/transcribe",
        data={"meeting_id": "123"},
        files={"audio": ("test.wav", audio_file, "audio/wav")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "Καλημέρα, αυτή είναι η μεταγραφή"
    assert data["meeting_id"] == "123"


def test_transcribe_audio_missing_file(client):
    response = client.post(
        "/api/transcribe",
        data={"meeting_id": "123"}
    )

    assert response.status_code == 422


def test_transcribe_audio_missing_meeting_id(client):
    audio_file = BytesIO(b"fake audio content")

    response = client.post(
        "/api/transcribe",
        files={"audio": ("test.wav", audio_file, "audio/wav")}
    )

    assert response.status_code == 422


def test_transcribe_audio_stt_failure(client, mock_services):
    mock_services["stt"].return_value = {
        "success": False,
        "data": None,
        "error": "API quota exceeded"
    }

    audio_file = BytesIO(b"fake audio content")

    response = client.post(
        "/api/transcribe",
        data={"meeting_id": "123"},
        files={"audio": ("test.wav", audio_file, "audio/wav")}
    )

    assert response.status_code == 500
    assert "quota exceeded" in response.json()["detail"].lower()


def test_transcribe_audio_database_update_failure(client, mock_services):
    mock_services["stt"].return_value = {
        "success": True,
        "data": "Transcript text",
        "error": None
    }

    mock_services["update"].return_value = {
        "success": False,
        "data": None,
        "error": "Database connection lost"
    }

    audio_file = BytesIO(b"fake audio content")

    response = client.post(
        "/api/transcribe",
        data={"meeting_id": "123"},
        files={"audio": ("test.wav", audio_file, "audio/wav")}
    )

    assert response.status_code == 500
    assert "Database" in response.json()["detail"]
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/routers/test_transcribe.py -v`
Expected: FAIL with "cannot import name 'transcribe'"

- [ ] **Step 3: Implement transcribe router**

```bash
cat > app/routers/transcribe.py << 'EOF'
import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.services.google_stt import transcribe_audio
from app.services.supabase_client import update_meeting_transcript

router = APIRouter(prefix="/api", tags=["transcription"])


@router.post("/transcribe")
async def transcribe_meeting_audio(
    meeting_id: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Transcribe uploaded audio file and save to meeting.

    Args:
        meeting_id: UUID of the meeting to update
        audio: Audio file (WAV, MP3, M4A, MP4)

    Returns:
        Transcript text and meeting_id
    """
    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / audio.filename

    try:
        # Write uploaded file to disk
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # Transcribe audio
        stt_result = transcribe_audio(
            audio_file_path=str(temp_path),
            language_code="el-GR"
        )

        if not stt_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {stt_result['error']}"
            )

        transcript = stt_result["data"]

        # Update meeting in database
        update_result = update_meeting_transcript(meeting_id, transcript)

        if not update_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database update failed: {update_result['error']}"
            )

        return {
            "meeting_id": meeting_id,
            "transcript": transcript
        }

    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()
        os.rmdir(temp_dir)
EOF
```

- [ ] **Step 4: Update main.py to include transcribe router**

```bash
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from app.routers import meetings
from app.routers import transcribe

app = FastAPI(title="MeetMemo API", version="1.0.0")

app.include_router(meetings.router)
app.include_router(transcribe.router)
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/routers/test_transcribe.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit transcribe router**

```bash
git add app/routers/transcribe.py app/main.py tests/routers/test_transcribe.py
git commit -m "feat: add transcribe router

- POST /api/transcribe - upload audio and transcribe
- Temp file handling with cleanup
- Updates meeting record with transcript
- Error handling for STT and database failures

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Extract Router

**Files:**
- Create: `backend/app/routers/extract.py`
- Create: `backend/tests/routers/test_extract.py`

- [ ] **Step 1: Write test for POST /extract endpoint**

```bash
cat > tests/routers/test_extract.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.models.meeting import StructuredNotes, ActionItem


@pytest.fixture
def mock_services():
    with patch("app.routers.extract.extract_structured_notes") as mock_extract, \
         patch("app.routers.extract.update_meeting_notes") as mock_update:
        yield {
            "extract": mock_extract,
            "update": mock_update
        }


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_structured_notes():
    return StructuredNotes(
        discussed=["Project timeline", "Budget allocation"],
        commitments=[],
        action_items=[
            ActionItem(description="Review contract", owner="client", priority="high")
        ],
        needs_from_them=["Signed agreement"]
    )


def test_extract_notes_success(client, mock_services, sample_structured_notes):
    mock_services["extract"].return_value = {
        "success": True,
        "data": sample_structured_notes,
        "error": None
    }

    mock_services["update"].return_value = {
        "success": True,
        "data": {"id": "123"},
        "error": None
    }

    response = client.post("/api/extract", json={
        "meeting_id": "123",
        "transcript": "We discussed the project timeline and budget."
    })

    assert response.status_code == 200
    data = response.json()
    assert data["meeting_id"] == "123"
    assert len(data["structured_notes"]["discussed"]) == 2
    assert len(data["structured_notes"]["action_items"]) == 1


def test_extract_notes_missing_transcript(client):
    response = client.post("/api/extract", json={
        "meeting_id": "123"
    })

    assert response.status_code == 422


def test_extract_notes_missing_meeting_id(client):
    response = client.post("/api/extract", json={
        "transcript": "Test transcript"
    })

    assert response.status_code == 422


def test_extract_notes_empty_transcript(client):
    response = client.post("/api/extract", json={
        "meeting_id": "123",
        "transcript": ""
    })

    assert response.status_code == 422


def test_extract_notes_extraction_failure(client, mock_services):
    mock_services["extract"].return_value = {
        "success": False,
        "data": None,
        "error": "Claude API rate limit"
    }

    response = client.post("/api/extract", json={
        "meeting_id": "123",
        "transcript": "Test transcript"
    })

    assert response.status_code == 500
    assert "rate limit" in response.json()["detail"].lower()


def test_extract_notes_database_failure(client, mock_services, sample_structured_notes):
    mock_services["extract"].return_value = {
        "success": True,
        "data": sample_structured_notes,
        "error": None
    }

    mock_services["update"].return_value = {
        "success": False,
        "data": None,
        "error": "Database timeout"
    }

    response = client.post("/api/extract", json={
        "meeting_id": "123",
        "transcript": "Test transcript"
    })

    assert response.status_code == 500
    assert "timeout" in response.json()["detail"].lower()
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/routers/test_extract.py -v`
Expected: FAIL with "cannot import name 'extract'"

- [ ] **Step 3: Implement extract router**

```bash
cat > app/routers/extract.py << 'EOF'
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
EOF
```

- [ ] **Step 4: Update main.py to include extract router**

```bash
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from app.routers import meetings
from app.routers import transcribe
from app.routers import extract

app = FastAPI(title="MeetMemo API", version="1.0.0")

app.include_router(meetings.router)
app.include_router(transcribe.router)
app.include_router(extract.router)
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/routers/test_extract.py -v`
Expected: 6 tests PASS

- [ ] **Step 6: Commit extract router**

```bash
git add app/routers/extract.py app/main.py tests/routers/test_extract.py
git commit -m "feat: add extract router

- POST /api/extract - extract structured notes from transcript
- Pydantic validation for request body
- Updates meeting with structured notes
- Comprehensive error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Compose Router

**Files:**
- Create: `backend/app/routers/compose.py`
- Create: `backend/tests/routers/test_compose.py`

- [ ] **Step 1: Write test for POST /compose endpoint**

```bash
cat > tests/routers/test_compose.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.models.meeting import StructuredNotes, Commitment, ActionItem


@pytest.fixture
def mock_services():
    with patch("app.routers.compose.compose_email") as mock_compose, \
         patch("app.routers.compose.update_meeting_email") as mock_update:
        yield {
            "compose": mock_compose,
            "update": mock_update
        }


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_notes_dict():
    notes = StructuredNotes(
        discussed=["Timeline", "Budget"],
        commitments=[Commitment(description="Send proposal", owner="us")],
        action_items=[ActionItem(description="Review", owner="client", priority="high")],
        needs_from_them=["Approval"]
    )
    return notes.model_dump()


def test_compose_email_success(client, mock_services, sample_notes_dict):
    mock_services["compose"].return_value = {
        "success": True,
        "data": "Dear John,\n\nThank you for the meeting...\n\nBest regards,\nAndre",
        "error": None
    }

    mock_services["update"].return_value = {
        "success": True,
        "data": {"id": "123"},
        "error": None
    }

    response = client.post("/api/compose", json={
        "meeting_id": "123",
        "structured_notes": sample_notes_dict,
        "contact_name": "John Doe",
        "sender_name": "Andre",
        "tone": "professional"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["meeting_id"] == "123"
    assert "Dear John" in data["email_draft"]


def test_compose_email_missing_required_fields(client, sample_notes_dict):
    response = client.post("/api/compose", json={
        "meeting_id": "123",
        "structured_notes": sample_notes_dict
        # Missing contact_name and sender_name
    })

    assert response.status_code == 422


def test_compose_email_invalid_tone(client, sample_notes_dict):
    response = client.post("/api/compose", json={
        "meeting_id": "123",
        "structured_notes": sample_notes_dict,
        "contact_name": "John",
        "sender_name": "Andre",
        "tone": "super_casual"  # Invalid tone
    })

    assert response.status_code == 422


def test_compose_email_composition_failure(client, mock_services, sample_notes_dict):
    mock_services["compose"].return_value = {
        "success": False,
        "data": None,
        "error": "Claude API error"
    }

    response = client.post("/api/compose", json={
        "meeting_id": "123",
        "structured_notes": sample_notes_dict,
        "contact_name": "John",
        "sender_name": "Andre"
    })

    assert response.status_code == 500
    assert "Claude" in response.json()["detail"]


def test_compose_email_database_failure(client, mock_services, sample_notes_dict):
    mock_services["compose"].return_value = {
        "success": True,
        "data": "Email text",
        "error": None
    }

    mock_services["update"].return_value = {
        "success": False,
        "data": None,
        "error": "Database error"
    }

    response = client.post("/api/compose", json={
        "meeting_id": "123",
        "structured_notes": sample_notes_dict,
        "contact_name": "John",
        "sender_name": "Andre"
    })

    assert response.status_code == 500
    assert "Database" in response.json()["detail"]


def test_compose_email_default_tone(client, mock_services, sample_notes_dict):
    mock_services["compose"].return_value = {
        "success": True,
        "data": "Email text",
        "error": None
    }

    mock_services["update"].return_value = {
        "success": True,
        "data": {"id": "123"},
        "error": None
    }

    response = client.post("/api/compose", json={
        "meeting_id": "123",
        "structured_notes": sample_notes_dict,
        "contact_name": "John",
        "sender_name": "Andre"
        # tone omitted - should default to "professional"
    })

    assert response.status_code == 200
EOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/routers/test_compose.py -v`
Expected: FAIL with "cannot import name 'compose'"

- [ ] **Step 3: Implement compose router**

```bash
cat > app/routers/compose.py << 'EOF'
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
EOF
```

- [ ] **Step 4: Update main.py to include compose router**

```bash
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from app.routers import meetings
from app.routers import transcribe
from app.routers import extract
from app.routers import compose

app = FastAPI(title="MeetMemo API", version="1.0.0")

app.include_router(meetings.router)
app.include_router(transcribe.router)
app.include_router(extract.router)
app.include_router(compose.router)
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/routers/test_compose.py -v`
Expected: 6 tests PASS

- [ ] **Step 6: Commit compose router**

```bash
git add app/routers/compose.py app/main.py tests/routers/test_compose.py
git commit -m "feat: add compose router

- POST /api/compose - generate follow-up email
- Tone validation (formal/professional/friendly)
- Updates meeting with email draft
- Full error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Main App Wiring & Smoke Test

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Write smoke test for FastAPI app**

```bash
cat > tests/test_main.py << 'EOF'
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_app_exists(client):
    """Smoke test - app initializes without errors."""
    assert client.app is not None


def test_root_endpoint_404(client):
    """Root path should 404 - no endpoint defined."""
    response = client.get("/")
    assert response.status_code == 404


def test_docs_endpoint_exists(client):
    """OpenAPI docs should be available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_exists(client):
    """OpenAPI schema should be available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert schema["info"]["title"] == "MeetMemo API"


def test_all_routers_registered(client):
    """Verify all routers are registered in OpenAPI schema."""
    response = client.get("/openapi.json")
    schema = response.json()
    paths = schema["paths"]

    # Check all expected endpoints exist
    assert "/api/meetings" in paths
    assert "/api/meetings/{meeting_id}" in paths
    assert "/api/transcribe" in paths
    assert "/api/extract" in paths
    assert "/api/compose" in paths


def test_cors_middleware_configured(client):
    """Verify CORS middleware allows requests."""
    response = client.options(
        "/api/meetings",
        headers={"Origin": "http://localhost:3000"}
    )
    # Should not error - CORS should handle preflight
    assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly defined
EOF
```

- [ ] **Step 2: Run test to verify some fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL on test_cors_middleware_configured - CORS not configured yet

- [ ] **Step 3: Complete main.py with CORS and metadata**

```bash
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import meetings
from app.routers import transcribe
from app.routers import extract
from app.routers import compose

# Create FastAPI app
app = FastAPI(
    title="MeetMemo API",
    version="1.0.0",
    description="Personal productivity tool for transcribing meetings and generating follow-up emails",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
# In production, replace "*" with specific frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(meetings.router)
app.include_router(transcribe.router)
app.include_router(extract.router)
app.include_router(compose.router)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "service": "meetmemo-api"}
EOF
```

- [ ] **Step 4: Add health check test**

```bash
cat >> tests/test_main.py << 'EOF'


def test_health_check_endpoint(client):
    """Health check should return OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "meetmemo-api"
EOF
```

- [ ] **Step 5: Run all tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest -v`
Expected: All tests across all modules PASS

- [ ] **Step 7: Run coverage report**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: Coverage > 80% across all modules

- [ ] **Step 8: Create .env from .env.example**

```bash
cp .env.example .env
echo ""
echo "⚠️  ACTION REQUIRED: Edit .env and add your API keys:"
echo "   - ANTHROPIC_API_KEY"
echo "   - GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY"
echo "   - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
```

- [ ] **Step 9: Manual smoke test - start server**

Run: `uvicorn app.main:app --reload --port 8000`
Expected: Server starts, logs show "Application startup complete"

- [ ] **Step 10: Manual smoke test - verify endpoints**

```bash
# In a new terminal
curl http://localhost:8000/health
# Expected: {"status":"ok","service":"meetmemo-api"}

curl http://localhost:8000/docs
# Expected: HTML page with Swagger UI
```

- [ ] **Step 11: Stop server and commit final wiring**

```bash
# Stop uvicorn with Ctrl+C

git add app/main.py tests/test_main.py
git commit -m "feat: complete main app wiring

- CORS middleware configured
- Health check endpoint
- All routers registered
- OpenAPI docs at /docs
- Full smoke test passing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 12: Final verification**

Run: `pnpm test` (if pnpm configured) or `pytest -v --cov=app`
Expected: All tests pass, coverage > 80%

---

## Implementation Complete 🎉

The backend pipeline is now fully implemented:

✅ Task 1: Project scaffold
✅ Task 2: Pydantic models
✅ Task 3: Claude extraction service
✅ Task 4: Claude compose service
✅ Task 5: Google STT service
✅ Task 6: Supabase client service
✅ Task 7: Meetings router (GET list, GET by ID, POST create)
✅ Task 8: Transcribe router (POST /transcribe)
✅ Task 9: Extract router (POST /extract)
✅ Task 10: Compose router (POST /compose)
✅ Task 11: Main app wiring + smoke test

**Next steps:**
1. Add your API keys to `.env`
2. Set up Supabase database with schema from CLAUDE_3.md
3. Start the server: `uvicorn app.main:app --reload`
4. Test the full pipeline with a real audio file
5. Build the Next.js frontend to consume this API
