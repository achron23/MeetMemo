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
