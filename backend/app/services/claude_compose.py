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
