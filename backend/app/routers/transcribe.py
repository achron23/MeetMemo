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
