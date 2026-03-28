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
