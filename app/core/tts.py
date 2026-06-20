import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_MODEL = "eleven_turbo_v2_5"
TMP_AUDIO_PATH = "/tmp/response_audio.mp3"


@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_path: str
    used_fallback: bool = False
    error: Optional[str] = None


def synthesise_speech(text: str) -> TTSResult:
    """Convert text to speech. Uses ElevenLabs; falls back to gTTS."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        try:
            return _elevenlabs_tts(text, api_key)
        except Exception as e:
            logger.warning(f"ElevenLabs TTS failed: {e}. Falling back to gTTS.")

    return _gtts_tts(text)


def _elevenlabs_tts(text: str, api_key: str) -> TTSResult:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import save

    client = ElevenLabs(api_key=api_key)
    audio_generator = client.generate(
        text=text,
        voice=ELEVENLABS_VOICE_ID,
        model=ELEVENLABS_MODEL,
    )
    audio_bytes = b"".join(audio_generator)
    with open(TMP_AUDIO_PATH, "wb") as f:
        f.write(audio_bytes)
    return TTSResult(audio_bytes=audio_bytes, audio_path=TMP_AUDIO_PATH, used_fallback=False)


def _gtts_tts(text: str) -> TTSResult:
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_bytes = buf.read()
        # Also save to tmp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        return TTSResult(audio_bytes=audio_bytes, audio_path=tmp_path, used_fallback=True)
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        return TTSResult(audio_bytes=b"", audio_path="", used_fallback=True, error=str(e))
