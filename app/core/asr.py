import io
import logging
import os
import wave
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        logger.info(f"Loading Whisper model: {model_size}")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded.")
    return _whisper_model


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_seconds: float
    error: Optional[str] = None


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
    """Transcribe raw PCM audio bytes to text using faster-whisper."""
    if not audio_bytes:
        return TranscriptionResult(text="", language="", confidence=0.0,
                                   duration_seconds=0.0, error="Empty audio input.")

    duration = len(audio_bytes) / (sample_rate * 2)  # 16-bit = 2 bytes per sample
    if duration < 1.0:
        return TranscriptionResult(text="", language="", confidence=0.0,
                                   duration_seconds=duration,
                                   error="Audio too short (< 1 second).")

    wav_buffer = _pcm_to_wav(audio_bytes, sample_rate)
    try:
        model = _get_model()
        segments, info = model.transcribe(wav_buffer, beam_size=5)
        segments = list(segments)
        if not segments:
            return TranscriptionResult(text="", language=info.language,
                                       confidence=0.0, duration_seconds=duration,
                                       error="No speech detected.")

        text = " ".join(seg.text.strip() for seg in segments).strip()
        avg_conf = sum(
            sum(w.probability for w in seg.words) / max(len(seg.words), 1)
            for seg in segments
            if seg.words
        ) / max(len([s for s in segments if s.words]), 1)

        logger.info(f"Transcribed: '{text}' | lang={info.language} | conf={avg_conf:.2f}")
        return TranscriptionResult(text=text, language=info.language,
                                   confidence=avg_conf, duration_seconds=duration)
    except Exception as e:
        return TranscriptionResult(text="", language="", confidence=0.0,
                                   duration_seconds=duration, error=str(e))


def transcribe_file_bytes(file_bytes: bytes) -> TranscriptionResult:
    """Transcribe audio from file bytes (any format supported by ffmpeg/Whisper)."""
    if not file_bytes or len(file_bytes) < 1000:
        return TranscriptionResult(text="", language="", confidence=0.0,
                                   duration_seconds=0.0, error="Audio too short or empty.")
    try:
        model = _get_model()
        audio_io = io.BytesIO(file_bytes)
        segments, info = model.transcribe(audio_io, beam_size=5)
        segments = list(segments)
        if not segments:
            return TranscriptionResult(text="", language=info.language,
                                       confidence=0.0, duration_seconds=0.0,
                                       error="No speech detected.")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(text=text, language=info.language,
                                   confidence=0.9, duration_seconds=0.0)
    except Exception as e:
        return TranscriptionResult(text="", language="", confidence=0.0,
                                   duration_seconds=0.0, error=str(e))


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> io.BytesIO:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf
