import collections
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30          # 30ms frames
SILENCE_THRESHOLD_MS = 1500     # stop after 1.5s silence
VAD_AGGRESSIVENESS = 2          # 0-3, 2 is balanced


@dataclass
class AudioCaptureResult:
    audio_bytes: bytes
    duration_seconds: float
    error: Optional[str] = None


def capture_audio_from_mic() -> AudioCaptureResult:
    """
    Capture audio from microphone using PyAudio + webrtcvad.
    Records until 1.5 seconds of continuous silence.
    """
    try:
        import pyaudio
        import webrtcvad
    except ImportError as e:
        return AudioCaptureResult(
            audio_bytes=b"",
            duration_seconds=0.0,
            error=f"Missing dependency: {e}. Install pyaudio and webrtcvad.",
        )

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    pa = pyaudio.PyAudio()

    frame_bytes = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # 16-bit = 2 bytes/sample
    silence_frames_needed = int(SILENCE_THRESHOLD_MS / FRAME_DURATION_MS)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=frame_bytes // 2,
    )

    logger.info("VAD: Listening for speech...")
    frames: list[bytes] = []
    triggered = False
    ring_buffer: collections.deque[tuple[bytes, bool]] = collections.deque(
        maxlen=silence_frames_needed
    )

    try:
        while True:
            frame = stream.read(frame_bytes // 2, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = sum(1 for _, s in ring_buffer if s)
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    logger.info("VAD: Speech detected, recording...")
                    frames.extend(f for f, _ in ring_buffer)
                    ring_buffer.clear()
            else:
                frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = sum(1 for _, s in ring_buffer if not s)
                if num_unvoiced == ring_buffer.maxlen:
                    logger.info("VAD: Silence detected, stopping.")
                    break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    audio_bytes = b"".join(frames)
    duration = len(frames) * FRAME_DURATION_MS / 1000
    return AudioCaptureResult(audio_bytes=audio_bytes, duration_seconds=duration)


def process_uploaded_audio(audio_bytes: bytes) -> AudioCaptureResult:
    """Accept pre-recorded audio bytes (from Streamlit audio_input widget)."""
    if not audio_bytes:
        return AudioCaptureResult(audio_bytes=b"", duration_seconds=0.0,
                                  error="No audio data received.")
    # Rough duration estimate for WAV: total_bytes / (sample_rate * bit_depth_bytes * channels)
    duration = len(audio_bytes) / (SAMPLE_RATE * 2 * 1)
    return AudioCaptureResult(audio_bytes=audio_bytes, duration_seconds=duration)
