"""
Telephony service configuration (Waybeo) for Kia VoiceAgent.

This service is intentionally separated from the UI Gemini proxy so that:
- UI deployment remains stable (Gemini WS routed via nginx on /geminiWs)
- Telephony can own raw WS ports 8080 (/ws) and 8081 (/wsNew1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", os.getenv("PYTHON_PORT", "8081")))
    WS_PATH: str = os.getenv("WS_PATH", "/ws")

    DEBUG: bool = _env_bool("DEBUG", False)

    # GCP / Gemini
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GEMINI_LOCATION: str = os.getenv("GEMINI_LOCATION", "us-central1")
    GEMINI_MODEL: str = os.getenv(
        "GEMINI_MODEL", "gemini-live-2.5-flash-native-audio"
    )
    GEMINI_VOICE: str = os.getenv("GEMINI_VOICE", "Aoede")

    # Audio
    TELEPHONY_SR: int = int(os.getenv("TELEPHONY_SR", "8000"))  # Waybeo input/output
    GEMINI_INPUT_SR: int = int(os.getenv("GEMINI_INPUT_SR", "16000"))  # Gemini mic input
    GEMINI_OUTPUT_SR: int = int(os.getenv("GEMINI_OUTPUT_SR", "24000"))  # Gemini audio output

    # Buffers (ms)
    AUDIO_BUFFER_MS_INPUT: int = int(os.getenv("AUDIO_BUFFER_MS_INPUT", "200"))
    AUDIO_BUFFER_MS_OUTPUT: int = int(os.getenv("AUDIO_BUFFER_MS_OUTPUT", "200"))

    @property
    def AUDIO_BUFFER_SAMPLES_INPUT(self) -> int:
        return int((self.AUDIO_BUFFER_MS_INPUT / 1000.0) * self.TELEPHONY_SR)

    @property
    def AUDIO_BUFFER_SAMPLES_OUTPUT(self) -> int:
        return int((self.AUDIO_BUFFER_MS_OUTPUT / 1000.0) * self.TELEPHONY_SR)

    @property
    def model_uri(self) -> str:
        return (
            f"projects/{self.GCP_PROJECT_ID}/locations/{self.GEMINI_LOCATION}"
            f"/publishers/google/models/{self.GEMINI_MODEL}"
        )

    @classmethod
    def validate(cls, cfg: "Config") -> None:
        if not cfg.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID is required (e.g. voiceagentprojects)")

        if not cfg.WS_PATH.startswith("/"):
            raise ValueError("WS_PATH must start with '/' (e.g. /ws or /wsNew1)")

    def print_config(self) -> None:
        print("=" * 68)
        print("📞 Kia VoiceAgent Telephony (Gemini Live) – Configuration")
        print("=" * 68)
        print(f"🌐 Server: ws://{self.HOST}:{self.PORT}{self.WS_PATH}")
        print(f"🧠 Gemini model: {self.GEMINI_MODEL}")
        print(f"🎙️  Voice: {self.GEMINI_VOICE}")
        print(f"📍 Location: {self.GEMINI_LOCATION}")
        print(f"🏷️  Project: {self.GCP_PROJECT_ID}")
        print(
            f"🎵 Audio SR: telephony={self.TELEPHONY_SR}Hz, "
            f"gemini_in={self.GEMINI_INPUT_SR}Hz, gemini_out={self.GEMINI_OUTPUT_SR}Hz"
        )
        print(
            f"🎵 Buffers: in={self.AUDIO_BUFFER_MS_INPUT}ms "
            f"({self.AUDIO_BUFFER_SAMPLES_INPUT} samples), "
            f"out={self.AUDIO_BUFFER_MS_OUTPUT}ms "
            f"({self.AUDIO_BUFFER_SAMPLES_OUTPUT} samples)"
        )
        print(f"🐞 DEBUG: {self.DEBUG}")
        print("=" * 68)


