"""
Waybeo Telephony WebSocket service (Gemini Live backend) - MVP.

Protocol assumption matches the working singleinterface telephony service:
Client sends JSON messages with:
- event: "start" | "media" | "stop"
- ucid: string (call/session id)
- data.samples: number[] (int16 PCM samples at 8kHz)

This service bridges telephony audio to Gemini Live:
- Waybeo 8kHz -> resample -> Gemini 16kHz PCM16 base64
- Gemini audio output (assumed 24kHz PCM16 base64) -> resample -> Waybeo 8kHz samples
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from config import Config
from audio_processor import AudioProcessor, AudioRates
from gemini_live import GeminiLiveSession, GeminiSessionConfig


@dataclass
class TelephonySession:
    ucid: str
    client_ws: websockets.WebSocketServerProtocol
    gemini: GeminiLiveSession
    input_buffer: list[int]
    output_buffer: list[int]
    closed: bool = False


def _read_prompt_text() -> str:
    prompt_file = os.getenv("PROMPT_FILE", os.path.join(os.path.dirname(__file__), "kia_prompt.txt"))
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # fallback: minimal prompt if file missing
        return "You are a helpful Kia Motors sales assistant. Be concise and friendly."


def _extract_audio_b64_from_gemini_message(msg: Dict[str, Any]) -> Optional[str]:
    parts = msg.get("serverContent", {}).get("modelTurn", {}).get("parts") or []
    if not parts:
        return None
    inline = parts[0].get("inlineData") if isinstance(parts[0], dict) else None
    if inline and isinstance(inline, dict):
        return inline.get("data")
    return None


def _is_interrupted(msg: Dict[str, Any]) -> bool:
    return bool(msg.get("serverContent", {}).get("interrupted"))


async def _gemini_reader(
    session: TelephonySession, audio_processor: AudioProcessor, cfg: Config
) -> None:
    try:
        async for msg in session.gemini.messages():
            if cfg.DEBUG:
                if msg.get("setupComplete"):
                    print(f"[{session.ucid}] 🏁 Gemini setupComplete")

            if _is_interrupted(msg):
                # Barge-in: clear any queued audio to telephony
                if cfg.DEBUG:
                    print(f"[{session.ucid}] 🛑 Gemini interrupted → clearing output buffer")
                session.output_buffer.clear()
                continue

            audio_b64 = _extract_audio_b64_from_gemini_message(msg)
            if not audio_b64:
                continue

            samples_8k = audio_processor.process_output_gemini_b64_to_8k_samples(audio_b64)
            session.output_buffer.extend(samples_8k)

            # send consistent chunks
            while len(session.output_buffer) >= cfg.AUDIO_BUFFER_SAMPLES_OUTPUT:
                chunk = session.output_buffer[: cfg.AUDIO_BUFFER_SAMPLES_OUTPUT]
                session.output_buffer = session.output_buffer[cfg.AUDIO_BUFFER_SAMPLES_OUTPUT :]

                payload = {
                    "event": "media",
                    "type": "media",
                    "ucid": session.ucid,
                    "data": {
                        "samples": chunk,
                        "bitsPerSample": 16,
                        "sampleRate": cfg.TELEPHONY_SR,
                        "channelCount": 1,
                        "numberOfFrames": len(chunk),
                        "type": "data",
                    },
                }
                if session.client_ws.open:
                    await session.client_ws.send(json.dumps(payload))
    except Exception as e:
        if cfg.DEBUG:
            print(f"[{session.ucid}] ❌ Gemini reader error: {e}")


async def handle_client(client_ws, path: str):
    cfg = Config()
    Config.validate(cfg)

    # Only accept configured path (e.g. /ws or /wsNew1)
    if path != cfg.WS_PATH:
        await client_ws.close(code=1008, reason="Invalid path")
        return

    rates = AudioRates(
        telephony_sr=cfg.TELEPHONY_SR,
        gemini_input_sr=cfg.GEMINI_INPUT_SR,
        gemini_output_sr=cfg.GEMINI_OUTPUT_SR,
    )
    audio_processor = AudioProcessor(rates)

    prompt = _read_prompt_text()

    service_url = (
        "wss://us-central1-aiplatform.googleapis.com/ws/"
        "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
    )
    gemini_cfg = GeminiSessionConfig(
        service_url=service_url,
        model_uri=cfg.model_uri,
        voice=cfg.GEMINI_VOICE,
        system_instructions=prompt,
        enable_affective_dialog=True,
        enable_input_transcription=False,
        enable_output_transcription=False,
        vad_silence_ms=300,
        vad_prefix_ms=400,
        activity_handling="START_OF_ACTIVITY_INTERRUPTS",
    )

    # Create session with temporary ucid until 'start' arrives
    ucid = "UNKNOWN"
    gemini = GeminiLiveSession(gemini_cfg)

    session = TelephonySession(
        ucid=ucid,
        client_ws=client_ws,
        gemini=gemini,
        input_buffer=[],
        output_buffer=[],
    )

    try:
        # Wait for start event to get real UCID before connecting upstream
        first = await asyncio.wait_for(client_ws.recv(), timeout=10.0)
        start_msg = json.loads(first)
        if start_msg.get("event") != "start":
            await client_ws.close(code=1008, reason="Expected start event")
            return

        session.ucid = (
            start_msg.get("ucid")
            or start_msg.get("start", {}).get("ucid")
            or start_msg.get("data", {}).get("ucid")
            or "UNKNOWN"
        )

        if cfg.DEBUG:
            print(f"[{session.ucid}] 🎬 start event received on path={path}")

        # Connect to Gemini
        await session.gemini.connect()
        if cfg.DEBUG:
            print(f"[{session.ucid}] ✅ Connected to Gemini Live")

        # Start reader task
        gemini_task = asyncio.create_task(_gemini_reader(session, audio_processor, cfg))

        # Process remaining messages
        async for raw in client_ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = msg.get("event")
            if event in {"stop", "end", "close"}:
                if cfg.DEBUG:
                    print(f"[{session.ucid}] 📞 stop event received")
                break

            if event == "media" and msg.get("data"):
                samples = msg["data"].get("samples", [])
                if not samples:
                    continue

                session.input_buffer.extend(samples)

                while len(session.input_buffer) >= cfg.AUDIO_BUFFER_SAMPLES_INPUT:
                    chunk = session.input_buffer[: cfg.AUDIO_BUFFER_SAMPLES_INPUT]
                    session.input_buffer = session.input_buffer[cfg.AUDIO_BUFFER_SAMPLES_INPUT :]

                    samples_np = audio_processor.waybeo_samples_to_np(chunk)
                    audio_b64 = audio_processor.process_input_8k_to_gemini_16k_b64(samples_np)
                    await session.gemini.send_audio_b64_pcm16(audio_b64)

        gemini_task.cancel()
        try:
            await gemini_task
        except asyncio.CancelledError:
            pass

    except asyncio.TimeoutError:
        await client_ws.close(code=1008, reason="Timeout waiting for start event")
    except ConnectionClosed:
        pass
    except Exception as e:
        if cfg.DEBUG:
            print(f"[{session.ucid}] ❌ Telephony handler error: {e}")
    finally:
        try:
            await session.gemini.close()
        except Exception:
            pass


async def main() -> None:
    cfg = Config()
    Config.validate(cfg)
    cfg.print_config()

    # websockets.serve passes (websocket, path) for the legacy API; handler accepts both.
    async with websockets.serve(handle_client, cfg.HOST, cfg.PORT):
        print(f"✅ Telephony WS listening on ws://{cfg.HOST}:{cfg.PORT}{cfg.WS_PATH}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Telephony service stopped")


