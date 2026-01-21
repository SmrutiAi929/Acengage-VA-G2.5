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
import time
import os
import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed
import numpy as np

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
    phone_type: str = "ozone"  # Support for ozone/elision (ozone is default)
    closed: bool = False


def _read_prompt_text() -> str:
    prompt_file = os.getenv("PROMPT_FILE", os.path.join(os.path.dirname(__file__), "Acengage Prompt.txt"))
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # fallback: minimal prompt if file missing
        return "You are a helpful Kia Motors sales assistant. Be concise and friendly."


def _extract_audio_b64_from_gemini_message(msg: Dict[str, Any]) -> Optional[str]:
    """Extract base64 audio data from Gemini Live API response.
    
    Handles multiple possible message structures:
    - serverContent.modelTurn.parts[].inlineData.data
    - serverContent.modelTurn.parts[].inlineData (if data is at top level)
    """
    server_content = msg.get("serverContent", {})
    if not server_content:
        return None
    
    model_turn = server_content.get("modelTurn", {})
    if not model_turn:
        return None
    
    parts = model_turn.get("parts", [])
    if not parts:
        return None
    
    # Try to extract audio from parts
    for part in parts:
        if not isinstance(part, dict):
            continue
        
        inline_data = part.get("inlineData")
        if inline_data and isinstance(inline_data, dict):
            # Check for data field
            audio_data = inline_data.get("data")
            if audio_data:
                return audio_data
            # Also check if inlineData itself contains the data
            if "mimeType" in inline_data and "audio" in inline_data.get("mimeType", ""):
                audio_data = inline_data.get("data")
                if audio_data:
                    return audio_data
    
    return None


def _is_interrupted(msg: Dict[str, Any]) -> bool:
    return bool(msg.get("serverContent", {}).get("interrupted"))


async def _gemini_reader(
    session: TelephonySession, audio_processor: AudioProcessor, cfg: Config
) -> None:
    audio_chunk_count = 0
    message_count = 0
    try:
        if cfg.DEBUG:
            print(f"[{session.ucid}] 🔄 Starting Gemini message reader...")
        async for msg in session.gemini.messages():
            message_count += 1
            if cfg.DEBUG:
                # Log ALL messages for debugging
                msg_keys = list(msg.keys())
                print(f"[{session.ucid}] 📨 Gemini message #{message_count}: keys={msg_keys} ")
                
                if msg.get("setupComplete"):
                    print(f"[{session.ucid}] 🏁 Gemini setup complete - ready for conversation")

            # Check for transcription messages (user input or bot response)
            if "text" in msg:
                transcription = msg.get("text", "").strip()
                if transcription:
                    print(f"[{session.ucid}] 🤖 Bot: {transcription}")

            if "input_transcription" in msg.get("serverContent", {}) or "inputTranscription" in msg.get("serverContent", {}):
                user_transcription = msg.get("serverContent", {}).get("inputTranscription", {}).get("text", {})
                if user_transcription:
                    print(f"[{session.ucid}] 👤 User: {user_transcription}")

            if "output_transcription" in msg.get("serverContent",{}) or "outputTranscription" in msg.get("serverContent", {}):
                bot_transcription = msg.get("serverContent", {}).get("outputTranscription", {}).get("text", {})
                if bot_transcription:
                    print(f"[{session.ucid}] 🤖 Bot: {bot_transcription}")
                    exitWord=['goodbye', 'bye', 'bye.', 'take care', "you're welcome"]
                    common_elements = [item for item in bot_transcription.split() if item.lower() in exitWord]
                    if len(common_elements) > 0:
                        time.sleep(15)
                        if session.phone_type=='ozone':
                            d = json.dumps({ "command": "callDisconnect" })
                        elif session.phone_type=='elision':
                            d = json.dumps({ "event": "stop" })
                        await session.client_ws.send(d)
                    continue

            if _is_interrupted(msg):
                # Barge-in: clear any queued audio to telephony
                if cfg.DEBUG:
                    print(f"[{session.ucid}] 🛑 Gemini interrupted → clearing output buffer")
                session.output_buffer.clear()
                if session.phone_type=='ozone' :
                        d = json.dumps({
                            "command": "clearBuffer"
                        })
                elif session.phone_type=='elision': 
                        d = json.dumps({
                            "event": "clear"
                        })
                await session.client_ws.send(d)
                continue

            audio_b64 = _extract_audio_b64_from_gemini_message(msg)

            if not audio_b64:
                if cfg.DEBUG and message_count <= 10:
                    print(f"[{session.ucid}] ⚠️ No audio found in message #{message_count} (this is normal for non-audio messages)")
                continue

            audio_chunk_count += 1

            try:
                if session.phone_type == "ozone":
                    samples_8k = audio_processor.process_output_gemini_b64_to_8k_samples(audio_b64)
                    session.output_buffer.extend(samples_8k)
                else:
                    samples_8k = audio_processor.process_output_gemini_b64_to_8k_elision(audio_b64)
                    session.output_buffer.extend(samples_8k)
            except Exception as e:
                if cfg.DEBUG:
                    print(f"[{session.ucid}] ❌ Error processing audio: {e}")
                continue

            # send consistent chunks
            while len(session.output_buffer) >= cfg.AUDIO_BUFFER_SAMPLES_OUTPUT:
                chunk = session.output_buffer[: cfg.AUDIO_BUFFER_SAMPLES_OUTPUT]
                session.output_buffer = session.output_buffer[cfg.AUDIO_BUFFER_SAMPLES_OUTPUT :]

                # Send audio in correct format based on phone type
                if session.phone_type == "ozone":
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
                elif session.phone_type == "elision":
                    # Convert samples to bytes and base64 encode for elision
                    chunk=bytes(chunk)
                    payload_b64 = base64.b64encode(chunk).decode('utf-8')
                    payload = {
                            "event": "media",
                            "stream_sid": session.ucid,
                            "media": {
                                "payload": payload_b64
                            }
                    }
                else:
                    # Default to ozone format
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
                # Send audio to client (websockets 13.0: ServerConnection doesn't have 'open' attribute)
                # Just try to send - if connection is closed, it will raise an exception
                try:
                    await session.client_ws.send(json.dumps(payload))
                except Exception as e:
                    if cfg.DEBUG:
                        print(f"[{session.ucid}] ⚠️ Failed to send audio (connection may be closed): {e}")
                    break
        
        # Flush remaining buffer when connection closes (websockets 13.0: no 'open' attribute)
        if session.output_buffer:
            try:
                if cfg.DEBUG:
                    print(f"[{session.ucid}] 🚰 Flushing remaining {len(session.output_buffer)} samples")
                # Flush remaining buffer in correct format
                if session.phone_type == "ozone":
                    payload = {
                        "event": "media",
                        "type": "media",
                        "ucid": session.ucid,
                        "data": {
                            "samples": session.output_buffer,
                            "bitsPerSample": 16,
                            "sampleRate": cfg.TELEPHONY_SR,
                            "channelCount": 1,
                            "numberOfFrames": len(session.output_buffer),
                            "type": "data",
                        },
                    }
                elif session.phone_type == "elision":
                    # For elision, output_buffer contains A-law bytes (0-255) as list of ints
                    audio_bytes = bytes(session.output_buffer)
                    payload_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    payload = {
                        "event": "media",
                        "stream_sid": session.ucid,
                        "media": {
                            "payload": payload_b64
                        }
                    }
                else:
                    # Default to ozone format
                    payload = {
                        "event": "media",
                        "type": "media",
                        "ucid": session.ucid,
                        "data": {
                            "samples": session.output_buffer,
                            "bitsPerSample": 16,
                            "sampleRate": cfg.TELEPHONY_SR,
                            "channelCount": 1,
                            "numberOfFrames": len(session.output_buffer),
                            "type": "data",
                        },
                    }
                await session.client_ws.send(json.dumps(payload))
                session.output_buffer.clear()
            except Exception as e:
                if cfg.DEBUG:
                    print(f"[{session.ucid}] ⚠️ Failed to flush buffer (connection closed): {e}")
    except Exception as e:
        if cfg.DEBUG:
            import traceback
            print(f"[{session.ucid}] ❌ Gemini reader error: {e}")
            print(f"[{session.ucid}] ❌ Traceback: {traceback.format_exc()}")
        raise


async def handle_client(client_ws, path: str):
    cfg = Config()
    Config.validate(cfg)

    # websockets 13.0+ passes the request path via client_ws.request.path (including querystring e.g. "/wsNew1?agent=spotlight").
    # Waybeo/Ozonetel commonly append query params; accept those as long as the base path matches.
    if cfg.DEBUG:
        print(f"[telephony] 🔍 Connection received on path: {path!r}")
    
    base_path = (path or "").split("?", 1)[0]

    # Only accept configured base path (e.g. /ws or /wsNew1)
    if base_path != cfg.WS_PATH:
        if cfg.DEBUG:
            print(
                f"[telephony] ❌ Rejecting connection: path={path!r} base_path={base_path!r} expected={cfg.WS_PATH!r}"
            )
        await client_ws.close(code=1008, reason="Invalid path")
        return

    # Detect phone type from query parameters (ozone/elision support)
    from urllib.parse import parse_qs
    query_params = parse_qs(path.split("?", 1)[1] if "?" in path else "")
    phone_type = query_params.get("phone", [None])[0]

    # Also check for 'agent' parameter (some clients use this)
    if not phone_type:
        agent_param = query_params.get("agent", [None])[0]
        if agent_param:
            # Map agent parameter to phone type
            phone_type = "elision"  # Assume agent parameter means elision format

    # Default to ozone if no phone specified
    if phone_type not in ["ozone", "elision"]:
        phone_type = "ozone"
    
    # Log detected phone type
    print(f"[telephony] 📱 Phone type detected: {phone_type} (default: ozone)")

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
        enable_input_transcription=True,  # Enable user speech transcription
        enable_output_transcription=True, # Enable bot response transcription
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
        phone_type=phone_type,
    )

    try:
        # Wait for start event to get real UCID before connecting upstream
        first = await asyncio.wait_for(client_ws.recv(), timeout=10.0)
        start_msg = json.loads(first)
        if start_msg.get("event") != "start":
            await client_ws.close(code=1008, reason="Expected start event")
            return

        session.ucid = (
            start_msg.get("ucid") or start_msg.get("stream_sid")
            or start_msg.get("start", {}).get("ucid")
            or start_msg.get("data", {}).get("ucid")
            or "UNKNOWN"
        )

        print(f"[{session.ucid}] 📞 Call started - UCID: {session.ucid}")
        if cfg.DEBUG:
            print(f"[{session.ucid}] 🎬 start event received on path={path}")

        # Connect to Gemini
        try:
            await session.gemini.connect()
            print(f"[{session.ucid}] 🔗 Connected to Gemini Live")
            if cfg.DEBUG:
                print(f"[{session.ucid}] ✅ Connected to Gemini Live, waiting for setupComplete...")
            
            # Verify WebSocket is still open (handle websockets 13.0 API)
            if session.gemini._ws:
                try:
                    # Safe check for websockets 13.0 compatibility
                    if hasattr(session.gemini._ws, 'closed') and session.gemini._ws.closed:
                        if cfg.DEBUG:
                            print(f"[{session.ucid}] ❌ Gemini WebSocket closed immediately after connection!")
                        raise RuntimeError("Gemini WebSocket closed immediately after connection")
                    # websockets 13.0: ClientConnection doesn't have 'closed' attribute
                    # If we get here without exception, connection is open
                except AttributeError:
                    # websockets 13.0: assume connection is open
                    pass
        except Exception as e:
            if cfg.DEBUG:
                print(f"[{session.ucid}] ❌ Failed to connect to Gemini: {e}")
                import traceback
                traceback.print_exc()
            raise

        # Start reader task
        gemini_task = asyncio.create_task(_gemini_reader(session, audio_processor, cfg))
        
        # Wait for setupComplete message from Gemini before sending initial text
        # This ensures Gemini is ready to receive input
        if cfg.DEBUG:
            print(f"[{session.ucid}] ⏳ Waiting for Gemini setupComplete...")
        
        # Give Gemini time to send setupComplete (usually comes quickly)
        await asyncio.sleep(1.5)
        
        # Check if WebSocket is still open (handle websockets 13.0 API)
        if session.gemini._ws:
            try:
                is_closed = session.gemini._ws.closed if hasattr(session.gemini._ws, 'closed') else False
                if is_closed:
                    if cfg.DEBUG:
                        print(f"[{session.ucid}] ⚠️ Gemini WebSocket closed during setup wait!")
                    raise RuntimeError("Gemini WebSocket closed")
            except AttributeError:
                # websockets 13.0: assume connection is open
                pass
        
        # Send an initial text message to trigger Gemini to start speaking
        # Gemini Live responds to text input with audio output
        # The prompt instructs: "START: Begin in English greeting"
        try:
            initial_text = {
                "realtime_input": {
                    "text": "Hello"
                }
            }
            await session.gemini.send_json(initial_text)
            if cfg.DEBUG:
                print(f"[{session.ucid}] 💬 Sent initial text 'Hello' to trigger Gemini greeting")
        except Exception as e:
            if cfg.DEBUG:
                print(f"[{session.ucid}] ❌ Failed to send initial text: {e}")
                import traceback
                traceback.print_exc()
        
        if cfg.DEBUG:
            print(f"[{session.ucid}] 🎤 Ready to receive audio input and send to Gemini")

        # Process remaining messages from telephony client
        async for raw in client_ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = msg.get("event")
            if event in {"stop", "end", "close"}:
                print(f"[{session.ucid}] 📞 Call ended - Event: {event}")
                if cfg.DEBUG:
                    print(f"[{session.ucid}] 📞 stop event received")
                break

            if event == "media":
                samples = []

                if session.phone_type == "ozone" and msg.get("data"):
                    # Ozone format: data.samples array
                    samples = msg["data"].get("samples", [])
                    if not samples:
                        continue
                    session.input_buffer.extend(samples)

                elif session.phone_type == "elision" and msg.get("media"):
                    # Elision format: media.payload base64
                    payload_b64 = msg["media"].get("payload", "")
                    if not payload_b64:
                        continue

                    # Decode base64 to get raw A-law bytes, then convert to PCM16 samples
                    audio_bytes = base64.b64decode(payload_b64)
                    audio_samples = audio_processor.alaw_to_openai_format(audio_bytes)
                    session.input_buffer.extend(audio_samples)

                else:
                    # Unknown format or missing data
                    continue

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
