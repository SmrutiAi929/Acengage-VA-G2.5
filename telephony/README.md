## Kia VoiceAgent Telephony (Waybeo) – Gemini Live MVP

This folder adds a **Python telephony WebSocket service** modeled after the working `singleinterface2.0ws` release
([reference branch](https://github.com/pragyaa-ai/singleinterface2.0ws/tree/v4.3.7-kia-python-librosa-UI)).

### Goals (MVP)
- Accept **Waybeo** telephony WebSocket connections (8kHz PCM sample arrays)
- Bridge audio to **Gemini Live** (`gemini-live-2.5-flash-native-audio`) using the **same Kia prompt + Aoede voice**
- Support responsive playback and basic barge‑in (buffer clearing on interruption events)

### Ports / paths (match your existing convention)
- **Production telephony:** `ws://<VM_IP>:8080/ws`
- **Testing telephony:** `ws://<VM_IP>:8081/wsNew1`

> Note: This telephony service is plain **WS** (not WSS) because Waybeo uses `ws://` on VM ports.

### Install

```bash
cd telephony
python3 -m pip install -r requirements.txt
```

### Run (two processes)

**Prod**
```bash
HOST=0.0.0.0 PORT=8080 WS_PATH=/ws python3 main.py
```

**Test**
```bash
HOST=0.0.0.0 PORT=8081 WS_PATH=/wsNew1 python3 main.py
```

### Required environment variables
- `GCP_PROJECT_ID` – e.g. `voiceagentprojects`
- `GEMINI_MODEL` – default `gemini-live-2.5-flash-native-audio`
- `GEMINI_LOCATION` – default `us-central1`

Optional:
- `AUDIO_BUFFER_MS_INPUT` / `AUDIO_BUFFER_MS_OUTPUT` (default 200ms)
- `DEBUG=true`

### VM prerequisites
- VM Service Account must have `roles/aiplatform.user` + `roles/serviceusage.serviceUsageConsumer`
- VM OAuth access scopes include `cloud-platform`


