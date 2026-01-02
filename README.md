# Kia VoiceAgent (Gemini Live 2.5) — UI + Telephony WebSockets

This repo is a **VoiceAgent WebSocket stack** for:

- **Browser UI testing** (mic ↔ Gemini Live ↔ speaker) over **WSS** (required for browser audio APIs)
- **Telephony testing (Ozonetel/Waybeo style)** over **WS/WSS** using the same Gemini Live conversation flow

It is built around **Gemini Live** using the model:

- `gemini-live-2.5-flash-native-audio`

---

## What runs where

### 1) Browser UI + Gemini WS proxy (`server.py`)
- **HTTP UI**: defaults to `http://127.0.0.1:3001`
- **WS proxy (internal)**: defaults to `ws://127.0.0.1:9001`
- Intended to be exposed via **nginx** as:
  - UI: `https://<domain>/kiavoiceagent/`
  - UI WS: `wss://<domain>/geminiWs` → `127.0.0.1:9001`

### 2) Telephony WS service (`telephony/main.py`)
- **Prod**: `0.0.0.0:8080/ws`
- **Test**: `0.0.0.0:8081/wsNew1`
- Accepts `/wsNew1?agent=spotlight` style query strings.

For telephony details, see `telephony/README.md`.

---

## Quick start (local dev)

### UI + Gemini WS proxy

```bash
cd Kia-VA-G2.5
python3 -m pip install -r requirements.txt
gcloud auth application-default login
HTTP_PORT=3001 WS_PORT=9001 python3 server.py
```

Open `http://localhost:3001` and set:
- **Proxy WebSocket URL**: `ws://localhost:9001` (local) or `wss://<domain>/geminiWs` (nginx)
- **Project ID**: your GCP project (e.g. `voiceagentprojects`)
- **Model**: `gemini-live-2.5-flash-native-audio`

### Telephony service (test port)

```bash
cd Kia-VA-G2.5/telephony
python3 -m pip install -r requirements.txt
GCP_PROJECT_ID=voiceagentprojects HOST=0.0.0.0 PORT=8081 WS_PATH=/wsNew1 python3 main.py
```

---

## Ozonetel / Waybeo testing (telephony)

You can point the provider to either:

- **Direct VM port (WS)**: `ws://<VM_IP>:8081/wsNew1?agent=spotlight`
- **Via nginx (WSS)**: `wss://<telephony-domain>/wsNew1?agent=spotlight`

> Use **WSS** if your provider requires TLS. If the provider can connect to a raw VM port, WS is simplest.

---

## Nginx routing (recommended on VM)

You need HTTPS for the UI (browser mic/worklets require a secure context), and you should terminate WSS at nginx.

Example snippets (adapt to your vhost):

```nginx
location ^~ /kiavoiceagent/ {
  proxy_pass http://127.0.0.1:3001/;
}

location = /geminiWs {
  proxy_pass http://127.0.0.1:9001;
  proxy_http_version 1.1;
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "Upgrade";
  proxy_read_timeout 3600s;
  proxy_send_timeout 3600s;
}
```

Telephony WSS termination (if used) is typically a separate server_name (example `ws-...`) that proxies to `127.0.0.1:8081`.

---

## Systemd (auto-start on reboot)

On the VM, run these as services:

- **UI + Gemini WS proxy**: `server.py` (HTTP 3001 + WS 9001)
- **Telephony test**: `telephony/main.py` (WS 8081 `/wsNew1`)
- **Telephony prod**: `telephony/main.py` (WS 8080 `/ws`)

If you already have unit files, keep them. If not, ask and we’ll generate the exact units for your VM paths.

---

## Customization points (for other teams)

- **Prompt / greeting**: update the system instructions in `frontend/index.html` (UI) and `telephony/kia_prompt.txt` (telephony)
- **Voice**: UI voice is chosen in `frontend/index.html`; telephony defaults live in telephony config
- **Routing**: keep UI WS separate from telephony WS to avoid port collisions
- **Model**: keep fixed to `gemini-live-2.5-flash-native-audio` unless you explicitly want to test other models

---

## Licensing

- See `LICENSE` for commercial terms for this repository.
- See `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt` for third‑party license attributions that must be retained.
