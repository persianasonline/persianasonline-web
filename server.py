import os
import json
import base64
import asyncio
import audioop
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
import httpx

load_dotenv()

app = FastAPI()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")

SOFIA_SYSTEM_PROMPT = """Eres Sofia, la asistente virtual de Persianas Online (marca ABYS).
Tu funcion es atender llamadas telefonicas de clientes.
Servicios: motorizacion de persianas, toldos, domotica, smart home Matter 1.5.
Visita tecnica: 20 euros (Madrid), 50 euros (+50 km o gama alta). Se descuenta del presupuesto.
Telefono: +34 624 420 102 (WhatsApp). Web: persianasonline.es
Horario: Lunes a Viernes 9:00-19:00, Sabados 10:00-14:00.
Habla en espanol. Se breve y natural, como en una conversacion telefonica real.
No uses emojis. No uses markdown."""

@app.post("/incoming-call")
async def incoming_call(request: Request):
    server_domain = os.getenv("SERVER_DOMAIN")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="es-ES">Bienvenido a Persianas Online. Un momento, por favor.</Say>
    <Connect>
        <Stream url="wss://{server_domain}/media-stream" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    stream_sid = None
    audio_buffer = bytearray()
    conversation_history = []
    silence_task = None
    SILENCE_TIMEOUT = 1.5

    async def process_speech():
        nonlocal audio_buffer, conversation_history, stream_sid
        if len(audio_buffer) < 1600:
            audio_buffer = bytearray()
            return
        pcm_audio = audioop.ulaw2lin(bytes(audio_buffer), 2)
        pcm_16k = audioop.ratecv(pcm_audio, 2, 1, 8000, 16000, None)[0]
        audio_buffer = bytearray()
        transcript = await transcribe_deepgram(pcm_16k)
        if not transcript or not transcript.strip():
            return
        print(f"[STT] Cliente: {transcript}")
        conversation_history.append({"role": "user", "content": transcript})
        response_text = await call_llm(conversation_history)
        conversation_history.append({"role": "assistant", "content": response_text})
        print(f"[LLM] Sofia: {response_text}")
        tts_audio_pcm = await elevenlabs_tts(response_text)
        pcm_8k = audioop.ratecv(tts_audio_pcm, 2, 1, 24000, 8000, None)[0]
        mulaw_audio = audioop.lin2ulaw(pcm_8k, 2)
        chunk_size = 640
        for i in range(0, len(mulaw_audio), chunk_size):
            chunk = mulaw_audio[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")
            media_msg = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            }
            await ws.send_json(media_msg)
            await asyncio.sleep(0.08)

    try:
        async for message in ws.iter_text():
            data = json.loads(message)
            event = data.get("event")
            if event == "start":
                stream_sid = data["start"]["streamSid"]
                print(f"[Twilio] Stream started: {stream_sid}")
            elif event == "media":
                chunk = base64.b64decode(data["media"]["payload"])
                audio_buffer.extend(chunk)
                if silence_task:
                    silence_task.cancel()
                silence_task = asyncio.create_task(
                    fire_after_silence(process_speech, SILENCE_TIMEOUT)
                )
            elif event == "stop":
                print("[Twilio] Stream stopped")
                break
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        if silence_task:
            silence_task.cancel()

async def fire_after_silence(callback, delay):
    await asyncio.sleep(delay)
    await callback()

async def transcribe_deepgram(pcm_audio: bytes) -> str:
    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-2",
        "language": "es",
        "encoding": "linear16",
        "sample_rate": 16000,
        "channels": 1,
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/octet-stream",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, params=params, headers=headers, content=pcm_audio)
        resp.raise_for_status()
        result = resp.json()
    try:
        return result["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        return ""

async def call_llm(conversation_history: list) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": LLM_API_KEY}
    contents = []
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    body = {
        "system_instruction": {"parts": [{"text": SOFIA_SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=headers, params=params, json=body)
        resp.raise_for_status()
        result = resp.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]

async def elevenlabs_tts(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/pcm",
    }
    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.6, "similarity_boost": 0.8, "style": 0.2},
        "output_format": "pcm_24000",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
