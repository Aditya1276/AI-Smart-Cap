import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import edge_tts
import lameenc
import miniaudio
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# Optional shared secret. If set, the ESP32 must send it as X-API-Key.
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "")

# Verify this name with client.models.list() if you get a 404 from Gemini.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Render sets RENDER_EXTERNAL_URL automatically.
BASE_URL = os.getenv(
    "RENDER_EXTERNAL_URL", "https://ai-smart-cap.onrender.com"
).rstrip("/")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Smart Cap Backend", version="1.1.0")

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

TTS_VOICE = "en-IN-NeerjaNeural"
AUDIO_MAX_AGE_SECONDS = 600
AUDIO_NAME_RE = re.compile(r"^speech_[0-9a-f]{32}\.mp3$")

# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image.

Identify:
1. Important everyday objects that are clearly visible.
2. Whether a person is present.
3. Indian currency notes if clearly visible.
4. Approximate position of important objects: left, center, right.

Rules:
- Do NOT identify people.
- Do NOT guess or invent objects.
- Only report objects that are clearly visible.
- Ignore unimportant tiny background objects.
- If currency denomination is unclear, return "unknown".
- Do not hallucinate currency.
- Keep the spoken summary very short.
- The summary must be suitable for text-to-speech.
- Do not include unnecessary technical details.

Return ONLY valid JSON in this exact structure:

{
  "person_present": true,
  "objects": [
    {"name": "chair", "position": "left", "confidence": 0.92}
  ],
  "currency": {
    "detected": true,
    "denomination": "500 INR",
    "confidence": 0.94
  },
  "summary": "A person is ahead. A chair is on your left and a 500-rupee note is visible."
}

If no object is detected: "objects": []
If no person is detected: "person_present": false
If currency is not detected:
"currency": {"detected": false, "denomination": "unknown", "confidence": 0.0}
"""

# ============================================================
# HELPERS
# ============================================================


def cleanup_old_audio():
    now = time.time()
    for f in AUDIO_DIR.glob("speech_*.mp3"):
        try:
            if now - f.stat().st_mtime > AUDIO_MAX_AGE_SECONDS:
                f.unlink()
        except OSError:
            pass


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def convert_to_a2dp_mp3(mp3_bytes: bytes, out_path: Path):
    """edge-tts gives 24 kHz mono MP3. A2DP needs 44.1 kHz stereo."""
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=2,
        sample_rate=44100,
    )
    enc = lameenc.Encoder()
    enc.set_bit_rate(96)
    enc.set_in_sample_rate(44100)
    enc.set_out_sample_rate(44100)
    enc.set_channels(2)
    enc.set_quality(2)
    data = enc.encode(decoded.samples.tobytes()) + enc.flush()
    out_path.write_bytes(data)


async def generate_tts(text: str, out_path: Path):
    communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE)
    mp3 = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    if not mp3:
        raise RuntimeError("TTS returned no audio")
    await asyncio.to_thread(convert_to_a2dp_mp3, bytes(mp3), out_path)


async def call_gemini(image_bytes: bytes):
    return await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=[
            PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )


# ============================================================
# ROUTES
# ============================================================


@app.get("/")
def root():
    return {"success": True, "project": "AI Smart Cap", "message": "Backend is running"}


@app.get("/health")
def health():
    return {"success": True, "status": "healthy"}


@app.get("/ping")
def ping():
    return {"success": True, "message": "pong"}


@app.get("/audio/{filename}")
def get_audio(filename: str):
    if not AUDIO_NAME_RE.match(filename):
        return JSONResponse(
            status_code=404, content={"success": False, "error": "Audio file not found"}
        )
    file_path = AUDIO_DIR / filename
    if not file_path.is_file():
        return JSONResponse(
            status_code=404, content={"success": False, "error": "Audio file not found"}
        )
    return FileResponse(path=file_path, media_type="audio/mpeg", filename=filename)


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    if DEVICE_API_KEY and x_api_key != DEVICE_API_KEY:
        return JSONResponse(
            status_code=401, content={"success": False, "error": "Unauthorized"}
        )

    total_start = time.perf_counter()
    cleanup_old_audio()

    # ---------------- READ IMAGE ----------------
    t = time.perf_counter()
    image_bytes = await file.read()
    image_read_seconds = time.perf_counter() - t

    print("\n=== IMAGE RECEIVED ===", file.filename, len(image_bytes), "bytes")

    if len(image_bytes) < 100:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "Empty or invalid image"}
        )

    # ---------------- GEMINI ----------------
    t = time.perf_counter()
    try:
        response = await asyncio.wait_for(call_gemini(image_bytes), timeout=30)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504, content={"success": False, "error": "Gemini request timed out"}
        )
    except Exception as e:
        print("Gemini error:", e)
        return JSONResponse(
            status_code=500, content={"success": False, "error": f"Gemini error: {e}"}
        )
    gemini_seconds = time.perf_counter() - t
    print(f"Gemini completed in {gemini_seconds:.2f}s")

    raw_text = response.text or ""
    print("GEMINI RAW:", raw_text)

    # ---------------- JSON PARSE ----------------
    t = time.perf_counter()
    try:
        analysis = json.loads(clean_json_text(raw_text))
        if not isinstance(analysis, dict):
            raise ValueError("Gemini did not return a JSON object")
    except Exception as e:
        print("JSON parsing failed:", e)
        analysis = {
            "person_present": False,
            "objects": [],
            "currency": {"detected": False, "denomination": "unknown", "confidence": 0.0},
            "summary": "Unable to analyze the image.",
        }
    json_parse_seconds = time.perf_counter() - t

    summary = analysis.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "No important objects detected."
    print("SUMMARY:", summary)

    # ---------------- TTS ----------------
    t = time.perf_counter()
    audio_filename = f"speech_{uuid.uuid4().hex}.mp3"
    audio_path = AUDIO_DIR / audio_filename
    tts_success = False
    audio_url = None
    tts_error = None

    try:
        await asyncio.wait_for(generate_tts(summary, audio_path), timeout=20)
        tts_success = True
        audio_url = f"{BASE_URL}/audio/{audio_filename}"
        print("Audio URL:", audio_url)
    except Exception as e:
        tts_error = str(e) or e.__class__.__name__
        print("TTS failed:", tts_error)
    tts_seconds = time.perf_counter() - t

    server_total_seconds = time.perf_counter() - total_start
    print(f"Total: {server_total_seconds:.3f}s")

    return {
        "success": True,
        "analysis": analysis,
        "audio": {"success": tts_success, "audio_url": audio_url, "error": tts_error},
        "latency": {
            "image_read_seconds": round(image_read_seconds, 4),
            "gemini_seconds": round(gemini_seconds, 4),
            "json_parse_seconds": round(json_parse_seconds, 4),
            "tts_seconds": round(tts_seconds, 4),
            "server_total_seconds": round(server_total_seconds, 4),
        },
    }
