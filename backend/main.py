import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

MODEL_NAME = "gemini-3.5-flash-lite"

RENDER_BASE_URL = "https://ai-smart-cap.onrender.com"


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="AI Smart Cap Backend",
    description="Vision + Gemini + TTS backend for AI Smart Cap",
    version="1.0.0"
)


# =========================================================
# AUDIO DIRECTORY
# =========================================================

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)


TTS_VOICE = "en-IN-NeerjaNeural"


# =========================================================
# GEMINI PROMPT
# =========================================================

PROMPT = """
You are the visual intelligence system of an assistive wearable for a visually impaired user.

Analyze the supplied image.

Identify:

1. Important everyday objects that are clearly visible.
2. Whether a person is present.
3. Indian currency notes if clearly visible.
4. Approximate position of important objects:
   - left
   - center
   - right

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
    {
      "name": "chair",
      "position": "left",
      "confidence": 0.92
    }
  ],
  "currency": {
    "detected": true,
    "denomination": "500 INR",
    "confidence": 0.94
  },
  "summary": "A person is ahead. A chair is on your left and a 500-rupee note is visible."
}

If no object is detected:

"objects": []

If no person is detected:

"person_present": false

If currency is not detected:

"currency": {
  "detected": false,
  "denomination": "unknown",
  "confidence": 0.0
}
"""


# =========================================================
# TTS
# =========================================================

async def generate_tts(text: str, output_file: str):

    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate="+0%",
        volume="+0%"
    )

    await communicate.save(output_file)


# =========================================================
# GEMINI FUNCTION
# =========================================================

def call_gemini(image_bytes: bytes):

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            PROMPT,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            )
        )
    )

    return response


# =========================================================
# JSON CLEANING
# =========================================================

def clean_json_text(text: str):

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "success": True,
        "project": "AI Smart Cap",
        "message": "Backend is running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy"
    }


# =========================================================
# PING
# =========================================================

@app.get("/ping")
def ping():

    return {
        "success": True,
        "message": "pong"
    }


# =========================================================
# TTS TEST
# =========================================================

@app.get("/tts-test")
async def tts_test():

    filename = f"test_{uuid.uuid4().hex}.mp3"

    output_file = AUDIO_DIR / filename

    try:

        await asyncio.wait_for(
            generate_tts(
                "AI Smart Cap audio test successful.",
                str(output_file)
            ),
            timeout=15
        )

        audio_url = f"{RENDER_BASE_URL}/audio/{filename}"

        return {
            "success": True,
            "message": "TTS generated successfully",
            "audio_url": audio_url
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# =========================================================
# AUDIO FILE
# =========================================================

@app.get("/audio/{filename}")
def get_audio(filename: str):

    file_path = AUDIO_DIR / filename

    if not file_path.exists():

        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Audio file not found"
            }
        )

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )


# =========================================================
# DIRECT TTS
# =========================================================

@app.post("/tts")
async def tts_endpoint(text: str):

    filename = f"tts_{uuid.uuid4().hex}.mp3"

    output_file = AUDIO_DIR / filename

    try:

        await asyncio.wait_for(
            generate_tts(
                text,
                str(output_file)
            ),
            timeout=15
        )

        audio_url = f"{RENDER_BASE_URL}/audio/{filename}"

        return {
            "success": True,
            "text": text,
            "audio_url": audio_url
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# =========================================================
# MAIN UPLOAD
# =========================================================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    total_start = time.perf_counter()

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_start = time.perf_counter()

    image_bytes = await file.read()

    image_read_seconds = (
        time.perf_counter() - image_start
    )

    print()
    print("========================================")
    print("IMAGE RECEIVED")
    print("========================================")
    print("Filename:", file.filename)
    print("Image size:", len(image_bytes), "bytes")

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    gemini_start = time.perf_counter()

    print("Sending image to Gemini...")

    try:

        response = await asyncio.wait_for(
            asyncio.to_thread(
                call_gemini,
                image_bytes
            ),
            timeout=30
        )

    except asyncio.TimeoutError:

        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Gemini request timed out"
            }
        )

    except Exception as e:

        print("Gemini error:", e)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Gemini error: {str(e)}"
            }
        )

    gemini_seconds = (
        time.perf_counter() - gemini_start
    )

    print(
        f"Gemini completed in "
        f"{gemini_seconds:.2f} seconds"
    )

    # -----------------------------------------------------
    # RESPONSE TEXT
    # -----------------------------------------------------

    raw_text = response.text or ""

    print()
    print("GEMINI RAW RESPONSE")
    print("----------------------------------------")
    print(raw_text)
    print("----------------------------------------")

    # -----------------------------------------------------
    # JSON PARSE
    # -----------------------------------------------------

    json_start = time.perf_counter()

    try:

        cleaned_text = clean_json_text(raw_text)

        analysis = json.loads(cleaned_text)

    except Exception as e:

        print("JSON parsing failed:", e)

        analysis = {
            "person_present": False,
            "objects": [],
            "currency": {
                "detected": False,
                "denomination": "unknown",
                "confidence": 0.0
            },
            "summary": "Unable to analyze the image."
        }

    json_parse_seconds = (
        time.perf_counter() - json_start
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary = analysis.get(
        "summary",
        "No important objects detected."
    )

    if not summary:

        summary = "No important objects detected."

    print()
    print("SUMMARY")
    print("----------------------------------------")
    print(summary)
    print("----------------------------------------")

    # -----------------------------------------------------
    # TTS
    # -----------------------------------------------------

    tts_start = time.perf_counter()

    audio_filename = (
        f"speech_{uuid.uuid4().hex}.mp3"
    )

    audio_path = AUDIO_DIR / audio_filename

    tts_success = False
    audio_url = None
    tts_error = None

    print("Generating TTS...")

    try:

        await asyncio.wait_for(
            generate_tts(
                summary,
                str(audio_path)
            ),
            timeout=15
        )

        tts_success = True

        audio_url = (
            f"{RENDER_BASE_URL}/audio/"
            f"{audio_filename}"
        )

        print("TTS completed")
        print("Audio URL:", audio_url)

    except Exception as e:

        tts_error = str(e)

        print(
            "TTS failed:",
            tts_error
        )

    tts_seconds = (
        time.perf_counter() - tts_start
    )

    # -----------------------------------------------------
    # TOTAL
    # -----------------------------------------------------

    server_total_seconds = (
        time.perf_counter() - total_start
    )

    print()
    print("========================================")
    print("REQUEST COMPLETE")
    print("========================================")

    print(
        f"Image read: {image_read_seconds:.3f}s"
    )

    print(
        f"Gemini: {gemini_seconds:.3f}s"
    )

    print(
        f"JSON parse: {json_parse_seconds:.3f}s"
    )

    print(
        f"TTS: {tts_seconds:.3f}s"
    )

    print(
        f"Total: {server_total_seconds:.3f}s"
    )

    print("========================================")
    print()

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "success": True,

        "analysis": analysis,

        "audio": {
            "success": tts_success,
            "audio_url": audio_url,
            "error": tts_error
        },

        "latency": {

            "image_read_seconds":
                round(image_read_seconds, 4),

            "gemini_seconds":
                round(gemini_seconds, 4),

            "json_parse_seconds":
                round(json_parse_seconds, 4),

            "tts_seconds":
                round(tts_seconds, 4),

            "server_total_seconds":
                round(server_total_seconds, 4)
        }
    }


# =========================================================
# TEST UPLOAD
# =========================================================

@app.post("/upload-test")
async def upload_test(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    return {
        "success": True,
        "filename": file.filename,
        "size": len(image_bytes),
        "message": "Upload test successful"
    }