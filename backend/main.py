import os
import json
import uuid
import time
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from google import genai
from google.genai import types

import edge_tts


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

MODEL_NAME = "gemini-3.5-flash-lite"

RENDER_BASE_URL = "https://ai-smart-cap.onrender.com"


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "audio"
RECEIVED_DIR = BASE_DIR / "received_images"

AUDIO_DIR.mkdir(exist_ok=True)
RECEIVED_DIR.mkdir(exist_ok=True)


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)


PROMPT = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image.

Identify:

1. Important everyday objects.
2. Whether a person is present.
3. Indian currency notes if clearly visible.
4. Approximate position of objects:
   - left
   - center
   - right

Do not identify people.

Do not guess or invent objects.

If currency denomination is unclear, return unknown.

Keep the spoken summary short and useful.

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
    "detected": false,
    "denomination": "unknown",
    "confidence": 0.0
  },
  "summary": "A person is present. A chair is on your left."
}
"""


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Smart Cap Backend",
    version="2.0"
)


# ============================================================
# LATEST RESULT STORAGE
# ============================================================

# Device ID -> latest result
latest_results = {}

# Used to make sure the same result is not returned repeatedly
result_counter = {}


# ============================================================
# TTS
# ============================================================

async def generate_tts(text: str, output_file: str):

    communicate = edge_tts.Communicate(
        text=text,
        voice="en-IN-NeerjaNeural",
        rate="+0%",
        volume="+0%"
    )

    await communicate.save(output_file)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "success": True,
        "service": "AI Smart Cap Backend",
        "status": "running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "success": True,
        "status": "healthy"
    }


# ============================================================
# PING
# ============================================================

@app.get("/ping")
async def ping():

    return {
        "success": True,
        "message": "pong"
    }


# ============================================================
# UPLOAD IMAGE
# ============================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    device_id: str = "audio01"
):

    start_time = time.time()

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not an image"
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty image"
        )

    image_read_seconds = time.time() - start_time

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    image_filename = (
        uuid.uuid4().hex +
        Path(file.filename or "camera.jpg").suffix
    )

    image_path = RECEIVED_DIR / image_filename

    image_path.write_bytes(image_bytes)

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    gemini_start = time.time()

    try:

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,

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
            ),

            timeout=30
        )

    except asyncio.TimeoutError:

        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {str(e)}"
        )

    gemini_seconds = time.time() - gemini_start

    # --------------------------------------------------------
    # Parse Gemini JSON
    # --------------------------------------------------------

    json_parse_start = time.time()

    try:

        analysis = json.loads(response.text)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Invalid Gemini JSON: {str(e)}"
        )

    json_parse_seconds = time.time() - json_parse_start

    # --------------------------------------------------------
    # Get spoken summary
    # --------------------------------------------------------

    summary = analysis.get(
        "summary",
        "No important objects detected."
    )

    # --------------------------------------------------------
    # Generate TTS
    # --------------------------------------------------------

    tts_start = time.time()

    audio_filename = f"speech_{uuid.uuid4().hex}.mp3"

    audio_path = AUDIO_DIR / audio_filename

    try:

        await generate_tts(
            summary,
            str(audio_path)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"TTS error: {str(e)}"
        )

    tts_seconds = time.time() - tts_start

    # --------------------------------------------------------
    # Audio URL
    # --------------------------------------------------------

    audio_url = (
        f"{RENDER_BASE_URL}/audio/{audio_filename}"
    )

    # --------------------------------------------------------
    # Create result ID
    # --------------------------------------------------------

    current_counter = result_counter.get(device_id, 0) + 1

    result_counter[device_id] = current_counter

    result_id = str(current_counter)

    # --------------------------------------------------------
    # Store result for ESP32 Audio Board
    # --------------------------------------------------------

    latest_results[device_id] = {

        "result_id": result_id,

        "audio_url": audio_url,

        "summary": summary,

        "analysis": analysis,

        "created_at": time.time()
    }

    server_total_seconds = time.time() - start_time

    # --------------------------------------------------------
    # IMPORTANT:
    # We return the response to the camera only so that
    # the HTTP request completes successfully.
    #
    # The camera does NOT need to use this response.
    #
    # The audio ESP32 gets the same result independently
    # through /next-audio/{device_id}
    # --------------------------------------------------------

    return {

        "success": True,

        "analysis": analysis,

        "audio": {

            "success": True,

            "audio_url": audio_url,

            "error": None
        },

        "image": {

            "filename": image_filename,

            "size_bytes": len(image_bytes)
        },

        "latency": {

            "image_read_seconds": round(
                image_read_seconds,
                4
            ),

            "gemini_seconds": round(
                gemini_seconds,
                4
            ),

            "json_parse_seconds": round(
                json_parse_seconds,
                4
            ),

            "tts_seconds": round(
                tts_seconds,
                4
            ),

            "server_total_seconds": round(
                server_total_seconds,
                4
            )
        }
    }


# ============================================================
# AUDIO POLLING ENDPOINT
# ============================================================

@app.get("/next-audio/{device_id}")
async def next_audio(
    device_id: str,
    last_id: str = "0"
):

    result = latest_results.get(device_id)

    # No result available
    if result is None:

        return {
            "success": True,
            "new_audio": False
        }

    # Same result as previous request
    if result["result_id"] == last_id:

        return {
            "success": True,
            "new_audio": False
        }

    # New result
    return {

        "success": True,

        "new_audio": True,

        "result_id": result["result_id"],

        "audio_url": result["audio_url"],

        "summary": result["summary"]
    }


# ============================================================
# AUDIO FILE
# ============================================================

@app.get("/audio/{filename}")
async def get_audio(filename: str):

    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Audio file not found"
        )

    return FileResponse(
        audio_path,
        media_type="audio/mpeg"
    )