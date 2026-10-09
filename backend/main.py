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
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="AI Smart Cap Backend",
    description="Vision + Gemini + TTS backend for AI Smart Cap",
    version="2.0.0"
)


# =========================================================
# AUDIO DIRECTORY
# =========================================================

AUDIO_DIR = Path("audio")

AUDIO_DIR.mkdir(
    exist_ok=True
)


TTS_VOICE = "en-IN-NeerjaNeural"


# =========================================================
# LATEST AUDIO STORAGE
# =========================================================

# Example:
#
# latest_audio["audio01"] = {
#     "id": 5,
#     "audio_url": "...",
#     "summary": "A person is ahead..."
# }

latest_audio = {}

audio_counter = 0


# =========================================================
# GEMINI PROMPT
# =========================================================

PROMPT = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image.

Identify:

1. Important everyday objects.
2. Whether a person is present.
3. Indian currency notes if clearly visible.
4. Approximate position of important objects:
   left, center, or right.

Do not identify people.

Do not guess or invent objects.

If currency denomination is unclear,
return unknown.

Keep the spoken summary short and useful.

Return ONLY valid JSON in exactly this structure:

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
  "summary": "A person is ahead. A chair is on your left."
}
"""


# =========================================================
# TTS
# =========================================================

async def generate_tts(
    text: str,
    output_file: str
):

    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate="+0%",
        volume="+0%"
    )

    await communicate.save(
        output_file
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "AI Smart Cap backend is running"
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
# AUDIO FILE
# =========================================================

@app.get("/audio/{filename}")
def get_audio(
    filename: str
):

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
# NEXT AUDIO
# =========================================================

@app.get("/next-audio/{device_id}")
def next_audio(
    device_id: str,
    last_id: int = 0
):

    item = latest_audio.get(
        device_id
    )

    # No audio has been generated yet
    if item is None:

        return {
            "success": True,
            "available": False,
            "id": last_id
        }


    # Audio already received
    if item["id"] <= last_id:

        return {
            "success": True,
            "available": False,
            "id": last_id
        }


    # New audio available
    return {
        "success": True,
        "available": True,
        "id": item["id"],
        "audio_url": item["audio_url"],
        "summary": item["summary"]
    }


# =========================================================
# UPLOAD IMAGE
# =========================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    device_id: str = "audio01"
):

    global audio_counter


    total_start = time.perf_counter()


    # =====================================================
    # READ IMAGE
    # =====================================================

    image_start = time.perf_counter()

    image_bytes = await file.read()

    image_read_seconds = (
        time.perf_counter()
        -
        image_start
    )


    if not image_bytes:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Empty image"
            }
        )


    print()
    print("========================================")
    print("NEW IMAGE")
    print("========================================")

    print(
        "Filename:",
        file.filename
    )

    print(
        "Image size:",
        len(image_bytes),
        "bytes"
    )

    print(
        "Device:",
        device_id
    )


    # =====================================================
    # GEMINI
    # =====================================================

    gemini_start = time.perf_counter()


    analysis = None


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

                    thinking_config=
                    types.ThinkingConfig(
                        thinking_level="minimal"
                    )
                )
            ),

            timeout=30
        )


        raw_text = response.text.strip()


        print()
        print("GEMINI RESPONSE")
        print("----------------------------------------")
        print(raw_text)
        print("----------------------------------------")


        analysis = json.loads(
            raw_text
        )


    except Exception as e:

        print(
            "Gemini error:",
            str(e)
        )

        return JSONResponse(

            status_code=500,

            content={
                "success": False,
                "error": "Gemini analysis failed",
                "details": str(e)
            }
        )


    gemini_seconds = (
        time.perf_counter()
        -
        gemini_start
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    summary = analysis.get(
        "summary",
        "No important objects detected."
    )


    if not summary:

        summary = (
            "No important objects detected."
        )


    print()
    print("SUMMARY")
    print("----------------------------------------")
    print(summary)
    print("----------------------------------------")


    # =====================================================
    # TTS
    # =====================================================

    tts_start = time.perf_counter()


    audio_filename = (
        f"speech_{uuid.uuid4().hex}.mp3"
    )


    audio_path = (
        AUDIO_DIR /
        audio_filename
    )


    tts_success = False

    audio_url = None

    tts_error = None


    print(
        "Generating TTS..."
    )


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
            f"{RENDER_BASE_URL}"
            f"/audio/{audio_filename}"
        )


        print(
            "TTS completed"
        )

        print(
            "Audio URL:",
            audio_url
        )


    except Exception as e:

        tts_error = str(e)

        print(
            "TTS failed:",
            tts_error
        )


    tts_seconds = (
        time.perf_counter()
        -
        tts_start
    )


    # =====================================================
    # SAVE AUDIO FOR AUDIO ESP32
    # =====================================================

    if (
        tts_success
        and
        audio_url
    ):

        audio_counter += 1


        latest_audio[device_id] = {

            "id":
                audio_counter,

            "audio_url":
                audio_url,

            "summary":
                summary
        }


        print()
        print("========================================")
        print("NEW AUDIO AVAILABLE")
        print("========================================")

        print(
            "Device:",
            device_id
        )

        print(
            "Audio ID:",
            audio_counter
        )

        print(
            "Audio URL:",
            audio_url
        )

        print(
            "Summary:",
            summary
        )

        print("========================================")


    # =====================================================
    # TOTAL TIME
    # =====================================================

    server_total_seconds = (
        time.perf_counter()
        -
        total_start
    )


    print()
    print("========================================")
    print("REQUEST COMPLETE")
    print("========================================")

    print(
        f"Image read: "
        f"{image_read_seconds:.3f}s"
    )

    print(
        f"Gemini: "
        f"{gemini_seconds:.3f}s"
    )

    print(
        f"TTS: "
        f"{tts_seconds:.3f}s"
    )

    print(
        f"Total: "
        f"{server_total_seconds:.3f}s"
    )

    print("========================================")


    # =====================================================
    # RESPONSE TO CAMERA
    #
    # The camera does not need to process this.
    # It simply sends the image and ignores the JSON.
    # =====================================================

    return {

        "success": True,

        "analysis":
            analysis,

        "audio": {

            "success":
                tts_success,

            "audio_url":
                audio_url,

            "error":
                tts_error
        },

        "device_id":
            device_id,

        "audio_id":
            audio_counter
            if tts_success
            else None,

        "latency": {

            "image_read_seconds":
                round(
                    image_read_seconds,
                    4
                ),

            "gemini_seconds":
                round(
                    gemini_seconds,
                    4
                ),

            "tts_seconds":
                round(
                    tts_seconds,
                    4
                ),

            "server_total_seconds":
                round(
                    server_total_seconds,
                    4
                )
        }
    }


# =========================================================
# UPLOAD TEST
# =========================================================

@app.post("/upload-test")
async def upload_test(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    return {

        "success": True,

        "filename":
            file.filename,

        "size":
            len(image_bytes),

        "message":
            "Upload test successful"
    }