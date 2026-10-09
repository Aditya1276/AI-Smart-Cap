import os
import json
import uuid
import time
import asyncio
from pathlib import Path

import edge_tts
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Add it to your .env file locally or Render Environment Variables."
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

RENDER_BASE_URL = os.getenv(
    "RENDER_BASE_URL",
    "https://ai-smart-cap.onrender.com"
).rstrip("/")


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RECEIVED_IMAGES_DIR = BASE_DIR / "received_images"
AUDIO_DIR = BASE_DIR / "audio"

RECEIVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Smart Cap Backend",
    description="Backend for AI Smart Cap visual assistance system",
    version="1.0.0"
)


# ============================================================
# SERVE AUDIO FILES
# ============================================================

app.mount(
    "/audio",
    StaticFiles(directory=str(AUDIO_DIR)),
    name="audio"
)


# ============================================================
# GEMINI PROMPT
# ============================================================

PROMPT = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image carefully.

Your tasks:

1. Detect important everyday objects.
2. Detect whether a person is present.
3. Detect Indian currency notes if clearly visible.
4. Estimate the approximate position of important objects:
   - left
   - center
   - right
5. Create a very short spoken summary.
6. Do not identify or name people.
7. Do not guess objects that are not clearly visible.
8. Do not invent currency denominations.
9. If currency is unclear, return denomination as "unknown".
10. Only include useful objects that matter for an assistive system.
11. Keep the summary short and natural for text-to-speech.

IMPORTANT:
Return ONLY valid JSON.

Required JSON structure:

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
    "summary": "A person is ahead. A chair is on your left."
}

If no person is detected:

"person_present": false

If no useful objects are detected:

"objects": []

If currency is not visible:

{
    "detected": false,
    "denomination": "unknown",
    "confidence": 0.0
}

Confidence must be between 0.0 and 1.0.

Position must be one of:

"left"
"center"
"right"

Do not include explanations outside the JSON.
"""


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {
        "success": True,
        "project": "AI Smart Cap",
        "message": "AI Smart Cap backend is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "healthy"
    }


@app.get("/ping")
async def ping():
    return {
        "success": True,
        "message": "pong"
    }


# ============================================================
# GEMINI ANALYSIS
# ============================================================

async def analyze_image_with_gemini(image_bytes: bytes):
    """
    Send image to Gemini and return structured JSON.
    """

    def generate():

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

        return response.text

    try:

        result_text = await asyncio.wait_for(
            asyncio.to_thread(generate),
            timeout=30
        )

    except asyncio.TimeoutError:
        raise RuntimeError(
            "Gemini request timed out after 30 seconds."
        )

    except Exception as e:
        raise RuntimeError(
            f"Gemini API error: {str(e)}"
        )

    if not result_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    # --------------------------------------------------------
    # Clean possible markdown JSON
    # --------------------------------------------------------

    result_text = result_text.strip()

    if result_text.startswith("```json"):
        result_text = result_text[7:]

    elif result_text.startswith("```"):
        result_text = result_text[3:]

    if result_text.endswith("```"):
        result_text = result_text[:-3]

    result_text = result_text.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        analysis = json.loads(result_text)

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Gemini returned invalid JSON: {str(e)}\n"
            f"Raw response: {result_text}"
        )

    return analysis


# ============================================================
# VALIDATE / NORMALIZE GEMINI RESPONSE
# ============================================================

def normalize_analysis(data):

    # --------------------------------------------------------
    # person_present
    # --------------------------------------------------------

    person_present = bool(
        data.get("person_present", False)
    )

    # --------------------------------------------------------
    # objects
    # --------------------------------------------------------

    objects = data.get("objects", [])

    if not isinstance(objects, list):
        objects = []

    clean_objects = []

    for obj in objects:

        if not isinstance(obj, dict):
            continue

        name = str(
            obj.get("name", "unknown")
        ).strip()

        position = str(
            obj.get("position", "center")
        ).strip().lower()

        if position not in [
            "left",
            "center",
            "right"
        ]:
            position = "center"

        try:
            confidence = float(
                obj.get("confidence", 0.0)
            )
        except:
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        if name and name.lower() != "unknown":

            clean_objects.append({
                "name": name,
                "position": position,
                "confidence": round(
                    confidence,
                    2
                )
            })

    # --------------------------------------------------------
    # currency
    # --------------------------------------------------------

    currency = data.get("currency", {})

    if not isinstance(currency, dict):
        currency = {}

    currency_detected = bool(
        currency.get("detected", False)
    )

    denomination = str(
        currency.get(
            "denomination",
            "unknown"
        )
    ).strip()

    try:
        currency_confidence = float(
            currency.get(
                "confidence",
                0.0
            )
        )
    except:
        currency_confidence = 0.0

    currency_confidence = max(
        0.0,
        min(1.0, currency_confidence)
    )

    if not currency_detected:
        denomination = "unknown"
        currency_confidence = 0.0

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary = str(
        data.get(
            "summary",
            "No important objects detected."
        )
    ).strip()

    if not summary:
        summary = "No important objects detected."

    return {
        "person_present": person_present,

        "objects": clean_objects,

        "currency": {
            "detected": currency_detected,
            "denomination": denomination,
            "confidence": round(
                currency_confidence,
                2
            )
        },

        "summary": summary
    }


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def generate_tts(
    text: str,
    output_file: str
):

    communicate = edge_tts.Communicate(
        text=text,

        # Indian English female voice
        voice="en-IN-NeerjaNeural",

        rate="+0%",
        volume="+0%"
    )

    await communicate.save(
        output_file
    )


# ============================================================
# UPLOAD IMAGE + AI ANALYSIS + TTS
# ============================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    request_start = time.perf_counter()

    # ========================================================
    # READ IMAGE
    # ========================================================

    image_start = time.perf_counter()

    try:

        image_bytes = await file.read()

    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Could not read image: {str(e)}"
            }
        )

    image_read_seconds = (
        time.perf_counter() - image_start
    )

    if not image_bytes:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Empty image received."
            }
        )

    # ========================================================
    # CHECK IMAGE SIZE
    # ========================================================

    image_size = len(image_bytes)

    if image_size > 10 * 1024 * 1024:

        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": "Image is larger than 10 MB."
            }
        )

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    image_filename = (
        f"{uuid.uuid4().hex}.jpg"
    )

    image_path = (
        RECEIVED_IMAGES_DIR /
        image_filename
    )

    try:

        image_path.write_bytes(
            image_bytes
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Could not save image: {str(e)}"
            }
        )

    # ========================================================
    # GEMINI
    # ========================================================

    gemini_start = time.perf_counter()

    try:

        raw_analysis = (
            await analyze_image_with_gemini(
                image_bytes
            )
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

    gemini_seconds = (
        time.perf_counter() - gemini_start
    )

    # ========================================================
    # NORMALIZE JSON
    # ========================================================

    json_start = time.perf_counter()

    try:

        analysis = normalize_analysis(
            raw_analysis
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"JSON normalization error: {str(e)}"
            }
        )

    json_parse_seconds = (
        time.perf_counter() - json_start
    )

    # ========================================================
    # TTS
    # ========================================================

    tts_start = time.perf_counter()

    audio_filename = (
        f"speech_{uuid.uuid4().hex}.mp3"
    )

    audio_path = (
        AUDIO_DIR /
        audio_filename
    )

    try:

        await generate_tts(
            analysis["summary"],
            str(audio_path)
        )

        tts_success = True
        tts_error = None

    except Exception as e:

        tts_success = False
        tts_error = str(e)

    tts_seconds = (
        time.perf_counter() - tts_start
    )

    # ========================================================
    # AUDIO URL
    # ========================================================

    audio_url = None

    if tts_success:

        audio_url = (
            f"{RENDER_BASE_URL}"
            f"/audio/{audio_filename}"
        )

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_seconds = (
        time.perf_counter()
        - request_start
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "success": True,

        "analysis": analysis,

        "audio": {
            "success": tts_success,
            "audio_url": audio_url,
            "error": tts_error
        },

        "image": {
            "filename": image_filename,
            "size_bytes": image_size
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
                total_seconds,
                4
            )
        }
    }


# ============================================================
# TEST TTS ENDPOINT
# ============================================================

@app.get("/tts")
async def test_tts():

    text = (
        "AI Smart Cap audio test successful."
    )

    filename = (
        f"test_{uuid.uuid4().hex}.mp3"
    )

    output_path = (
        AUDIO_DIR /
        filename
    )

    try:

        await generate_tts(
            text,
            str(output_path)
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

    audio_url = (
        f"{RENDER_BASE_URL}"
        f"/audio/{filename}"
    )

    return {
        "success": True,
        "message": "TTS test successful",
        "audio_url": audio_url
    }


# ============================================================
# SERVER START MESSAGE
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )