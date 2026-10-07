import os
import json
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Smart Cap API",
    description="Vision backend for AI Smart Cap",
    version="1.0.0"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "AI Smart Cap backend is running",
        "model": MODEL_NAME
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    total_start = time.perf_counter()


    # ========================================================
    # READ IMAGE
    # ========================================================

    image_start = time.perf_counter()

    image_bytes = await file.read()

    image_end = time.perf_counter()

    image_read_time = image_end - image_start


    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    if not image_bytes:

        return {
            "success": False,
            "error": "Empty image received"
        }


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image.

Identify only important information that would be useful to the user:

1. Whether a person is present.
2. Important everyday objects.
3. Indian currency notes if clearly visible.
4. Approximate position of important objects:
   - left
   - center
   - right

Do NOT identify or name people.

Do NOT guess objects.

If something is unclear, do not invent it.

For Indian currency:
- Only report a denomination when it is clearly visible.
- Otherwise return "unknown".

Keep the response concise because it will eventually be converted
into spoken audio.

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
  "summary": "A person is present. A chair is on the left."
}

Confidence must be a number between 0 and 1.
"""


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    gemini_start = time.perf_counter()

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                prompt,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ],

            config=types.GenerateContentConfig(

                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                )
            )
        )

    except Exception as e:

        gemini_end = time.perf_counter()

        gemini_time = gemini_end - gemini_start

        total_end = time.perf_counter()

        total_time = total_end - total_start

        return {
            "success": False,
            "error": str(e),
            "latency": {
                "image_read_seconds": round(
                    image_read_time,
                    4
                ),
                "gemini_seconds": round(
                    gemini_time,
                    4
                ),
                "server_total_seconds": round(
                    total_time,
                    4
                )
            }
        }


    gemini_end = time.perf_counter()

    gemini_time = gemini_end - gemini_start


    # ========================================================
    # GET GEMINI TEXT
    # ========================================================

    response_text = response.text.strip()


    # ========================================================
    # REMOVE MARKDOWN CODE BLOCK IF PRESENT
    # ========================================================

    if response_text.startswith("```"):

        response_text = response_text.replace(
            "```json",
            ""
        )

        response_text = response_text.replace(
            "```",
            ""
        )

        response_text = response_text.strip()


    # ========================================================
    # PARSE JSON
    # ========================================================

    json_start = time.perf_counter()

    try:

        analysis = json.loads(
            response_text
        )

    except json.JSONDecodeError:

        analysis = {
            "person_present": False,
            "objects": [],
            "currency": {
                "detected": False,
                "denomination": "unknown",
                "confidence": 0.0
            },
            "summary": response_text
        }

    json_end = time.perf_counter()

    json_parse_time = (
        json_end - json_start
    )


    # ========================================================
    # TOTAL SERVER TIME
    # ========================================================

    total_end = time.perf_counter()

    total_time = (
        total_end - total_start
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success": True,

        "analysis": analysis,

        "latency": {

            "image_read_seconds": round(
                image_read_time,
                4
            ),

            "gemini_seconds": round(
                gemini_time,
                4
            ),

            "json_parse_seconds": round(
                json_parse_time,
                6
            ),

            "server_total_seconds": round(
                total_time,
                4
            )
        }
    }