import os
import json
import time

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
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
# GEMINI
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
    version="1.0"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "success": True,
        "message": "AI Smart Cap backend is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "success": True,
        "status": "healthy"
    }


# ============================================================
# PING TEST
# ============================================================

@app.get("/ping")
async def ping():

    return {
        "success": True,
        "message": "pong"
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    server_start = time.perf_counter()


    # ========================================================
    # READ IMAGE
    # ========================================================

    image_read_start = time.perf_counter()

    image_bytes = await file.read()

    image_read_end = time.perf_counter()

    image_read_seconds = (
        image_read_end - image_read_start
    )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image.

Identify:

1. Whether a person is present.
2. Important everyday objects.
3. Approximate position of objects:
   - left
   - center
   - right
4. Indian currency notes if clearly visible.
5. A short description suitable for spoken audio.

Rules:

- Do not identify people.
- Do not guess or invent objects.
- Only report objects that are reasonably visible.
- If currency denomination is unclear, return unknown.
- Keep the summary short.
- Return ONLY valid JSON.

Use exactly this structure:

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
"""


    # ========================================================
    # GEMINI
    # ========================================================

    gemini_start = time.perf_counter()

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

    gemini_end = time.perf_counter()

    gemini_seconds = (
        gemini_end - gemini_start
    )


    # ========================================================
    # PARSE GEMINI RESPONSE
    # ========================================================

    json_parse_start = time.perf_counter()

    text = response.text.strip()


    # Remove markdown JSON fences if Gemini returns them

    if text.startswith("```json"):

        text = text[7:]

    elif text.startswith("```"):

        text = text[3:]


    if text.endswith("```"):

        text = text[:-3]


    text = text.strip()


    try:

        analysis = json.loads(text)

    except Exception:

        analysis = {
            "person_present": False,

            "objects": [],

            "currency": {
                "detected": False,
                "denomination": "unknown",
                "confidence": 0.0
            },

            "summary": text
        }


    json_parse_end = time.perf_counter()

    json_parse_seconds = (
        json_parse_end - json_parse_start
    )


    # ========================================================
    # TOTAL SERVER TIME
    # ========================================================

    server_end = time.perf_counter()

    server_total_seconds = (
        server_end - server_start
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success": True,

        "analysis": analysis,

        "latency": {

            "image_read_seconds":
                round(image_read_seconds, 4),

            "gemini_seconds":
                round(gemini_seconds, 4),

            "json_parse_seconds":
                round(json_parse_seconds, 6),

            "server_total_seconds":
                round(server_total_seconds, 4)
        }
    }