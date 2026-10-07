import os
import json
import time

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )


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
    title="AI Smart Cap API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GEMINI PROMPT
# =========================================================

VISION_PROMPT = """
You are the visual intelligence system of an AI Smart Cap
designed to assist a visually impaired user.

Analyze the supplied camera image carefully.

Your task is to identify useful visual information that can
be safely communicated through spoken audio.

RULES:

1. Detect important everyday objects.

2. Detect whether a person is visible.

3. NEVER identify a person's name or identity.

4. Detect Indian currency notes only when the denomination
   is clearly visible.

5. If currency denomination is unclear, use:
   denomination = "unknown"
   confidence = 0

6. Estimate object position using:
   - left
   - center
   - right

7. Do not invent objects.

8. Do not guess uncertain information.

9. Only report meaningful objects.

10. Keep the spoken summary short.

11. The summary must sound natural when spoken aloud.

12. Do not mention confidence values in the spoken summary.

13. If nothing useful is detected, say:
   "No important object detected."

Return ONLY valid JSON.

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
    "detected": true,
    "denomination": "500 INR",
    "confidence": 0.94
  },
  "summary": "A chair is on your left and a 500 rupee note is visible."
}

Confidence must be between 0 and 1.
"""


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def root():

    return {
        "success": True,
        "project": "AI Smart Cap",
        "status": "online"
    }


@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy"
    }


@app.get("/ping")
def ping():

    return {
        "success": True,
        "message": "pong"
    }


# =========================================================
# IMAGE ANALYSIS
# =========================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    server_start = time.perf_counter()

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_read_start = time.perf_counter()

    image_bytes = await file.read()

    image_read_time = (
        time.perf_counter()
        - image_read_start
    )

    if not image_bytes:

        return {
            "success": False,
            "error": "Empty image"
        }

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    gemini_start = time.perf_counter()

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                VISION_PROMPT,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ],

            config=types.GenerateContentConfig(

                thinking_config=
                types.ThinkingConfig(
                    thinking_level="minimal"
                )
            )
        )

    except Exception as e:

        return {
            "success": False,
            "error": "Gemini request failed",
            "details": str(e)
        }


    gemini_time = (
        time.perf_counter()
        - gemini_start
    )


    # -----------------------------------------------------
    # PARSE JSON
    # -----------------------------------------------------

    json_start = time.perf_counter()

    raw_text = response.text.strip()

    # Remove markdown fences if Gemini adds them.
    raw_text = raw_text.replace(
        "```json",
        ""
    )

    raw_text = raw_text.replace(
        "```",
        ""
    )

    raw_text = raw_text.strip()


    try:

        analysis = json.loads(
            raw_text
        )

    except Exception:

        # Safe fallback
        analysis = {

            "person_present": False,

            "objects": [],

            "currency": {

                "detected": False,

                "denomination": "unknown",

                "confidence": 0.0
            },

            "summary":
                "No important object detected."
        }


    json_parse_time = (
        time.perf_counter()
        - json_start
    )


    # -----------------------------------------------------
    # ENSURE REQUIRED FIELDS
    # -----------------------------------------------------

    person_present = bool(
        analysis.get(
            "person_present",
            False
        )
    )


    objects = analysis.get(
        "objects",
        []
    )


    currency = analysis.get(
        "currency",
        {
            "detected": False,
            "denomination": "unknown",
            "confidence": 0.0
        }
    )


    summary = analysis.get(
        "summary",
        "No important object detected."
    )


    # -----------------------------------------------------
    # AUDIO TEXT
    # -----------------------------------------------------

    audio_text = summary.strip()


    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    total_time = (
        time.perf_counter()
        - server_start
    )


    return {

        "success": True,

        "analysis": {

            "person_present":
                person_present,

            "objects":
                objects,

            "currency":
                currency,

            "summary":
                summary
        },

        "audio": {

            "text":
                audio_text
        },

        "latency": {

            "image_read_seconds":
                round(
                    image_read_time,
                    4
                ),

            "gemini_seconds":
                round(
                    gemini_time,
                    4
                ),

            "json_parse_seconds":
                round(
                    json_parse_time,
                    6
                ),

            "server_total_seconds":
                round(
                    total_time,
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

    start = time.perf_counter()

    image_bytes = await file.read()

    total = (
        time.perf_counter()
        - start
    )

    return {

        "success": True,

        "message":
            "Image received without Gemini",

        "image_size_bytes":
            len(image_bytes),

        "latency": {

            "server_total_seconds":
                round(
                    total,
                    6
                )
        }
    }