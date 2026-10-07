import os
import json
import time

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
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
    title="AI Smart Cap Backend",
    description="ESP32-CAM -> Gemini Vision Backend",
    version="1.0"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "AI Smart Cap backend is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy"
    }


# ============================================================
# PING
# ============================================================

@app.get("/ping")
def ping():

    return {
        "success": True,
        "message": "pong"
    }


# ============================================================
# GEMINI PROMPT
# ============================================================

PROMPT = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image carefully.

Identify:

1. Whether a person is present.
2. Important everyday objects visible in the image.
3. Approximate position of each important object:
   - left
   - center
   - right
4. Indian currency notes if clearly visible.
5. A short description suitable for spoken audio.

IMPORTANT RULES:

- Do not identify people.
- Do not guess objects that are not clearly visible.
- Do not invent information.
- Only report important objects.
- For currency, identify the denomination only when reasonably clear.
- If currency is unclear, use "unknown".
- Keep the summary short and natural for text-to-speech.
- Confidence must be between 0 and 1.

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
  "summary": "A person is present. A chair is on the left and a 500 rupee note is visible."
}

If no currency is visible:

{
  "detected": false,
  "denomination": "unknown",
  "confidence": 0.0
}
"""


# ============================================================
# UPLOAD IMAGE
# ============================================================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    server_start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_start = time.perf_counter()

        image_bytes = await file.read()

        image_read_time = time.perf_counter() - image_start

        if not image_bytes:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Empty image received"
                }
            )


        print()
        print("========================================")
        print("IMAGE RECEIVED")
        print("========================================")

        print("Filename:", file.filename)
        print("Content type:", file.content_type)
        print("Image size:", len(image_bytes), "bytes")


        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        gemini_start = time.perf_counter()

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

                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                )
            )
        )

        gemini_time = time.perf_counter() - gemini_start


        # ----------------------------------------------------
        # GET GEMINI TEXT
        # ----------------------------------------------------

        raw_text = response.text.strip()

        print()
        print("GEMINI RESPONSE:")
        print(raw_text)


        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        json_start = time.perf_counter()

        try:

            # Remove markdown JSON fences if Gemini returns them
            cleaned_text = raw_text

            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]

            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]

            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            cleaned_text = cleaned_text.strip()

            analysis = json.loads(cleaned_text)

        except json.JSONDecodeError:

            print("WARNING: Gemini returned invalid JSON")

            analysis = {
                "person_present": False,
                "objects": [],
                "currency": {
                    "detected": False,
                    "denomination": "unknown",
                    "confidence": 0.0
                },
                "summary": raw_text
            }


        json_parse_time = time.perf_counter() - json_start


        # ----------------------------------------------------
        # TOTAL SERVER TIME
        # ----------------------------------------------------

        server_total_time = time.perf_counter() - server_start


        # ----------------------------------------------------
        # PRINT LATENCY
        # ----------------------------------------------------

        print()
        print("========================================")
        print("LATENCY")
        print("========================================")

        print(
            "Image read:",
            round(image_read_time, 4),
            "seconds"
        )

        print(
            "Gemini:",
            round(gemini_time, 4),
            "seconds"
        )

        print(
            "JSON parse:",
            round(json_parse_time, 6),
            "seconds"
        )

        print(
            "Server total:",
            round(server_total_time, 4),
            "seconds"
        )

        print("========================================")
        print()


        # ----------------------------------------------------
        # RESPONSE TO ESP32
        # ----------------------------------------------------

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
                    server_total_time,
                    4
                )
            }
        }


    except Exception as e:

        print()
        print("========================================")
        print("ERROR")
        print("========================================")

        print(str(e))

        print("========================================")


        return JSONResponse(

            status_code=500,

            content={
                "success": False,
                "error": str(e)
            }
        )


# ============================================================
# UPLOAD TEST
# ============================================================

@app.post("/upload-test")
async def upload_test(file: UploadFile = File(...)):

    start = time.perf_counter()

    try:

        image_bytes = await file.read()

        total_time = time.perf_counter() - start

        return {
            "success": True,

            "message": "Image received without Gemini",

            "filename": file.filename,

            "image_size_bytes": len(image_bytes),

            "latency": {
                "server_total_seconds": round(
                    total_time,
                    6
                )
            }
        }

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "success": False,
                "error": str(e)
            }
        )