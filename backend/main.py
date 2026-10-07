from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types

from dotenv import load_dotenv

import os
import json
import time


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.1-flash-lite"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Smart Cap API",
    description="AI Smart Cap vision analysis backend"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Smart Cap API is running",
        "gemini_configured": client is not None,
        "model": MODEL_NAME
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "gemini_configured": client is not None,
        "model": MODEL_NAME
    }


# ============================================================
# IMAGE UPLOAD + GEMINI ANALYSIS
# ============================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # TOTAL SERVER TIMER
    # --------------------------------------------------------

    total_start = time.perf_counter()

    print()
    print("==============================================")
    print("NEW IMAGE REQUEST")
    print("==============================================")


    # --------------------------------------------------------
    # CHECK GEMINI
    # --------------------------------------------------------

    if client is None:

        print("ERROR: Gemini API key is not configured.")

        return {
            "success": False,
            "error": "Gemini API key is not configured"
        }


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    read_start = time.perf_counter()

    image_bytes = await file.read()

    read_end = time.perf_counter()

    read_time = read_end - read_start

    print(
        f"Image read time: {read_time:.3f} seconds"
    )

    print(
        f"Image size: {len(image_bytes)} bytes"
    )


    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = """
You are the visual intelligence system of an assistive wearable
for a visually impaired user.

Analyze the supplied image.

Identify:

1. Whether a person is present.
2. Important everyday objects.
3. Indian currency notes if clearly visible.
4. Approximate position of important objects:
   left, center, or right.

Do NOT identify people.

Do NOT guess or invent objects.

If currency denomination is unclear, return "unknown".

Keep the summary short and suitable for spoken audio.

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
"""


    # --------------------------------------------------------
    # GEMINI REQUEST
    # --------------------------------------------------------

    print()
    print("Sending image to Gemini...")

    gemini_start = time.perf_counter()

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                prompt
            ]
        )

    except Exception as e:

        gemini_end = time.perf_counter()

        gemini_time = gemini_end - gemini_start

        print(
            f"Gemini failed after: "
            f"{gemini_time:.3f} seconds"
        )

        print(f"Gemini error: {e}")

        return {
            "success": False,
            "error": str(e)
        }


    # --------------------------------------------------------
    # GEMINI TIMING
    # --------------------------------------------------------

    gemini_end = time.perf_counter()

    gemini_time = gemini_end - gemini_start

    print()
    print("----------------------------------------------")
    print(
        f"Gemini processing time: "
        f"{gemini_time:.3f} seconds"
    )
    print("----------------------------------------------")


    # --------------------------------------------------------
    # GET GEMINI TEXT
    # --------------------------------------------------------

    response_text = response.text.strip()

    print()
    print("Gemini raw response:")
    print("----------------------------------------------")
    print(response_text)
    print("----------------------------------------------")


    # --------------------------------------------------------
    # REMOVE MARKDOWN CODE BLOCK IF PRESENT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    parse_start = time.perf_counter()

    try:

        analysis = json.loads(response_text)

    except json.JSONDecodeError:

        print("WARNING: Gemini returned invalid JSON.")

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

    parse_end = time.perf_counter()

    parse_time = parse_end - parse_start

    print(
        f"JSON parsing time: "
        f"{parse_time:.6f} seconds"
    )


    # --------------------------------------------------------
    # TOTAL SERVER TIME
    # --------------------------------------------------------

    total_end = time.perf_counter()

    total_time = total_end - total_start

    print()
    print("==============================================")
    print("SERVER LATENCY")
    print("==============================================")

    print(
        f"Image read       : {read_time:.3f} sec"
    )

    print(
        f"Gemini           : {gemini_time:.3f} sec"
    )

    print(
        f"JSON parsing     : {parse_time:.6f} sec"
    )

    print(
        f"TOTAL SERVER     : {total_time:.3f} sec"
    )

    print("==============================================")
    print()


    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,

        "analysis": analysis,

        "latency": {
            "image_read_seconds": round(
                read_time,
                3
            ),

            "gemini_seconds": round(
                gemini_time,
                3
            ),

            "json_parse_seconds": round(
                parse_time,
                6
            ),

            "server_total_seconds": round(
                total_time,
                3
            )
        }
    }