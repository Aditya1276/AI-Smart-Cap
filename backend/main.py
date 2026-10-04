from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MODEL_NAME = "gemini-3.5-flash-lite"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Smart Cap Cloud Server is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": client is not None
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )

    image_data = await file.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Empty image received."
        )

    mime_type = file.content_type or "image/jpeg"

    prompt = """
You are the visual intelligence system of an AI Smart Cap
designed to assist a visually impaired user.

Analyze the supplied image carefully.

Identify:
1. Important everyday objects.
2. Whether a person is present.
3. Indian currency only if a currency note is clearly visible.
4. Approximate position of important objects:
   left, center, or right.
5. A short scene description suitable for spoken audio.

Rules:
- Do NOT identify or name people.
- Do NOT guess objects that are not clearly visible.
- Do NOT guess currency denomination.
- If currency denomination is unclear, use "unknown".
- Keep the spoken summary short.
- Return ONLY valid JSON.
- Do not use markdown.
- Confidence must be between 0 and 1.

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
    "confidence": 0
  },
  "summary": "A chair is on the left."
}
"""

    try:
        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                image_part,
                prompt
            ]
        )

        result_text = response.text.strip()

        # Remove accidental markdown code fences
        if result_text.startswith("```"):
            result_text = result_text.replace("```json", "")
            result_text = result_text.replace("```", "")
            result_text = result_text.strip()

        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {
                "person_present": False,
                "objects": [],
                "currency": {
                    "detected": False,
                    "denomination": "unknown",
                    "confidence": 0
                },
                "summary": result_text
            }

        return {
            "success": True,
            "analysis": result_json
        }

    except Exception as e:
        print(f"Gemini error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Gemini analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
