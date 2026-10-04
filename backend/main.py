from fastapi import FastAPI, File, UploadFile
from datetime import datetime
import os

# Create FastAPI application
app = FastAPI()

# Folder where ESP32-CAM images will be saved
UPLOAD_FOLDER = "received_images"

# Create folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Test endpoint
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Smart Cap server is running"
    }


# Image upload endpoint
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    # Create a timestamp for the image filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"capture_{timestamp}.jpg"

    # Complete file path
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Read image data
    image_data = await file.read()

    # Save image
    with open(filepath, "wb") as f:
        f.write(image_data)

    print(f"Image received: {filename}")
    print(f"Image size: {len(image_data)} bytes")

    return {
        "success": True,
        "filename": filename,
        "size": len(image_data)
    }