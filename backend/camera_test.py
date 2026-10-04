import cv2
from google import genai
from dotenv import load_dotenv
import os
import time

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit()

# Gemini client
client = genai.Client(api_key=api_key)

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open laptop camera")
    exit()

print("Camera started.")
print("Press SPACE to capture an image.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    # Show camera
    cv2.imshow("AI Smart Cap - Laptop Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE = capture
    if key == 32:

        print("\nImage captured. Sending to Gemini...")

        # Save captured image temporarily
        image_path = "camera_capture.jpg"
        cv2.imwrite(image_path, frame)

        # Upload image to Gemini
        image = client.files.upload(file=image_path)

        # Ask Gemini to analyze image
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                image,
                """
                You are the visual intelligence system of an AI Smart Cap
                designed to assist a visually impaired user.

                Analyze this image and identify:

                1. Important everyday objects
                2. Whether a person is present
                3. Indian currency notes if clearly visible
                4. Approximate position of important objects:
                   left, center, or right
                5. A short description of the scene

                Do not identify the person.
                Do not guess objects that are unclear.
                If currency denomination is unclear, say unknown.

                Keep the response short and suitable for converting
                into speech using text-to-speech.

                Example:
                "A person is in front. A chair is on the left and
                a table is in the center."
                """
            ]
        )

        print("\nGemini Response:")
        print(response.text)

        print("\nPress SPACE to capture another image.")
        print("Press Q to quit.")

        # Small delay
        time.sleep(1)

    # Q = quit
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()