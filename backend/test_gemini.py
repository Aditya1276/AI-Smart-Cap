from google import genai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit()

# Create Gemini client
client = genai.Client(api_key=api_key)

# Test Gemini
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain in one short sentence what an AI Smart Cap is."
)

print("\nGemini Response:")
print(response.text)