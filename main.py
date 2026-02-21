from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"API key loaded! Starts with: {api_key[:8]}...")
else:
    print("❌ API key not found — check your .env file")