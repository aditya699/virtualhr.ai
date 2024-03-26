from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Access the API key
API_KEY = os.getenv("OPENAI_API_KEY")
