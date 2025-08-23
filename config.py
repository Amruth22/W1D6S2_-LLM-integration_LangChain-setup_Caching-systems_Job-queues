from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# For memory broker, we can define it directly in celery_app.py or here
# CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "memory://")