from app.celery_app import celery_app
# Import the LangChain setup function that includes LRU caching
from app.langchain_setup import get_cached_llm_response
# Import config to ensure it's loaded for the worker process
from config import GEMINI_API_KEY
import logging

# Configure logging for tasks
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def generate_content_task(self, question: str) -> str:
    """
    Celery task to generate content using the LLM via LangChain.
    This task utilizes the LRU caching implemented within LangChain setup.

    Args:
        self: The Celery task instance.
        question (str): The user's question/prompt.

    Returns:
        str: The LLM's response text (either from cache or a new call via LangChain).
    """
    logger.info(f"Task {self.request.id} started for question: {question}")

    try:
        # This call will check the LRU cache (managed by langchain_setup.py)
        # and call the LLM via LangChain if necessary.
        response_text = get_cached_llm_response(question)

        logger.info(f"Task {self.request.id} completed for question: {question}")
        return response_text

    except Exception as exc:
        logger.error(f"LLM call via LangChain failed for question '{question}': {exc}")
        # Re-raise to mark the task as failed
        raise