from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
# Import from the 'app' package
from app.langchain_setup import get_cached_llm_response
from app.tasks import generate_content_task
from app.celery_app import celery_app
# Import config from the project root
from config import GEMINI_API_KEY
import uvicorn
import os

# Basic check for API key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")

app = FastAPI(title="LLM API with LangChain, LRU Cache, and In-Memory Job Queue")

# Pydantic models for request/response bodies
class QuestionRequest(BaseModel):
    question: str

class TaskResponse(BaseModel):
    task_id: str

class ResultResponse(BaseModel):
    task_id: str
    status: str # 'SUCCESS', 'PENDING', 'FAILURE'
    result: str = None
    error: str = None

class GenerateResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def read_root():
    return {"message": "LLM API with LangChain, LRU Cache, and In-Memory Job Queue is running!"}


@app.post("/generate/", response_model=GenerateResponse)
def generate_sync(request: QuestionRequest):
    """
    Synchronously generates content using the LLM via LangChain with LRU caching.
    """
    question = request.question

    try:
        answer = get_cached_llm_response(question)
        return GenerateResponse(question=question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call via LangChain failed: {str(e)}")


@app.post("/generate-async/", response_model=TaskResponse)
def generate_async(request: QuestionRequest):
    """
    Asynchronously generates content using the LLM via a Celery task.
    The Celery task uses LangChain with LRU caching.
    Returns a task ID immediately.
    Note: With the in-memory broker, this works within the same process.
    For true async, a separate worker process with a real broker is needed.
    """
    question = request.question
    task = generate_content_task.delay(question)
    return TaskResponse(task_id=task.id)


@app.get("/tasks/{task_id}", response_model=ResultResponse)
def get_task_status(task_id: str):
    """
    Gets the status and result of an asynchronous task.
    Note: With the in-memory broker, this works within the same process.
    """
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        return ResultResponse(task_id=task_id, status=task.state)
    elif task.state == 'SUCCESS':
        response_text = task.result
        return ResultResponse(task_id=task_id, status=task.state, result=response_text)
    elif task.state == 'FAILURE':
        error_details = str(task.info) if task.info else "Unknown error"
        return ResultResponse(task_id=task_id, status=task.state, error=error_details)
    else:
        return ResultResponse(task_id=task_id, status=task.state)


# This block allows running the application directly with `python main.py`
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    reload = os.getenv("DEV_MODE", "false").lower() == "true"

    # Note: With 'memory://' broker, the worker is effectively in the same process.
    # For a truly separate worker, you'd run `celery -A app.celery_app worker` in another terminal,
    # but it would need a real broker like Redis/RabbitMQ to communicate.
    uvicorn.run("main:app", host=host, port=port, reload=reload)