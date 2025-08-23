from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
# Import the LangChain function with integrated LRU cache
from app.langchain_setup import get_cached_llm_response
# Import the Celery task
from app.tasks import generate_content_task
from app.celery_app import celery_app

app = FastAPI(title="LLM API with LangChain, LRU Cache, and Job Queue")

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


@app.get("/")
def read_root():
    return {"message": "LLM API with LangChain, LRU Cache, and Job Queue is running!"}


@app.post("/generate/", response_model=ResultResponse)
def generate_sync(request: QuestionRequest):
    """
    Synchronously generates content using the LLM via LangChain with LRU caching.
    """
    question = request.question

    try:
        # The caching logic (LRU) and LangChain call are handled by this function.
        response_text = get_cached_llm_response(question)

        return ResultResponse(
            task_id=str(uuid.uuid4()), # Generate a dummy ID for sync calls
            status="SUCCESS",
            result=response_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call via LangChain failed: {str(e)}")


@app.post("/generate-async/", response_model=TaskResponse)
def generate_async(request: QuestionRequest):
    """
    Asynchronously generates content using the LLM via a Celery task.
    The Celery task uses LangChain with LRU caching.
    Returns a task ID immediately.
    """
    question = request.question
    # Send the task to the Celery queue
    task = generate_content_task.delay(question)
    return TaskResponse(task_id=task.id)


@app.get("/tasks/{task_id}", response_model=ResultResponse)
def get_task_status(task_id: str):
    """
    Gets the status and result of an asynchronous task.
    """
    # Get the task instance from Celery
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        # Task is waiting to be processed or unknown ID
        return ResultResponse(task_id=task_id, status=task.state)
    elif task.state == 'SUCCESS':
        # Task completed successfully
        response_text = task.result
        return ResultResponse(task_id=task_id, status=task.state, result=response_text)
    elif task.state == 'FAILURE':
        # Task failed
        error_details = str(task.info) if task.info else "Unknown error"
        return ResultResponse(task_id=task_id, status=task.state, error=error_details)
    else:
        # Task is in another state (e.g., STARTED, RETRY)
        return ResultResponse(task_id=task_id, status=task.state)