from celery import Celery

# Create the Celery application instance
# Using an in-memory broker for simplicity and to avoid Redis dependency.
# Note: The 'memory' broker is for development/testing and won't work
# across separate processes (e.g., main app and celery worker CLI).
# For production or true async, a real broker like Redis or RabbitMQ is needed.
celery_app = Celery(
    'llm_tasks',
    broker='memory://',      # In-memory broker
    backend='cache+memory://' # In-memory result backend
)

# Optional: Configure Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_default_queue='llm-default',
    # For in-memory backend, configure the cache
    cache_backend_options={'url': 'memory://'},
    result_backend='cache+memory://'
)