from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache
import hashlib
# Import config from the project root package (one level up)
from config import GEMINI_API_KEY

# --- 1. Initialize the LLM ---
# Ensure GEMINI_API_KEY is set in your environment
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in the .env file.")

# Updated model name to a more commonly available one
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)

# --- 2. Define a Prompt Template ---
# A simple prompt template that takes user input
prompt_template = PromptTemplate.from_template("Answer the following question: {question}")

# --- 3. Create a Basic Chain ---
# This chain combines the prompt, LLM, and a simple string output parser
basic_chain = (
    {"question": RunnablePassthrough()} # Pass the input directly as 'question'
    | prompt_template
    | llm
    | StrOutputParser()
)

# --- 4. Define a Cached Wrapper Function ---
# This function wraps the chain invocation and applies LRU caching.

@lru_cache(maxsize=128)
def _cached_chain_invoke(question: str) -> str:
    """
    Internal function to invoke the LangChain and apply Lru_cache.
    The question is the cache key.

    Args:
        question (str): The user's question/prompt.

    Returns:
        str: The LLM's response text.
    """
    # The actual call to the LangChain
    response = basic_chain.invoke(question)
    return response

def get_cached_llm_response(question: str) -> str:
    """
    Gets a cached LLM response for a given question, or calls the LLM if not cached.

    Args:
        question (str): The user's question/prompt.

    Returns:
        str: The LLM's response text.
    """
    return _cached_chain_invoke(question)