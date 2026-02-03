from __future__ import annotations
import os, threading, concurrent.futures, time, re
from functools import lru_cache, wraps
import oracledb
import asyncio
from typing import Tuple, Dict, Any, Callable
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from anthropic import AnthropicFoundry
from pypdf import PdfReader

ENV_PATH = os.getenv("ENV_PATH", ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Oracle client initialization
instant_client_path = r"C:\Users\yRK3KOR\Desktop\instantclient_23_0"
try:
    oracledb.init_oracle_client(lib_dir=instant_client_path)
    print("Oracle Client initialized in Thick mode.")
except oracledb.Error as e:
    print(f"Error initializing Oracle Client: {e}")

# =========================================================
# MODEL CONFIGURATIONS
# =========================================================
MODELS = {
    "Claude Sonnet 4.5": {
        "provider": "anthropic_foundry",
        "deployment": "claude-sonnet-4-5-saarathi02",
        "endpoint": "https://bdo-internal01-resource.services.ai.azure.com/anthropic/",
    },
    "GPT-5": {
        "provider": "azure_openai",
        "deployment": "gpt-5-Saarathi",
        "endpoint": "https://bdo-internal01-resource.cognitiveservices.azure.com",
        "api_version": "2025-01-01-preview",
        "token_param": "max_completion_tokens",
        "supports_temperature": False
    },
    "GPT-4o": {
        "provider": "azure_openai",
        "deployment": "gpt-4o_Interns01",
        "endpoint": "https://bdo-internal01-resource.cognitiveservices.azure.com",
        "api_version": "2025-01-01-preview",
        "token_param": "max_completion_tokens",
        "supports_temperature": True
    },
    "DeepSeek-R1": {
        "provider": "azure_openai",
        "deployment": "DeepSeek-R1",
        "endpoint": "https://bdo-internal01-resource.services.ai.azure.com",
        "api_version": "2024-05-01-preview",
        "token_param": "max_tokens",
        "supports_temperature": False,
        "strip_think": True
    },
    "DeepSeek-V3.1": {
        "provider": "azure_openai",
        "deployment": "DeepSeek-V3.1",
        "endpoint": "https://bdo-internal01-resource.services.ai.azure.com",
        "api_version": "2024-05-01-preview",
        "token_param": "max_tokens",
        "supports_temperature": False,
        "strip_think": True
    }
}

# GPT-4o configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# GPT-5 configuration
AZURE_GPT5_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_GPT5_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_GPT5_API_VER = os.getenv("AZURE_GPT5_API_VERSION", "2024-12-01-preview")
AZURE_GPT5_DEPLOYMENT = os.getenv("AZURE_GPT5_DEPLOYMENT", "gpt-5-Saarathi")

# Claude/Anthropic Foundry configuration 
CLAUDE_ENDPOINT_BASE = os.getenv("AZURE_ENDPOINT", "")
CLAUDE_ENDPOINT = (CLAUDE_ENDPOINT_BASE.rstrip("/") + "/anthropic") if CLAUDE_ENDPOINT_BASE else ""
CLAUDE_API_KEY = os.getenv("AZURE_API_KEY", "")
CLAUDE_DEPLOYMENT = os.getenv("AZURE_CLAUDE_DEPLOYMENT", "claude-sonnet-4-5-saarathi02")
CLAUDE_MAX_RETRIES = int(os.getenv("CLAUDE_MAX_RETRIES", "3"))
CLAUDE_RETRY_DELAY = float(os.getenv("CLAUDE_RETRY_DELAY", "1.0"))

# Database configuration
ORACLE_USERNAME = os.getenv("ORACLE_USERNAME")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE_NAME = os.getenv("ORACLE_SERVICE_NAME")
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", 120))

# Docupedia configuration
DOCUPEDIA_PAT = os.getenv("DOCUPEDIA_PAT")
DOCUPEDIA_BASE_URL = os.getenv("DOCUPEDIA_BASE_URL")

# Thread pool configuration
DB_THREAD_WORKERS = int(os.getenv("DB_THREAD_WORKERS", 8))
AI_THREAD_WORKERS = int(os.getenv("AI_THREAD_WORKERS", 16))

DB_POOL_LOCK = threading.Lock()
DB_POOL: oracledb.ConnectionPool | None = None

DB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=DB_THREAD_WORKERS)
AI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=AI_THREAD_WORKERS)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_text_from_files(files):
    """Extract text from uploaded PDF or TXT files."""
    text = ""
    for f in files:
        if f.type == "application/pdf":
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        else:
            text += f.read().decode("utf-8")
    return text[:12000]

def clean_response(text, strip_think=False):
    """Clean response text, optionally removing <think> tags."""
    if strip_think:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# Retry decorator for Claude API calls
def retry_on_rate_limit(max_retries: int = CLAUDE_MAX_RETRIES, delay: float = CLAUDE_RETRY_DELAY):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    if "rate" in error_msg or "429" in error_msg or "overload" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = delay * (2 ** attempt)
                            print(f"⚠️ Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                    raise
            raise last_exception
        return wrapper
    return decorator

# Oracle DB connection pool initialization
def _init_pool() -> None:
    global DB_POOL
    if DB_POOL is not None:
        return
    
    with DB_POOL_LOCK:
        if DB_POOL is not None:
            return
        
        DB_POOL = oracledb.create_pool(
            user=ORACLE_USERNAME,
            password=ORACLE_PASSWORD,
            host=ORACLE_HOST,
            port=ORACLE_PORT,
            service_name=ORACLE_SERVICE_NAME,
            min=1,
            max=DB_THREAD_WORKERS,
            increment=1,
            timeout=DB_POOL_TIMEOUT,
        )

def get_db_connection():
    _init_pool()
    return DB_POOL.acquire()

# Context manager for safe DB connection handling
class db_connection:
    def __enter__(self):
        _init_pool()
        self.conn = DB_POOL.acquire()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        return False

# Execute blocking DB functions (returns Future for non-blocking execution)
def run_in_executor(fn, *args, **kwargs):
    return DB_EXECUTOR.submit(fn, *args, **kwargs)

async def run_in_executor_async(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(DB_EXECUTOR, lambda: fn(*args, **kwargs))

# Execute AI functions (returns Future for non-blocking execution)
def run_ai_blocking(fn, *args, **kwargs):
    return AI_EXECUTOR.submit(fn, *args, **kwargs)

async def run_ai_async(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(AI_EXECUTOR, lambda: fn(*args, **kwargs))


# Wrapper class with retry logic for Claude
class _ClientWrapper:
    def __init__(self, client, client_type):
        self.client = client
        self.client_type = client_type
    
    def invoke(self, prompt, **kwargs):
        if self.client_type == "gpt4o":
            response = self.client.invoke(prompt)
            return response
        elif self.client_type == "gpt5":
            response = self.client.chat.completions.create(
                model=AZURE_GPT5_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            class _Response:
                def __init__(self, text):
                    self.content = text
            return _Response(response.choices[0].message.content)
        elif self.client_type == "claude":
            return self._invoke_claude_with_retry(prompt, **kwargs)
    
    @retry_on_rate_limit()
    def _invoke_claude_with_retry(self, prompt, **kwargs):
        response = self.client.messages.create(
            model=CLAUDE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        class _Response:
            def __init__(self, text):
                self.content = text
        return _Response(response.content[0].text)


# GPT-4o client with configurable temperature (cached)
@lru_cache(maxsize=1)
def get_llm(temp: float = 0.0):
    client = AzureChatOpenAI(
        deployment_name=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VER,
        temperature=temp,
    )
    return _ClientWrapper(client, "gpt4o")

# GPT-5 client (cached)
@lru_cache(maxsize=1)
def get_gpt5():
    client = AzureOpenAI(
        api_version=AZURE_GPT5_API_VER,
        azure_endpoint=AZURE_GPT5_ENDPOINT,
        api_key=AZURE_GPT5_API_KEY,
    )
    return _ClientWrapper(client, "gpt5")

# Claude client with retry logic (cached)
@lru_cache(maxsize=1)
def get_claude():
    client = AnthropicFoundry(
        api_key=CLAUDE_API_KEY,
        base_url=CLAUDE_ENDPOINT,
    )
    return _ClientWrapper(client, "claude")

# Backwards compatibility aliases
def get_gpt5_model():
    return get_gpt5()

def get_claude_model():
    return get_claude()

# Clear all cached resources
def clear_all_caches():
    get_llm.cache_clear()
    get_gpt5.cache_clear()
    get_claude.cache_clear()
    get_embeddings.cache_clear()
    get_openai_client.cache_clear()
    get_docupedia_config.cache_clear()
    print("✅ All caches cleared")



# Cached embedding model
@lru_cache(maxsize=1)
def get_embeddings() -> AzureOpenAIEmbeddings:
    try:
        return AzureOpenAIEmbeddings(
            deployment=EMBED_DEPLOYMENT,
            openai_api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_API_VER,
        )
    except Exception as e:
        print(f"⚠️ Embedding init failed: {e} — using no-op fallback")
        class _ZeroEmb(AzureOpenAIEmbeddings):
            def embed_query(self, text: str): return [0.0] * 10
            def embed_documents(self, texts): return [[0.0] * 10 for _ in texts]
        return _ZeroEmb()

# Cached direct OpenAI client
@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VER,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

# Cached Docupedia config (only if credentials provided)
@lru_cache(maxsize=1)
def get_docupedia_config() -> Tuple[str, str]:
    if not DOCUPEDIA_PAT or not DOCUPEDIA_BASE_URL:
        raise ValueError("Docupedia credentials not configured")
    return DOCUPEDIA_PAT, DOCUPEDIA_BASE_URL

# System health snapshot
def shared_health_snapshot() -> Dict[str, Any]:
    pool_info = {}
    if DB_POOL:
        try:
            pool_info = {
                "open": DB_POOL.open_count,
                "busy": getattr(DB_POOL, "busy_count", None),
                "available": DB_POOL.open_count - getattr(DB_POOL, "busy_count", 0)
            }
        except Exception:
            pool_info = {"error": "unavailable"}

    return {
        "timestamp": time.time(),
        "db_pool": pool_info,
        "embeddings_cached": bool(get_embeddings.cache_info().currsize),
        "docupedia_configured": bool(DOCUPEDIA_PAT and DOCUPEDIA_BASE_URL),
        "db_executor_pending": DB_EXECUTOR._work_queue.qsize(),
        "ai_executor_pending": AI_EXECUTOR._work_queue.qsize(),
    }

# Preload resources at server startup
def warm_start():
    _init_pool()
    get_llm()
    get_gpt5()
    get_claude()
    get_embeddings()
    if DOCUPEDIA_PAT and DOCUPEDIA_BASE_URL:
        try:
            get_docupedia_config()
        except Exception as e:
            print(f"⚠️ Docupedia config failed: {e}")
    print("✅ Warm start complete.")

# =========================================================
# CHAT COMPLETION FUNCTIONS
# =========================================================
def create_chat_stream(model_name, messages, temperature, max_tokens, file_context=""):
    """
    Create a streaming chat completion for the specified model.
    Returns a generator that yields response chunks.
    """
    cfg = MODELS[model_name]
    api_key = AZURE_API_KEY
    
    if cfg["provider"] == "azure_openai":
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=cfg["endpoint"],
            api_version=cfg["api_version"]
        )
        
        messages_copy = messages.copy()
        if file_context:
            messages_copy.insert(1, {
                "role": "system",
                "content": f"Context:\n{file_context}"
            })
        
        args = {
            "model": cfg["deployment"],
            "messages": messages_copy,
            "stream": True
        }
        args[cfg["token_param"]] = max_tokens
        
        if cfg.get("supports_temperature"):
            args["temperature"] = temperature
        
        stream = client.chat.completions.create(**args)
        
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
    
    elif cfg["provider"] == "anthropic_foundry":
        client = AnthropicFoundry(
            api_key=api_key,
            base_url=cfg["endpoint"]
        )
        
        # Get the last user message
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        prompt = user_messages[-1]["content"] if user_messages else ""
        
        message = client.messages.create(
            model=cfg["deployment"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        full_response = "".join(block.text for block in message.content)
        yield full_response

# Graceful shutdown - cleanup all resources
def graceful_shutdown():
    global DB_POOL
    try:
        clear_all_caches()
        
        if DB_POOL:
            DB_POOL.close()
            DB_POOL = None
            print("✅ DB pool closed")
        
        DB_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        print("✅ DB executor shutdown")
        
        AI_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        print("✅ AI executor shutdown")
        
        print("✅ Graceful shutdown complete")
    except Exception as e:
        print(f"⚠️ Shutdown error: {e}")

