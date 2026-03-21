import os
# CRITICAL: Disable progress bars BEFORE any other imports
os.environ['TQDM_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Monkey-patch tqdm to disable all progress bars BEFORE any imports
import sys
import builtins
import io

# Create a no-op tqdm class that does nothing
class NoOpTqdm:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, *args, **kwargs):
        pass
    def close(self):
        pass
    def __iter__(self):
        return iter([])
    def __call__(self, *args, **kwargs):
        return self
    def __bool__(self):
        return False
    def refresh(self, *args, **kwargs):
        pass
    def set_description(self, *args, **kwargs):
        pass
    def set_postfix(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs):
        pass

# Patch tqdm before it's imported anywhere
try:
    import tqdm
    tqdm.tqdm = NoOpTqdm
    if hasattr(tqdm, 'auto'):
        tqdm.auto.tqdm = NoOpTqdm
    if hasattr(tqdm, 'tqdm'):
        tqdm.tqdm = NoOpTqdm
    # Patch all tqdm variants
    for attr in dir(tqdm):
        if 'tqdm' in attr.lower():
            try:
                setattr(tqdm, attr, NoOpTqdm)
            except:
                pass
except (ImportError, AttributeError):
    # Create a dummy module if tqdm isn't imported yet
    class TqdmModule:
        tqdm = NoOpTqdm
        auto = type('auto', (), {'tqdm': NoOpTqdm})()
        trange = lambda *args, **kwargs: iter([])
    sys.modules['tqdm'] = TqdmModule()
    builtins.tqdm = NoOpTqdm
    builtins.trange = lambda *args, **kwargs: iter([])

# Also patch sys.modules to prevent tqdm from being imported with progress bars
if 'tqdm' not in sys.modules:
    sys.modules['tqdm'] = type('Module', (), {
        'tqdm': NoOpTqdm,
        'auto': type('auto', (), {'tqdm': NoOpTqdm})(),
        'trange': lambda *args, **kwargs: iter([])
    })()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from settings import settings
import json
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

# Retry config for ChromaDB rate limits (429 Too Many Requests)
_CHROMA_RATE_LIMIT_RETRIES = 3
_CHROMA_RATE_LIMIT_BACKOFF_BASE = 2.0  # seconds

log = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZED CHROMADB INITIALIZATION
# ============================================================================
# Initialize optimized ChromaDB manager for better performance
# Falls back to direct client if optimization fails (backward compatibility)

_optimized_manager: Optional[object] = None
_use_optimized = True  # Feature flag to enable/disable optimization
_optimized_manager_initializing = False

def _get_optimized_manager():
    """Lazy initialization of optimized manager (sync-safe)"""
    global _optimized_manager, _use_optimized, _optimized_manager_initializing
    
    if not _use_optimized:
        return None
    
    if _optimized_manager is not None:
        return _optimized_manager
    
    if _optimized_manager_initializing:
        return None  # Already initializing, return None to use direct client
    
    try:
        from core.chromadb_optimizer import get_optimized_chromadb, ChromaDBOptimizerConfig
        
        # Try to initialize if event loop exists and is not running
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                _optimized_manager_initializing = True
                config = ChromaDBOptimizerConfig(
                    enable_query_cache=settings.USE_LLM_CACHE if hasattr(settings, 'USE_LLM_CACHE') else True,
                    cache_ttl_seconds=getattr(settings, 'CACHE_TTL_SECONDS', 300),
                    batch_size=100,
                    enable_metrics=True
                )
                _optimized_manager = loop.run_until_complete(get_optimized_chromadb(config))
                log.info("✅ Optimized ChromaDB manager initialized")
                _optimized_manager_initializing = False
                return _optimized_manager
        except RuntimeError:
            # No event loop available, will initialize on first async call
            pass
        except Exception as e:
            log.debug(f"Could not initialize optimized manager synchronously: {e}")
            _use_optimized = False
            _optimized_manager_initializing = False
    except ImportError:
        log.debug("Optimized ChromaDB manager not available")
        _use_optimized = False
    
    return None

# DEBUG: Log ChromaDB connection settings
log.debug("\n" + "="*60)
log.debug("🔍 CHROMADB CONNECTION DEBUG")
log.debug("="*60)
log.debug(f"API Key: {'*' * (len(settings.CHROMA_API_KEY) - 4) + settings.CHROMA_API_KEY[-4:] if len(settings.CHROMA_API_KEY) > 4 else '***'}")
log.debug(f"Tenant: {settings.CHROMA_TENANT}")
log.debug(f"Database: {settings.CHROMA_DATABASE}")
log.debug(f"Optimized Mode: {_use_optimized}")
log.debug("="*60 + "\n")

log.info("🔍 Initializing ChromaDB Cloud Client...")
log.info(f"   Tenant: {settings.CHROMA_TENANT}")
log.info(f"   Database: {settings.CHROMA_DATABASE}")
log.info(f"   API Key: {'*' * (len(settings.CHROMA_API_KEY) - 4) + settings.CHROMA_API_KEY[-4:] if len(settings.CHROMA_API_KEY) > 4 else '***'}")
log.info(f"   Optimized Mode: {_use_optimized}")

try:
    client = chromadb.CloudClient(
        api_key = settings.CHROMA_API_KEY,
        tenant = settings.CHROMA_TENANT,
        database = settings.CHROMA_DATABASE
    )
    log.info("✅ ChromaDB Cloud Client initialized successfully")
    
    # DEBUG: Try to list collections to verify connection
    # Suppress progress bars during collection listing
    original_tqdm = os.environ.get('TQDM_DISABLE', '0')
    os.environ['TQDM_DISABLE'] = '1'
    try:
        import contextlib
        from io import StringIO
        # Redirect stdout/stderr to suppress progress bars
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            collections = client.list_collections()
        log.info(f"📋 Found {len(collections)} collections in database '{settings.CHROMA_DATABASE}' (tenant: '{settings.CHROMA_TENANT}'):")
        for col in collections:
            # Suppress progress bars during count operations
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                count = col.count()
            log.info(f"   - {col.name} (count: {count})")
    except Exception as e:
        log.warning(f"⚠️ Could not list collections: {e}")
    finally:
        os.environ['TQDM_DISABLE'] = original_tqdm
        
except Exception as e:
    log.error(f"❌ Failed to initialize ChromaDB Cloud Client: {e}")
    import traceback
    log.error(f"Traceback: {traceback.format_exc()}")
    raise

# Use a free local embedding model (lazy initialization to avoid import errors)
_embedding_fn: Optional[SentenceTransformerEmbeddingFunction] = None
_embedding_lock = None  # Will be initialized as threading.Lock
_embedding_semaphore = None  # Will be initialized as asyncio.Semaphore

def _get_embedding_fn():
    """Lazy initialization of embedding function to avoid import-time errors"""
    global _embedding_fn, _embedding_lock
    if _embedding_fn is None:
        import threading
        if _embedding_lock is None:
            _embedding_lock = threading.Lock()
        
        with _embedding_lock:  # Double-check locking pattern
            if _embedding_fn is None:
                try:
                    # Disable progress bars to reduce noise in logs and improve performance
                    import os
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer warnings
                    os.environ['TQDM_DISABLE'] = '1'  # Ensure tqdm is disabled
                    
                    # Create embedding function with progress bars disabled
                    # Set environment variables before creating the function
                    os.environ['TQDM_DISABLE'] = '1'
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                    
                    # Temporarily redirect stdout/stderr to suppress any progress output
                    import contextlib
                    from io import StringIO
                    
                    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                        # Allow overriding the embedding model via ENV; default to 384-dim model for collection compatibility
                        # NOTE: Existing collections were built with 384-d embeddings. To use a 768-d model,
                        # you must recreate collections to avoid dimension mismatch errors.
                        model_name = os.getenv("CHROMA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
                        _embedding_fn = SentenceTransformerEmbeddingFunction(
                            model_name=model_name,  # Use stronger model than MiniLM
                            device="cpu"  # Explicitly set device
                        )
                    
                    # Disable progress bars in the underlying SentenceTransformer model
                    if hasattr(_embedding_fn, 'model'):
                        model = _embedding_fn.model
                        # Disable progress bar setting
                        if hasattr(model, 'set_show_progress_bar'):
                            model.set_show_progress_bar(False)
                        
                        # Monkey-patch the encode method to always disable progress bars
                        original_encode = model.encode
                        def encode_no_progress(sentences, *args, **kwargs):
                            # Force disable progress bar in kwargs
                            kwargs['show_progress_bar'] = False
                            # Suppress environment during encoding
                            os.environ['TQDM_DISABLE'] = '1'
                            # Ensure convert_to_numpy is set if not provided
                            if 'convert_to_numpy' not in kwargs:
                                kwargs['convert_to_numpy'] = True
                            # Redirect stdout/stderr during encoding
                            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                                return original_encode(sentences, *args, **kwargs)
                        model.encode = encode_no_progress
                    
                    log.debug("✅ Embedding function initialized (progress bars disabled)")
                except Exception as e:
                    log.error(f"❌ Failed to initialize embedding function: {e}")
                    # Fallback to default embedding function
                    try:
                        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                        _embedding_fn = DefaultEmbeddingFunction()
                        log.warning("⚠️ Using default embedding function as fallback")
                    except Exception as e2:
                        log.error(f"❌ Failed to initialize default embedding function: {e2}")
                        raise
    return _embedding_fn

def _get_embedding_semaphore():
    """Get or create semaphore to limit concurrent embedding operations"""
    global _embedding_semaphore
    if _embedding_semaphore is None:
        import asyncio
        # Limit to 10 concurrent embedding operations to prevent one request from monopolizing
        # This ensures fair access between job_matcher and JD analysis
        _embedding_semaphore = asyncio.Semaphore(10)
    return _embedding_semaphore

# Create a lazy wrapper class that implements the embedding function interface
class _LazyEmbeddingFunction:
    """
    Lazy wrapper for embedding function that initializes on first use.
    
    NOTE: SentenceTransformer models are thread-safe for inference, so multiple
    concurrent calls are safe. The embedding function is called from thread pools
    (via run_cpu_intensive), ensuring the event loop stays responsive.
    """
    def __call__(self, input):
        """
        Call the underlying embedding function (synchronous, blocking).
        
        This is called from thread pools (via run_cpu_intensive), so it's already
        non-blocking to the event loop. SentenceTransformer is thread-safe, so
        multiple concurrent calls are safe without additional locking.
        """
        return _get_embedding_fn()(input)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying embedding function"""
        return getattr(_get_embedding_fn(), name)
    
    def __repr__(self):
        """String representation"""
        if _embedding_fn is None:
            return "<LazyEmbeddingFunction (not yet initialized)>"
        return repr(_get_embedding_fn())

# Expose embedding_fn that initializes lazily (Section 8 Issue 2: optional embedding cache)
_raw_embedding_fn = _LazyEmbeddingFunction()
try:
    from core.embedding_cache import wrap_embedding_fn
    embedding_fn = wrap_embedding_fn(_raw_embedding_fn)
except Exception as e:
    log.debug("Embedding cache not used: %s", e)
    embedding_fn = _raw_embedding_fn

# Create / get collections (using optimized manager if available, otherwise direct)
def _get_collection(name: str):
    """Get collection using optimized manager if available, otherwise direct"""
    # Suppress progress bars during collection operations
    import contextlib
    from io import StringIO
    os.environ['TQDM_DISABLE'] = '1'
    
    # Ensure embedding function is initialized before collection access
    # This prevents ChromaDB from trying to rebuild it from config (which causes meta tensor errors)
    embedding_fn = None
    try:
        embedding_fn = _get_embedding_fn()
    except Exception as e:
        log.warning(f"⚠️ Failed to pre-initialize embedding function: {e}")
    
    manager = _get_optimized_manager()
    if manager:
        try:
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                return manager.get_collection(name)
        except Exception as e:
            log.debug(f"Optimized collection get failed, using direct: {e}")
    
    # QA: courses_knowledge_base was created with default embedder; use persisted (no embedding_function)
    app_env = (os.getenv("APP_ENV") or getattr(settings, "APP_ENV", "") or "").lower()
    if app_env == "qa" and name == "courses_knowledge_base":
        try:
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                return client.get_collection(name=name)
        except Exception as e:
            log.debug(f"QA courses_knowledge_base get_collection failed, falling through: {e}")
    
    # Use get_or_create_collection with explicit embedding function
    # This prevents ChromaDB from trying to rebuild the embedding function from stored config
    # which causes the "meta tensor" error
    try:
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            if embedding_fn:
                # Explicitly provide embedding function to prevent ChromaDB from rebuilding from config
                return client.get_or_create_collection(
                    name=name,
                    embedding_function=embedding_fn
                )
            else:
                # Fallback: try to get existing collection
                return client.get_collection(name=name)
    except (ValueError, NotImplementedError) as e:
        err_msg = str(e).lower()
        # Handle embedding function conflict: collection was created with different embedder (e.g. default)
        if "embedding function conflict" in err_msg or "already exists in the collection configuration" in err_msg:
            log.warning(f"⚠️ Collection '{name}' has different embedding function than current; using persisted embedder.")
            try:
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    return client.get_collection(name=name)
            except Exception as get_e:
                log.error(f"❌ Failed to get collection '{name}' with persisted embedder: {get_e}")
                raise
        # Handle meta tensor error - ChromaDB is trying to rebuild embedding function from config
        if "meta tensor" in err_msg or "cannot copy out of meta tensor" in err_msg or "could not build embedding function" in err_msg:
            log.warning(f"⚠️ Collection '{name}' has embedding function config issue. Attempting workaround...")
            # Try to get collection without embedding function, then manually set it
            # This is a workaround for the meta tensor issue
            try:
                # Force recreate with fresh embedding function
                # Note: This will lose existing data, but it's better than crashing
                try:
                    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                        client.delete_collection(name=name)
                    log.info(f"✅ Deleted problematic collection '{name}'")
                except Exception as del_e:
                    log.debug(f"Collection '{name}' may not exist or already deleted: {del_e}")
                
                # Create new collection with fresh embedding function
                if embedding_fn:
                    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                        return client.create_collection(
                            name=name,
                            embedding_function=embedding_fn
                        )
                else:
                    # Last resort: use default embedding function
                    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                        return client.create_collection(
                            name=name,
                            embedding_function=DefaultEmbeddingFunction()
                        )
            except Exception as recreate_e:
                log.error(f"❌ Failed to recreate collection '{name}': {recreate_e}")
                raise
        else:
            raise
    except Exception as e:
        # Collection doesn't exist, create it with embedding function
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            if embedding_fn:
                return client.create_collection(
                    name=name,
                    embedding_function=embedding_fn
                )
            else:
                # Fallback to default embedding function
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                return client.create_collection(
                    name=name,
                    embedding_function=DefaultEmbeddingFunction()
                )

# Initialize collections (backward compatible - same names)
collection = _get_collection("resume")
chat_sessions_collection = _get_collection("chat_sessions")

# Create / get job_descriptions collection
try:
    job_descriptions_collection = _get_collection("job_descriptions")
    # Suppress progress bars during count
    import contextlib
    from io import StringIO
    os.environ['TQDM_DISABLE'] = '1'
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        count = job_descriptions_collection.count()
    log.info(f"✅ job_descriptions collection ready (count: {count})")
except Exception as e:
    log.error(f"❌ Failed to get/create job_descriptions collection: {e}")
    raise

# NEW: session_chunks and session_history collections
session_chunks_collection = _get_collection("session_chunks")
session_history_collection = _get_collection("session_history")

# Dedicated collection for user assessments and completion status (separate from chat_sessions)
assessments_collection = _get_collection("user_assessments")

# Chroma document size limit (16KB per record; use 15.5KB to be safe)
CHROMA_DOCUMENT_MAX_BYTES = 15872


def _chunk_list_for_chroma(items: list, max_bytes: int = CHROMA_DOCUMENT_MAX_BYTES):
    """
    Yield (chunk_index, list_chunk) so each chunk's JSON size is <= max_bytes.
    Preserves full content; no truncation. Chunk index is 0, 1, 2, ... for doc ids.
    """
    if not items:
        yield (0, [])
        return
    chunk = []
    idx = 0
    for item in items:
        chunk.append(item)
        doc = json.dumps(chunk, default=str, ensure_ascii=False)
        if len(doc.encode("utf-8")) > max_bytes:
            chunk.pop()
            if chunk:
                yield (idx, chunk)
                idx += 1
            chunk = [item]
    if chunk:
        yield (idx, chunk)


def _truncate_strings_in_place(obj, max_chars: int):
    """Recursively truncate string values in dict/list to max_chars. Modifies in place."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str):
                if len(v) > max_chars:
                    obj[k] = v[:max_chars] + "…" if max_chars > 1 else v[:1]
            else:
                _truncate_strings_in_place(v, max_chars)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                if len(v) > max_chars:
                    obj[i] = v[:max_chars] + "…" if max_chars > 1 else v[:1]
            else:
                _truncate_strings_in_place(v, max_chars)


def _payload_to_doc_within_limit(payload: dict, max_bytes: int = CHROMA_DOCUMENT_MAX_BYTES, drop_keys: list = None) -> str:
    """
    Return JSON string of payload that fits within max_bytes (UTF-8).
    Drops optional keys and truncates strings as needed.
    """
    import copy
    drop_keys = list(drop_keys or [])
    base = copy.deepcopy(payload)
    for k in drop_keys:
        base.pop(k, None)
    for max_chars in (4000, 2000, 1000, 500, 200, 100):
        p = copy.deepcopy(base)
        _truncate_strings_in_place(p, max_chars)
        doc_str = json.dumps(p, default=str, ensure_ascii=False)
        if len(doc_str.encode("utf-8")) <= max_bytes:
            return doc_str
    doc_str = json.dumps(base, default=str, ensure_ascii=False)
    if len(doc_str.encode("utf-8")) <= max_bytes:
        return doc_str
    # Last resort: truncate the serialized string (may produce invalid JSON; avoid if possible)
    b = doc_str.encode("utf-8")
    if len(b) > max_bytes:
        doc_str = b[:max_bytes].decode("utf-8", errors="ignore")
        if not doc_str.endswith("}"):
            doc_str = doc_str.rsplit(",", 1)[0] + "}"
    return doc_str


# Cache for compare-candidate-job results keyed by (uid, job_id)
candidate_job_rankings_collection = _get_collection("candidate_job_rankings")


def get_candidate_job_ranking(uid: str, job_id: str, resume_hash: str = None):
    """
    Return cached candidate-job match result for (uid, job_id), or None if not found.
    Reassembles similar_jobs from main doc or from chunk docs if stored in chunks.
    If resume_hash is provided, returns None when stored resume_hash does not match (resume changed).
    """
    if not uid or not job_id:
        return None
    try:
        base_id = f"{uid}_{job_id}"
        out = candidate_job_rankings_collection.get(
            ids=[base_id],
            include=["documents", "metadatas"],
        )
        if not out or not out.get("ids") or not out.get("documents"):
            return None
        doc = (out["documents"] or [None])[0]
        metadatas = out.get("metadatas") or []
        meta = metadatas[0] if metadatas else {}
        if not doc:
            return None
        # Resume hash check: if resume changed, treat as cache miss
        if resume_hash is not None:
            stored_hash = meta.get("resume_hash") if isinstance(meta, dict) else None
            if stored_hash != resume_hash:
                return None
        data = json.loads(doc)
        if not isinstance(data, dict) or "match_score" not in data:
            return None
        chunk_ids = data.pop("similar_jobs_chunk_ids", None)
        if chunk_ids:
            out2 = candidate_job_rankings_collection.get(ids=chunk_ids, include=["documents"])
            if out2 and out2.get("documents"):
                similar = []
                for d in (out2["documents"] or []):
                    if d:
                        similar.extend(json.loads(d))
                data["similar_jobs"] = similar
            else:
                data["similar_jobs"] = []
        else:
            similar_id = f"{base_id}_similar_jobs"
            out2 = candidate_job_rankings_collection.get(ids=[similar_id], include=["documents"])
            if out2 and out2.get("ids") and out2["ids"] and out2.get("documents") and out2["documents"][0]:
                data["similar_jobs"] = json.loads(out2["documents"][0])
            elif "similar_jobs" not in data:
                data["similar_jobs"] = []
        return data
    except Exception as e:
        log.warning("get_candidate_job_ranking failed for %s/%s: %s", uid, job_id, e)
        return None


def upsert_candidate_job_ranking(uid: str, job_id: str, result: dict, resume_hash: str | None = None) -> bool:
    """
    Store full candidate-job match result; chunk similar_jobs across docs if needed so no info is lost.
    resume_hash: optional hash of resume content; used to invalidate cache when resume changes.
    """
    if not uid or not job_id or not result or not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    try:
        base_id = f"{uid}_{job_id}"
        now = datetime.utcnow().isoformat() + "Z"
        meta_dict = {"uid": uid, "job_id": str(job_id), "stored_at": now}
        if resume_hash:
            meta_dict["resume_hash"] = resume_hash
        meta = normalize_metadata(meta_dict)
        main_payload = {k: v for k, v in result.items() if k != "error" and k != "similar_jobs"}
        similar_jobs = result.get("similar_jobs") or []
        if similar_jobs:
            similar_str = json.dumps(similar_jobs, default=str, ensure_ascii=False)
            if len(similar_str.encode("utf-8")) <= CHROMA_DOCUMENT_MAX_BYTES:
                candidate_job_rankings_collection.upsert(
                    ids=[f"{base_id}_similar_jobs"],
                    documents=[similar_str],
                    metadatas=[meta],
                )
            else:
                chunk_ids = []
                for idx, chunk in _chunk_list_for_chroma(similar_jobs):
                    cid = f"{base_id}_similar_{idx}"
                    chunk_ids.append(cid)
                    candidate_job_rankings_collection.upsert(
                        ids=[cid],
                        documents=[json.dumps(chunk, default=str, ensure_ascii=False)],
                        metadatas=[meta],
                    )
                main_payload["similar_jobs_chunk_ids"] = chunk_ids
        doc_str = _payload_to_doc_within_limit(main_payload, drop_keys=[])
        candidate_job_rankings_collection.upsert(
            ids=[base_id],
            documents=[doc_str],
            metadatas=[meta],
        )
        log.info("Stored candidate_job_ranking in Chroma for uid=%s job_id=%s", uid, job_id)
        return True
    except Exception as e:
        log.warning("upsert_candidate_job_ranking failed for %s/%s: %s", uid, job_id, e)
        return False


# Cache for job_matcher results (one ranking per uid)
job_matcher_rankings_collection = _get_collection("job_matcher_rankings")

# Keys from job_matcher_agent return value to store (exclude state like structured_resume)
_JOB_MATCHER_CACHE_KEYS = (
    "top_matches", "matched_jobs", "total_matches_found", "job_matcher_status",
    "confidence_score", "processing_time_seconds", "total_jobs_matched",
    "semantic_filter_threshold", "processing_method", "batches_processed", "message",
)


def get_job_matcher_ranking(uid: str):
    """
    Return cached job_matcher result for uid, or None if not found.
    Reassembles matched_jobs (and top_matches) from chunk docs when stored in chunks.
    """
    if not uid:
        return None
    try:
        base_id = f"job_matcher_{uid}"
        out = job_matcher_rankings_collection.get(
            ids=[base_id],
            include=["documents", "metadatas"],
        )
        if not out or not out.get("ids") or not out.get("documents"):
            return None
        doc = (out["documents"] or [None])[0]
        if not doc:
            return None
        data = json.loads(doc)
        if not isinstance(data, dict):
            return None
        chunk_ids = data.pop("matched_jobs_chunk_ids", None)
        if chunk_ids:
            out2 = job_matcher_rankings_collection.get(ids=chunk_ids, include=["documents"])
            if out2 and out2.get("documents"):
                matched = []
                for d in (out2["documents"] or []):
                    if d:
                        matched.extend(json.loads(d))
                data["matched_jobs"] = matched
                data["top_matches"] = matched
                data["total_matches_found"] = len(matched)
                data["total_jobs_matched"] = len(matched)
        if "matched_jobs" not in data:
            return None
        return data
    except Exception as e:
        log.warning("get_job_matcher_ranking failed for %s: %s", uid, e)
        return None


def upsert_job_matcher_ranking(uid: str, result: dict, input_hash: Optional[str] = None) -> bool:
    """
    Store full job_matcher result; chunk matched_jobs across docs if needed so no info is lost.
    input_hash: Hash of the resume used for this run; used to invalidate cache when input changes.
    """
    if not uid or not result or not isinstance(result, dict):
        return False
    if result.get("error") or result.get("status") == "error":
        return False
    try:
        base_id = f"job_matcher_{uid}"
        now = datetime.utcnow().isoformat() + "Z"
        meta = normalize_metadata({"uid": uid, "stored_at": now})
        payload = {k: result[k] for k in _JOB_MATCHER_CACHE_KEYS if k in result}
        if not payload:
            return False
        if input_hash is not None:
            payload["_input_hash"] = input_hash
        matched_jobs = payload.get("matched_jobs") or payload.get("top_matches") or []
        main_payload = {k: v for k, v in payload.items() if k not in ("matched_jobs", "top_matches")}
        main_payload["total_matches_found"] = len(matched_jobs)
        main_payload["total_jobs_matched"] = len(matched_jobs)
        if not matched_jobs:
            main_payload["matched_jobs"] = []
            main_payload["top_matches"] = []
            doc_str = json.dumps(main_payload, default=str, ensure_ascii=False)
            if len(doc_str.encode("utf-8")) > CHROMA_DOCUMENT_MAX_BYTES:
                doc_str = _payload_to_doc_within_limit(main_payload)
            job_matcher_rankings_collection.upsert(ids=[base_id], documents=[doc_str], metadatas=[meta])
        else:
            full_doc = json.dumps(payload, default=str, ensure_ascii=False)
            if len(full_doc.encode("utf-8")) <= CHROMA_DOCUMENT_MAX_BYTES:
                job_matcher_rankings_collection.upsert(ids=[base_id], documents=[full_doc], metadatas=[meta])
            else:
                chunk_ids = []
                for idx, chunk in _chunk_list_for_chroma(matched_jobs):
                    cid = f"{base_id}_jobs_{idx}"
                    chunk_ids.append(cid)
                    job_matcher_rankings_collection.upsert(
                        ids=[cid],
                        documents=[json.dumps(chunk, default=str, ensure_ascii=False)],
                        metadatas=[meta],
                    )
                main_payload["matched_jobs_chunk_ids"] = chunk_ids
                doc_str = _payload_to_doc_within_limit(main_payload)
                job_matcher_rankings_collection.upsert(ids=[base_id], documents=[doc_str], metadatas=[meta])
        log.info("Stored job_matcher_ranking in Chroma for uid=%s", uid)
        return True
    except Exception as e:
        log.warning("upsert_job_matcher_ranking failed for %s: %s", uid, e)
        return False


# Byte-aware utilities for Chroma limits
def _split_text_into_byte_chunks(text: str, max_bytes: int = 15000, overlap_chars: int = 0) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        low = start + 1
        high = n
        best = low
        while low <= high:
            mid = (low + high) // 2
            b = text[start:mid].encode("utf-8")
            if len(b) <= max_bytes:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        chunk = text[start:best]
        if not chunk:
            # Force minimal forward progress
            chunk = text[start:start+1]
            best = start + 1
        chunks.append(chunk)
        if best >= n:
            break
        start = max(0, best - max(0, overlap_chars))
    return chunks

def _trim_metadata_value(value, max_bytes: int = 4000):
    if not isinstance(value, str):
        return value
    b = value.encode("utf-8")
    if len(b) <= max_bytes:
        return value
    low, high, best = 0, len(value), 0
    while low <= high:
        mid = (low + high) // 2
        bm = value[:mid].encode("utf-8")
        if len(bm) <= max_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return value[:best]

def to_primitive_metadata_value(value):
    """
    Convert any value to a ChromaDB-compatible primitive type (str, int, float, bool, or None).
    ChromaDB metadata only accepts primitive types, not lists or dicts.
    
    This is a public utility function for normalizing metadata values across the codebase.
    """
    # Already a primitive type - return as-is
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    
    # Handle lists: take first element if single, join if multiple
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        elif len(value) == 1:
            # Recursively convert the single element
            return to_primitive_metadata_value(value[0])
        else:
            # Join multiple elements with comma
            return ", ".join(str(to_primitive_metadata_value(item)) for item in value)
    
    # Handle dicts: convert to JSON string
    if isinstance(value, dict):
        try:
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return str(value)
    
    # For any other type, convert to string
    return str(value)

def normalize_metadata(meta: dict | None) -> dict:
    """
    Normalize metadata dictionary to ensure all values are ChromaDB-compatible primitives.
    Trims string values to fit ChromaDB's metadata size limits.
    - Keys in _METADATA_NO_TRIM_KEYS (e.g. structured_resume_json): trim to 8000 bytes
      (ChromaDB Cloud limit is 8182 per value).
    - Other keys: trim to 4000 bytes.

    This is a public utility function for normalizing metadata dictionaries across the codebase.
    Use this before passing metadata to any ChromaDB operation (upsert, add, update).
    """
    meta = dict(meta or {})
    for k, v in list(meta.items()):
        # First convert to primitive type
        v = to_primitive_metadata_value(v)
        if isinstance(v, str):
            max_bytes = _CHROMADB_METADATA_VALUE_MAX_BYTES if k in _METADATA_NO_TRIM_KEYS else 4000
            meta[k] = _trim_metadata_value(v, max_bytes=max_bytes)
        else:
            meta[k] = v
    return meta

# Keep _normalize_metadata as an alias for backward compatibility
_normalize_metadata = normalize_metadata

# =============================================================================
# TENANT ISOLATION HELPERS (Backward Compatible)
# =============================================================================
# These helpers support gradual migration to tenant isolation:
# - Writes ALWAYS include tenant_id in metadata (if provided)
# - Reads only filter by tenant_id if explicitly provided
# - Missing tenant_id logs a warning but doesn't break existing code

def _validate_tenant_id(tenant_id: str, operation: str) -> bool:
    """Validate tenant_id format. Returns True if valid or None (backward compatible)."""
    if tenant_id is None:
        log.warning(f"Tenant isolation: {operation} called without tenant_id (backward compatible mode)")
        return True  # Allow for backward compatibility
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        log.error(f"Tenant isolation violation: invalid tenant_id format for {operation}")
        return False
    return True

def _add_tenant_to_metadata(metadata: dict, tenant_id: str = None) -> dict:
    """Add tenant_id to metadata if provided. Always returns a dict."""
    meta = dict(metadata or {})
    if tenant_id and isinstance(tenant_id, str) and tenant_id.strip():
        meta["tenant_id"] = tenant_id
    return meta

def _build_tenant_where_clause(tenant_id: str = None, additional_filters: dict = None) -> dict:
    """Build a ChromaDB where clause that optionally includes tenant_id filter.
    
    Args:
        tenant_id: Optional tenant identifier. If None, no tenant filter is applied.
        additional_filters: Optional additional filter conditions.
        
    Returns:
        Combined where clause, or None if no filters.
    """
    if not tenant_id and not additional_filters:
        return None
    
    if tenant_id and additional_filters:
        return {"$and": [{"tenant_id": tenant_id}, additional_filters]}
    elif tenant_id:
        return {"tenant_id": tenant_id}
    else:
        return additional_filters

# Lightweight validation and hashing

def _validate_session_data(session_data: dict) -> tuple[bool, list[str]]:
    missing = []
    for req in ["uid", "timestamp", "status"]:
        if not session_data.get(req):
            missing.append(req)
    return (len(missing) == 0, missing)


def _calculate_data_hash(data: dict) -> str:
    import json, hashlib
    try:
        canonical = json.dumps(data, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    except Exception:
        canonical = str(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_json_safe(value):
    """
    Recursively coerce values into a JSON-serializable form.
    
    This is a safety net for unexpected objects (e.g. coroutines, custom classes)
    that might accidentally get attached to session data/state. Instead of
    failing hard when calling json.dumps, we:
    - preserve primitives, lists, and dicts structurally
    - convert datetime-like objects to ISO strings
    - fall back to str(value) for anything else
    """
    # Fast-path for common JSON types
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Containers
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]

    # Datetime-like objects
    try:
        import datetime as _dt  # Local import to avoid global dependency at module load
        if isinstance(value, (_dt.datetime, _dt.date)):
            return value.isoformat()
    except Exception:
        # If datetime import/type-checking fails, fall through to generic str()
        pass

    # Coroutines / async generators / other callables: stringify with type info
    try:
        import inspect as _inspect
        if _inspect.iscoroutine(value) or _inspect.isawaitable(value):
            return f"<coroutine {type(value).__name__}>"
        if _inspect.isasyncgen(value):
            return f"<async-generator {type(value).__name__}>"
    except Exception:
        # If inspect isn't available for some reason, fall through
        pass

    # Generic fallback
    return str(value)

# ----------------- RESUME FUNCTIONS (unchanged) -----------------

# def insert_resume(resume_id: str, resume_text: str, metadata: dict = None):
#     collection.add(
#         ids=[resume_id],
#         documents=[resume_text],
#         metadatas=[metadata] if metadata else [{}]
#     )
#     (f" Resume {resume_id} inserted.")

# Minimal document stored in resume collection; full resume can also be in metadata.
# Embedding is computed from full structured_resume so vector search is unchanged.
RESUME_DOC_PLACEHOLDER = ""

# Keys that get a higher trim limit (8000) instead of default 4000 - full resume for ranking.
# ChromaDB Cloud limit is 8182 bytes per metadata value; we cap at 8000 to be safe.
_CHROMADB_METADATA_VALUE_MAX_BYTES = 8000
_METADATA_NO_TRIM_KEYS = {"structured_resume_json", "structured_resume", "structuredResume", "resume_json"}


def _resume_to_metadata_payload(resume_data: dict) -> str:
    """Build a compact JSON string of the resume data for metadata storage.

    To keep Chroma metadata within quota and avoid storing unnecessary,
    large summaries, we:
    - Drop non-essential summary fields (e.g. user_interests_summary).
    - Serialize the remaining structured resume as compact JSON.
    """
    if not resume_data or not isinstance(resume_data, dict):
        return "{}"
    try:
        # Work on a shallow copy so we can safely drop noisy/large fields.
        data = dict(resume_data)

        # Explicitly exclude large, derived summaries that are not needed for ranking.
        for noisy_key in [
            "user_interests_summary",
            "userInterestsSummary",
            "user_interest_summary",
            "alternate_career_paths_summary",
            "alternateCareerPathsSummary",
        ]:
            data.pop(noisy_key, None)

        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def insert_resume(resume_id: str, resume_data: dict, metadata: dict = None, tenant_id: str = None):
    """
    Insert or update resume in the resume collection by storing only the embedding
    (and a minimal placeholder document). Full resume is stored in chat_sessions
    via upsert_resume_doc. This avoids ChromaDB document size quota limits.

    The embedding is computed from the full structured_resume JSON so vector
    search semantics are unchanged; the ranker resolves full resume from
    chat_sessions by uid after querying Chroma.

    Args:
        resume_id: Unique identifier (typically UID)
        resume_data: Dictionary containing structured resume data
        metadata: Optional metadata
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    if not _validate_tenant_id(tenant_id, f"insert_resume({resume_id})"):
        return False
    log.debug(f"insert_resume called with resume_id={resume_id}")
    log.debug(f"resume_data type={type(resume_data)}")
    log.debug(f"resume_data keys={list(resume_data.keys()) if isinstance(resume_data, dict) else 'Not a dict'}")

    try:
        # Build full resume text only for computing the embedding (not stored in Chroma)
        log.debug("Converting resume_data to compact JSON for embedding...")
        resume_text = json.dumps(resume_data, separators=(",", ":"), ensure_ascii=False)
        log.debug(f"Resume text length for embedding: {len(resume_text.encode('utf-8'))} bytes")
        
        # Issue 3.3: Calculate content hash to detect unchanged resumes
        content_hash = _calculate_data_hash(resume_data)
        
        # Check if existing embedding has same content (skip re-embedding if unchanged)
        try:
            existing = collection.get(ids=[resume_id], include=["metadatas"])
            if existing.get("ids") and existing.get("metadatas"):
                existing_hash = (existing["metadatas"][0] or {}).get("content_hash")
                if existing_hash == content_hash:
                    log.info(f"✅ Resume {resume_id} unchanged (hash match), skipping re-embedding")
                    return True
        except Exception as e:
            log.debug(f"Could not check existing resume hash: {e}")

        # Compute embedding from full resume so vector search is unchanged
        embedding_fn = _get_embedding_fn()
        embeddings_list = embedding_fn([resume_text])
        if not embeddings_list or len(embeddings_list[0]) == 0:
            raise ValueError("Embedding function returned empty result")
        embedding = embeddings_list[0]
        
        # Issue 3.1: Get embedding model name for version tracking
        embedding_model = os.getenv("CHROMA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        # Prepare metadata with resume details (no large document stored)
        from datetime import datetime
        current_timestamp = datetime.utcnow().isoformat()
        existing_metadata = metadata or {}
        if "created_at" not in existing_metadata:
            existing_metadata["created_at"] = current_timestamp
        existing_metadata["updated_at"] = current_timestamp

        # Groq parser uses "name"; resume_assembler uses "Name" - support both
        name_value = (
            resume_data.get("Name")
            or resume_data.get("name")
            or resume_data.get("fullName")
            or resume_data.get("full_name")
            or "N/A"
        )
        if isinstance(name_value, list) and len(name_value) > 0:
            name_value = name_value[0]
        elif not isinstance(name_value, str):
            name_value = str(name_value) if name_value else "N/A"

        # Groq uses contact_details; resume_assembler uses ContactDetails
        contact = resume_data.get("ContactDetails") or resume_data.get("contact_details") or {}
        location_value = (
            resume_data.get("Location")
            or (contact.get("Location") if isinstance(contact, dict) else None)
            or (contact.get("location") if isinstance(contact, dict) else None)
            or "N/A"
        )

        # Extract candidate_domains if available (optional field from newer resume parsing)
        candidate_domains_raw = (
            resume_data.get("candidate_domains")
            or resume_data.get("candidateDomains")
            or []
        )
        candidate_domains_str = ",".join(candidate_domains_raw[:3]) if candidate_domains_raw else ""

        # Store resume as JSON in metadata so ranker/get_resume can resolve from collection (no chat_sessions needed)
        structured_resume_json = _resume_to_metadata_payload(resume_data)

        resume_metadata = {
            "uid": resume_id,
            "name": name_value,
            "location": location_value,
            "skills_count": len(resume_data.get("Skills", []) or resume_data.get("skills", [])),
            "education_count": len(resume_data.get("Education", []) or resume_data.get("education", [])),
            "total_experience_years": resume_data.get("total_experience_years", "0 months"),
            "candidate_domains": candidate_domains_str,
            "doc_size": 0,
            "embedding_model": embedding_model,
            "content_hash": content_hash,
            "structured_resume_json": structured_resume_json,
            **existing_metadata
        }
        # Tenant isolation: add tenant_id to metadata if provided
        resume_metadata = _add_tenant_to_metadata(resume_metadata, tenant_id)
        log.debug(f"Metadata prepared: {resume_metadata}")

        resume_metadata = normalize_metadata(resume_metadata)
        log.debug(f"Metadata normalized: {resume_metadata}")

        # Store only embedding + minimal document in resume collection
        manager = _get_optimized_manager()
        if manager:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(
                        manager.batch_upsert(
                            collection_name="resume",
                            ids=[resume_id],
                            documents=[RESUME_DOC_PLACEHOLDER],
                            embeddings=[embedding],
                            metadatas=[resume_metadata],
                            flush_immediately=True,
                        )
                    )
                    log.info(f"✅ Resume {resume_id} upserted successfully (embedding only, optimized).")
                    return True
            except (RuntimeError, Exception) as e:
                log.debug(f"Optimized upsert not available, using direct: {e}")

        log.debug("Calling collection.upsert (embedding only)...")
        collection.upsert(
            ids=[resume_id],
            documents=[RESUME_DOC_PLACEHOLDER],
            embeddings=[embedding],
            metadatas=[resume_metadata],
        )
        log.info(f"✅ Resume {resume_id} upserted successfully (embedding only).")
        return True
    except Exception as e:
        import traceback
        log.error(f"❌ Error inserting resume {resume_id}: {e}")
        log.error(f"❌ Error type: {type(e).__name__}")
        log.error(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def _sanitize_json_string_values(s: str) -> str:
    """Fix unescaped newlines/tabs/control chars inside JSON string values."""
    out = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            out.append(c)
            if i + 1 < len(s):
                i += 1
                out.append(s[i])
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if in_string:
            if c == '\n':
                out.append('\\n')
            elif c == '\r':
                out.append('\\r')
            elif c == '\t':
                out.append('\\t')
            elif ord(c) < 0x20:
                out.append(f'\\u{ord(c):04x}')
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def _repair_truncated_json(s: str) -> dict | None:
    """
    Recover a dict from truncated JSON by closing open brackets/braces.
    Walks the string tracking nesting depth and string context,
    then appends the necessary closing tokens.
    """
    if not s or not s.strip().startswith("{"):
        return None
    s = s.strip()
    in_str = False
    stack = []  # track open { and [
    i = 0
    last_good = 0  # position after last complete key-value or array element
    while i < len(s):
        c = s[i]
        if c == '\\' and in_str:
            i += 2
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c in ('{', '['):
                stack.append(c)
            elif c == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    last_good = i + 1
            elif c == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    last_good = i + 1
            elif c == ',':
                last_good = i
        i += 1
    if not stack:
        return None  # Not truncated
    # Truncate to last safe position, then close open brackets
    # Find the last comma or complete value before truncation
    truncated = s[:last_good] if last_good > 1 else s
    # Remove trailing comma if present
    truncated = truncated.rstrip().rstrip(',').rstrip()
    # Close open brackets/braces in reverse order
    # Re-scan to get current stack after truncation
    stack2 = []
    in_str2 = False
    for j, c in enumerate(truncated):
        if c == '\\' and in_str2:
            continue
        if c == '"':
            in_str2 = not in_str2
        elif not in_str2:
            if c in ('{', '['):
                stack2.append(c)
            elif c == '}' and stack2 and stack2[-1] == '{':
                stack2.pop()
            elif c == ']' and stack2 and stack2[-1] == '[':
                stack2.pop()
    # Close remaining open brackets
    closing = ""
    for bracket in reversed(stack2):
        closing += "]" if bracket == "[" else "}"
    repaired = truncated + closing
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _try_parse_resume_json(raw: str) -> dict | None:
    """
    Parse resume JSON with repair for common Chroma metadata issues:
    truncated JSON (from 4000-byte metadata limit), unescaped newlines, trailing commas, single quotes.
    Returns dict or None.
    """
    if not raw or not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    # Stage 1: direct parse
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Stage 2: sanitize control chars inside string values
    s2 = _sanitize_json_string_values(s)
    try:
        parsed = json.loads(s2)
        if isinstance(parsed, dict):
            log.debug("Resume JSON parsed after sanitize (stage 2)")
            return parsed
    except json.JSONDecodeError:
        pass
    # Stage 3: fix trailing commas, line endings, Python literals
    import re
    s3 = s2.replace('\r\n', '\n').replace('\r', '\n')
    s3 = re.sub(r',(\s*[}\]])', r'\1', s3)
    s3 = re.sub(r'\bTrue\b', 'true', s3)
    s3 = re.sub(r'\bFalse\b', 'false', s3)
    s3 = re.sub(r'\bNone\b', 'null', s3)
    try:
        parsed = json.loads(s3)
        if isinstance(parsed, dict):
            log.debug("Resume JSON parsed after repair (stage 3)")
            return parsed
    except json.JSONDecodeError:
        pass
    # Stage 4: single-quote to double-quote
    s4 = re.sub(r"'([^']*)':", r'"\1":', s3)
    s4 = re.sub(r":\s*'([^']*)'", r': "\1"', s4)
    try:
        parsed = json.loads(s4)
        if isinstance(parsed, dict):
            log.debug("Resume JSON parsed after repair (stage 4)")
            return parsed
    except json.JSONDecodeError:
        pass
    # Stage 5: truncated JSON recovery — close open brackets/braces
    for attempt in (s3, s2, s):
        parsed = _repair_truncated_json(attempt)
        if parsed:
            log.debug(f"Resume JSON recovered from truncated metadata ({len(attempt)} chars, {len(parsed)} top-level keys)")
            return parsed
    log.debug("Resume JSON parse failed after all repair stages (including truncation recovery)")
    return None


def parse_resume_from_metadata(meta: dict | None) -> dict | None:
    """
    Parse structured resume from Chroma metadata (e.g. from resume collection).
    Metadata may contain 'structured_resume_json', 'structured_resume', 'structuredResume',
    or 'resume_json' (JSON string). Also accepts already-parsed dict.
    Uses JSON repair for truncated/malformed metadata (Chroma size limits, bulk imports).
    Returns dict or None if missing/invalid.
    """
    if not meta or not isinstance(meta, dict):
        return None
    # Try multiple keys (bulk imports may use different naming)
    raw = (
        meta.get("structured_resume_json")
        or meta.get("structured_resume")
        or meta.get("structuredResume")
        or meta.get("resume_json")
    )
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    return _try_parse_resume_json(raw)


def parse_resume_from_document(doc: str | None) -> dict | None:
    """
    Parse structured resume from Chroma document field (JSON string).
    Use when resume is stored as the document rather than in metadata.
    Returns dict or None if missing/invalid.
    """
    if not doc or not isinstance(doc, str) or not doc.strip():
        return None
    try:
        parsed = json.loads(doc)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def parse_resume_from_chroma_result(meta: dict | None, doc: str | None) -> dict | None:
    """
    Parse structured resume from Chroma query result.
    Retrieves from metadata of each uid first (structured_resume_json, structured_resume, etc.).
    Document used only as fallback when metadata parse fails.
    """
    parsed = parse_resume_from_metadata(meta)
    if parsed:
        return parsed
    return parse_resume_from_document(doc)


def reindex_resume_for_ranker(resume_id: str, resume_data: dict = None, tenant_id: str = None) -> bool:
    """
    Re-embed and upsert a candidate into the resume collection so the ranker can retrieve them.
    Use when candidates were bulk-imported with metadata only (no embeddings) or wrong embeddings.
    If resume_data is None, fetches from get_resume first.

    IMPORTANT: For consistency with Groq parsing, only the Groq parser's structured
    resume output (without proficiency scores) should be stored in metadata.
    This helper mirrors the behavior of `agents.groq_resume_parser._insert_resume_after_parse`.
    """
    if not resume_data:
        resume_data = get_resume(resume_id)
    if not resume_data or not isinstance(resume_data, dict):
        log.warning(f"reindex_resume_for_ranker: no resume data for {resume_id}")
        return False

    # Normalize to the same shape used by groq_resume_parser: prefer "structured_resume"
    base = resume_data.get("structured_resume") if isinstance(resume_data, dict) else None
    if isinstance(base, dict):
        resume_core = base
    else:
        resume_core = resume_data

    # Strip proficiency or extra fields from skills to keep only SkillName/name.
    try:
        from agents.groq_resume_parser import _strip_proficiency_from_skills

        clean_data = _strip_proficiency_from_skills(resume_core)
    except Exception:
        # Fallback: if import fails, use the raw core data but still proceed.
        clean_data = resume_core

    metadata = {"uid": resume_id}
    return insert_resume(resume_id, clean_data, metadata=metadata, tenant_id=tenant_id)


def _is_chroma_rate_limit_error(e: Exception) -> bool:
    """Check if exception is ChromaDB rate limit (429)."""
    err_str = str(e).lower()
    return "too many requests" in err_str or "429" in err_str or "rate limit" in err_str


def get_resume(resume_id: str):
    """
    Retrieve full resume data by resume_id (UID).
    Tries resume collection metadata first (structured_resume_json), then chat_sessions,
    then legacy document in resume collection.
    Retries with exponential backoff on ChromaDB rate limit (429).
    """
    if isinstance(resume_id, list):
        resume_id = resume_id[0]

    last_error = None
    for attempt in range(_CHROMA_RATE_LIMIT_RETRIES):
        try:
            # Primary: resume collection metadata (resumes stored as metadata)
            try:
                results = collection.get(ids=[resume_id], include=["metadatas"])
                if results.get("ids") and results["ids"] and results.get("metadatas"):
                    meta = (results["metadatas"][0]) or {}
                    parsed = parse_resume_from_metadata(meta)
                    if parsed and isinstance(parsed, dict):
                        return parsed
            except Exception as e:
                if _is_chroma_rate_limit_error(e):
                    raise
                log.debug(f"Resume metadata fetch for {resume_id}: {e}")

            # Fallback: chat_sessions (uid_resume doc)
            try:
                doc = get_resume_doc(resume_id)
                if doc:
                    out = doc.get("structured_resume") or doc
                    if out and isinstance(out, dict):
                        return out
            except Exception as e:
                if _is_chroma_rate_limit_error(e):
                    raise
                log.debug(f"Resume doc fetch for {resume_id}: {e}")

            # Fallback: legacy record in resume collection with stored document
            try:
                results = collection.get(ids=[resume_id])
                if results.get("ids") and results["ids"]:
                    raw = (results.get("documents") or [None])[0]
                    if raw and isinstance(raw, str) and raw.strip():
                        return json.loads(raw)
            except Exception as e:
                if _is_chroma_rate_limit_error(e):
                    raise
                log.debug(f"Legacy resume fetch for {resume_id}: {e}")

            return None

        except Exception as e:
            last_error = e
            if _is_chroma_rate_limit_error(e) and attempt < _CHROMA_RATE_LIMIT_RETRIES - 1:
                backoff = _CHROMA_RATE_LIMIT_BACKOFF_BASE ** (attempt + 1)
                log.warning(f"⚠️ ChromaDB rate limit for {resume_id}, retrying in {backoff:.1f}s (attempt {attempt + 1}/{_CHROMA_RATE_LIMIT_RETRIES})")
                time.sleep(backoff)
            else:
                log.error(f"❌ Error retrieving resume {resume_id}: {e}")
                return None

    return None

def match_job_description(jd_text: str, top_k: int = 5, where_clause: dict = None, tenant_id: str = None):
    """
    Match job description against resumes using optimized query if available.
    
    Args:
        jd_text: Job description text
        top_k: Number of results to return
        where_clause: Optional metadata filter (e.g., {"location": "Remote"})
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        Query results from ChromaDB
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, "match_job_description")
    
    # Tenant isolation: build combined where clause
    effective_where = _build_tenant_where_clause(tenant_id, where_clause)
    
    # Use optimized query if available (only in async contexts)
    manager = _get_optimized_manager()
    if manager:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                # Can use optimized query in non-running loop
                results = loop.run_until_complete(
                    manager.query_with_cache(
                        collection_name="resume",
                        query_texts=[jd_text],
                        n_results=top_k,
                        where=effective_where,
                        use_cache=True,
                        tenant_id=tenant_id  # Cache isolation by tenant
                    )
                )
                return results
        except (RuntimeError, Exception) as e:
            log.debug(f"Optimized query not available, using direct: {e}")
    
    # Fallback to direct query (include documents so ranker can parse resume from metadata or document)
    results = collection.query(
        query_texts=[jd_text],
        n_results=top_k,
        where=effective_where,
        include=["metadatas", "documents", "distances"]
    )
    return results

def insert_job_description(job_id: str, jd_data: dict, metadata: dict = None, tenant_id: str = None):
    """
    Insert job description data into the job_descriptions collection.
    
    Args:
        jd_id: Unique identifier for the job description (e.g., jd_id or company_jd_id)
        jd_data: Dictionary containing parsed job description data
        metadata: Optional metadata for the job description
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    if not _validate_tenant_id(tenant_id, f"insert_job_description({job_id})"):
        return False
    
    import json
    
    # Convert JD data to JSON string for storage
    jd_text = json.dumps(jd_data, indent=2)
    
    # Issue 3.1: Get embedding model name for version tracking
    embedding_model = os.getenv("CHROMA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # Extract job_domains if available (optional field from newer JD parsing)
    job_domains_raw = jd_data.get("jobDomains") or jd_data.get("job_domains") or []
    job_domains_str = ",".join(job_domains_raw[:3]) if job_domains_raw else ""
    # Store primary domain as a single filterable field (ChromaDB $eq works on strings)
    job_domain_primary = job_domains_raw[0].lower().strip() if job_domains_raw else ""

    # Prepare metadata
    jd_metadata = {
        "job_id": job_id,
        "job_title": jd_data.get("jobTitle", ""),
        "location": jd_data.get("location", ""),
        "experience": jd_data.get("experience", ""),
        "education_required": jd_data.get("educationRequired", ""),
        "job_type": jd_data.get("jobType", ""),
        "company": jd_data.get("company", ""),
        "skills_count": len(jd_data.get("requiredSkills", [])),
        "job_domains": job_domains_str,
        "job_domain_primary": job_domain_primary,
        "file_url": jd_data.get("fileUrl", ""),
        "embedding_model": embedding_model,
        **(metadata or {})
    }
    
    # Tenant isolation: add tenant_id to metadata if provided
    jd_metadata = _add_tenant_to_metadata(jd_metadata, tenant_id)
    
    # Normalize metadata to ensure all values are ChromaDB-compatible primitives
    jd_metadata = normalize_metadata(jd_metadata)
    
    # Upsert to handle both insert and update
    job_descriptions_collection.upsert(
        ids=[job_id],
        documents=[jd_text],
        metadatas=[jd_metadata]
    )
    log.info(f"✅ Job description {job_id} upserted into job_descriptions collection.")

def get_job_description(job_id: str):
    """
    Retrieve job description data by jd_id.
    
    Args:
        jd_id: Unique identifier for the job description
        
    Returns:
        Dictionary containing the job description data or None if not found
    """
    try:
        # Ensure jd_id is a string, not a list
        if isinstance(job_id, list):
            job_id = job_id[0]
        
        results = job_descriptions_collection.get(ids=[job_id])
        if results['ids']:
            import json
            jd_data = json.loads(results['documents'][0])
            return jd_data
        return None
    except Exception as e:
        log.error(f"Error retrieving job description {job_id}: {e}")
        return None

def update_job_description(job_id: str, jd_data: dict, metadata: dict = None, tenant_id: str = None):
    """
    Update existing job description data.
    
    Args:
        jd_id: Unique identifier for the job description
        jd_data: Updated dictionary containing job description data
        metadata: Optional metadata for the job description
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    if not _validate_tenant_id(tenant_id, f"update_job_description({job_id})"):
        return False
    
    import json
    
    # Convert JD data to JSON string for storage
    jd_text = json.dumps(jd_data, indent=2)
    
    # Extract job_domains (same logic as insert_job_description)
    job_domains_raw = jd_data.get("jobDomains") or jd_data.get("job_domains") or []
    job_domains_str = ",".join(job_domains_raw[:3]) if job_domains_raw else ""
    job_domain_primary = job_domains_raw[0].lower().strip() if job_domains_raw else ""
    
    # Prepare metadata
    jd_metadata = {
        "job_id": job_id,
        "job_title": jd_data.get("jobTitle", ""),
        "location": jd_data.get("location", ""),
        "experience": jd_data.get("experience", ""),
        "education_required": jd_data.get("educationRequired", ""),
        "job_type": jd_data.get("jobType", ""),
        "company": jd_data.get("company", ""),
        "skills_count": len(jd_data.get("requiredSkills", [])),
        "job_domains": job_domains_str,
        "job_domain_primary": job_domain_primary,
        "file_url": jd_data.get("fileUrl", ""),
        **(metadata or {})
    }
    
    # Tenant isolation: add tenant_id to metadata if provided
    jd_metadata = _add_tenant_to_metadata(jd_metadata, tenant_id)
    
    # Normalize metadata to ensure all values are ChromaDB-compatible primitives
    jd_metadata = normalize_metadata(jd_metadata)
    
    # Update the existing document
    try:
        job_descriptions_collection.update(
            ids=[job_id],
            documents=[jd_text],
            metadatas=[jd_metadata]
        )
        log.info(f"✅ Job description {job_id} updated successfully.")
    except Exception as e:
        # Try to insert if update fails (document might not exist)
        try:
            job_descriptions_collection.upsert(
                ids=[job_id],
                documents=[jd_text],
                metadatas=[jd_metadata]
            )
            log.info(f"✅ Job description {job_id} inserted (update failed, used upsert).")
        except Exception as e2:
            log.error(f"❌ Error updating/inserting job description {job_id}: {e2}")

def query_job_descriptions(query_text: str, top_k: int = 5, where_clause: dict = None, tenant_id: str = None):
    """
    Query job descriptions using semantic search.
    Uses optimized cached query if available.
    
    Args:
        query_text: Text to search for (e.g., skills, job title)
        top_k: Number of results to return
        where_clause: Optional metadata filter (e.g., {"location": "Remote"})
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        Query results from ChromaDB
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, "query_job_descriptions")
    
    # Tenant isolation: build combined where clause
    effective_where = _build_tenant_where_clause(tenant_id, where_clause)
    
    try:
        # Use optimized query if available (only in async contexts)
        manager = _get_optimized_manager()
        if manager:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    # Use optimized cached query
                    results = loop.run_until_complete(
                        manager.query_with_cache(
                            collection_name="job_descriptions",
                            query_texts=[query_text],
                            n_results=top_k,
                            where=effective_where,
                            use_cache=True,
                            tenant_id=tenant_id  # Cache isolation by tenant
                        )
                    )
                    return results
            except (RuntimeError, Exception) as e:
                log.debug(f"Optimized query not available, using direct: {e}")
        
        # Fallback to direct query
        results = job_descriptions_collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=effective_where
        )
        return results
    except Exception as e:
        log.error(f"Error querying job descriptions: {e}")
        return None


def get_top_matched_jobs_for_candidate(
    uid: str,
    exclude_job_id: str,
    top_k: int = 10,
    tenant_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Return top jobs this candidate has already been matched with (from cached candidate_job_rankings).
    Uses stored match_score for ranking. Falls back to empty list if no cached matches.
    Output format matches get_similar_jobs_for_job for drop-in compatibility (job_id, job_title, company, location, similarity, required_skills).

    Args:
        uid: Candidate user ID
        exclude_job_id: Job ID to exclude (the one currently being viewed)
        top_k: Max number of similar jobs to return
        tenant_id: Optional tenant identifier for isolation

    Returns:
        List of dicts with job_id, job_title, company, location, similarity (match_score), required_skills
    """
    if not uid or not exclude_job_id:
        return []
    try:
        _validate_tenant_id(tenant_id, "get_top_matched_jobs_for_candidate")
        exclude_str = str(exclude_job_id)
        res = candidate_job_rankings_collection.get(
            where={"uid": uid},
            limit=300,  # ChromaDB cloud quota limit for Get action
            include=["documents", "metadatas"],
        )
        if not res or not res.get("ids"):
            return []
        ids = res["ids"] or []
        docs = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        candidates = []
        for i, doc_id in enumerate(ids):
            if "_similar" in str(doc_id):
                continue
            meta = metadatas[i] if i < len(metadatas) else {}
            job_id_val = meta.get("job_id") or ""
            if not job_id_val and doc_id:
                parts = str(doc_id).split("_", 1)
                job_id_val = parts[1] if len(parts) > 1 else ""
            if job_id_val == exclude_str:
                continue
            doc_str = docs[i] if i < len(docs) else None
            if not doc_str:
                continue
            try:
                data = json.loads(doc_str)
                match_score = data.get("match_score")
                if match_score is None:
                    continue
                try:
                    ms = float(match_score)
                except (TypeError, ValueError):
                    continue
                candidates.append((job_id_val, ms))
            except (json.JSONDecodeError, TypeError):
                continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]
        if not candidates:
            return []
        out = []
        for jid, score in candidates:
            jd = get_job_description(jid)
            title = ""
            company = ""
            location = ""
            required_skills = []
            if jd:
                title = jd.get("jobTitle") or jd.get("title") or jid
                company = jd.get("company") or ""
                location = jd.get("location") or jd.get("jobLocation") or ""
                rs = jd.get("requiredSkills") or jd.get("required_skills") or []
                required_skills = rs[:10] if isinstance(rs, list) else [s.strip() for s in str(rs).split(",")][:10]
            score_val = round(float(score), 4)
            out.append({
                "job_id": jid,
                "job_title": title or jid,
                "company": company,
                "location": location,
                "similarity": score_val,
                "match_score": score_val,
                "required_skills": required_skills,
            })
        log.info(f"📎 Found {len(out)} top matched jobs for candidate {uid} (from cache)")
        return out
    except Exception as e:
        log.warning("get_top_matched_jobs_for_candidate failed: %s", e)
        return []


def get_similar_jobs_for_job(
    job_description: Dict[str, Any],
    applied_job_id: str,
    top_k: int = 10,
    tenant_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Find jobs in the database similar to the given job (e.g. the one the candidate applied to).
    Uses ChromaDB semantic search on job_descriptions; excludes the applied job.

    Args:
        job_description: The applied job's description dict (jobTitle, requiredSkills, description, etc.)
        applied_job_id: Job ID to exclude from results
        top_k: Max number of similar jobs to return
        tenant_id: Optional tenant identifier for isolation

    Returns:
        List of dicts with job_id, job_title, company, location, similarity, required_skills (optional)
    """
    try:
        _validate_tenant_id(tenant_id, "get_similar_jobs_for_job")
        title = job_description.get("jobTitle") or job_description.get("title") or ""
        skills = job_description.get("requiredSkills") or job_description.get("required_skills") or []
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",")]
        desc = job_description.get("description") or job_description.get("job_description") or ""
        company = job_description.get("company") or ""
        query_text = f"{title} {' '.join(skills)} {desc}".strip() or title
        if not query_text:
            return []

        fetch = top_k + 5  # fetch extra so we have enough after excluding applied
        results = query_job_descriptions(query_text, top_k=fetch, where_clause=None, tenant_id=tenant_id)
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results.get("distances")
        dist_list = (distances[0] if distances else []) or [0.0] * len(ids)
        documents = results.get("documents")
        doc_list = (documents[0] if documents else []) or []

        similar = []
        for i, jid in enumerate(ids):
            jid_str = str(jid)
            if jid_str == str(applied_job_id):
                continue
            if len(similar) >= top_k:
                break
            distance = dist_list[i] if i < len(dist_list) else 1.0
            similarity = max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
            doc = doc_list[i] if i < len(doc_list) else None
            job_title = title
            company_val = company
            location = ""
            required_skills = []
            if doc:
                try:
                    if isinstance(doc, str):
                        data = json.loads(doc)
                    else:
                        data = doc if isinstance(doc, dict) else {}
                    job_title = data.get("jobTitle") or data.get("title") or jid_str
                    company_val = data.get("company") or ""
                    location = data.get("location") or data.get("jobLocation") or ""
                    rs = data.get("requiredSkills") or data.get("required_skills") or []
                    required_skills = rs if isinstance(rs, list) else [s.strip() for s in str(rs).split(",")]
                except Exception:
                    pass
            sim_val = round(similarity, 4)
            similar.append({
                "job_id": jid_str,
                "job_title": job_title,
                "company": company_val,
                "location": location,
                "similarity": sim_val,
                "match_score": sim_val,
                "required_skills": required_skills[:10],
            })
        log.info(f"📎 Found {len(similar)} similar jobs for applied job {applied_job_id}")
        return similar
    except Exception as e:
        log.warning(f"Failed to get similar jobs: {e}")
        return []


def match_resumes_to_job(job_id: str, top_k: int = 10, tenant_id: str = None):
    """
    Find best matching resumes for a given job description.
    
    Args:
        jd_id: Job description ID
        top_k: Number of top matching resumes to return
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        List of matching resume IDs with similarity scores
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"match_resumes_to_job({job_id})")
    
    try:
        # Get the job description
        jd_data = get_job_description(job_id)
        if not jd_data:
            log.warning(f"Job description {job_id} not found")
            return None
        
        # Create search query from JD data
        search_query = f"{jd_data.get('jobTitle', '')} {' '.join(jd_data.get('requiredSkills', []))}"
        
        # Tenant isolation: build where clause
        effective_where = _build_tenant_where_clause(tenant_id)
        
        # Search in resume collection
        results = collection.query(
            query_texts=[search_query],
            n_results=top_k,
            where=effective_where
        )
        
        return results
    except Exception as e:
        log.error(f"Error matching resumes to job {job_id}: {e}")
        return None

def find_jd_by_company(company: str, top_k: int = 5, tenant_id: str = None):
    """
    Find job descriptions by company name.
    
    Args:
        company: Company name to search for
        top_k: Number of results to return
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        List of job description IDs
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"find_jd_by_company({company})")
    
    # Tenant isolation: build combined where clause
    company_filter = {"company": company}
    effective_where = _build_tenant_where_clause(tenant_id, company_filter)
    
    try:
        results = job_descriptions_collection.query(
            query_texts=[company],
            n_results=top_k,
            where=effective_where
        )
        
        if not results or not results.get('ids') or len(results['ids']) == 0:
            return None
        
        job_ids = results['ids']
        if isinstance(job_ids[0], list):
            job_ids = job_ids[0]
        
        return job_ids
        
    except Exception as e:
        log.error(f"Error finding job descriptions for company {company}: {e}")
        return None


def verify_chromadb_connection():
    """
    Verify ChromaDB connection and  diagnostic information.
    Useful for debugging connection issues.
    """
    log.debug("\n" + "="*60)
    log.debug("🔍 CHROMADB CONNECTION VERIFICATION")
    log.debug("="*60)
    
    # Check environment variables
    import os
    log.debug("\n📋 Environment Variables:")
    log.debug(f"   APP_ENV: {os.getenv('APP_ENV', 'NOT SET')}")
    log.debug(f"   CHROMA_API_KEY: {'SET' if os.getenv('CHROMA_API_KEY') else 'NOT SET'} ({'*' * (len(os.getenv('CHROMA_API_KEY', '')) - 4) + os.getenv('CHROMA_API_KEY', '')[-4:] if len(os.getenv('CHROMA_API_KEY', '')) > 4 else '***'})")
    log.debug(f"   CHROMA_TENANT: {os.getenv('CHROMA_TENANT', 'NOT SET')}")
    log.debug(f"   CHROMA_DATABASE: {os.getenv('CHROMA_DATABASE', 'NOT SET')}")
    
    # Check settings
    log.debug("\n📋 Settings (from settings.py):")
    log.debug(f"   APP_ENV: {settings.APP_ENV}")
    log.debug(f"   CHROMA_TENANT: {settings.CHROMA_TENANT}")
    log.debug(f"   CHROMA_DATABASE: {settings.CHROMA_DATABASE}")
    log.debug(f"   CHROMA_API_KEY: {'*' * (len(settings.CHROMA_API_KEY) - 4) + settings.CHROMA_API_KEY[-4:] if len(settings.CHROMA_API_KEY) > 4 else '***'}")
    
    # Check client connection
    log.debug("\n📋 Client Connection:")
    try:
        log.debug(f"   Client type: {type(client).__name__}")
        log.debug(f"   Client tenant: {getattr(client, 'tenant', 'N/A')}")
        log.debug(f"   Client database: {getattr(client, 'database', 'N/A')}")
    except Exception as e:
        log.error(f"   ❌ Error accessing client: {e}")
    
    # List all collections
    log.debug("\n📋 Collections in Database:")
    try:
        collections = client.list_collections()
        log.debug(f"   Total collections: {len(collections)}")
        for col in collections:
            try:
                count = col.count()
                log.debug(f"   - {col.name}: {count} documents")
            except Exception as e:
                log.warning(f"   - {col.name}: Error getting count ({e})")
    except Exception as e:
        import traceback
        log.error(f"   ❌ Error listing collections: {e}")
        log.error(f"   Traceback: {traceback.format_exc()}")
    
    # Check job_descriptions collection specifically
    log.debug("\n📋 job_descriptions Collection:")
    try:
        count = job_descriptions_collection.count()
        log.debug(f"   Document count: {count}")
        
        if count > 0:
            # Try to get a sample
            sample = job_descriptions_collection.peek(limit=1)
            if sample and sample.get('ids'):
                log.debug(f"   Sample ID: {sample['ids'][0] if sample['ids'] else 'None'}")
        else:
            log.warning("   ⚠️ Collection is empty!")
    except Exception as e:
        import traceback
        log.error(f"   ❌ Error accessing job_descriptions collection: {e}")
        log.error(f"   Traceback: {traceback.format_exc()}")
    
    log.debug("="*60 + "\n")
    
    log.info("ChromaDB connection verification completed")

# Existing functions remain unchanged...
def insert_chat_session(session_id: str, session_data: dict, metadata: dict = None, tenant_id: str = None):
    """
    Insert chat session data into the chat_sessions collection with chunking.
    IDs now include uid: main id = "{uid}_{session_id}".
    
    Args:
        session_id: Session identifier
        session_data: Session data dictionary
        metadata: Optional additional metadata
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    import json
    
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    if not _validate_tenant_id(tenant_id, f"insert_chat_session({session_id})"):
        pass  # Continue for backward compatibility

    # Ensure session_data is JSON-serializable (guard against accidental coroutines, etc.)
    session_data = _make_json_safe(session_data)

    ok, missing = _validate_session_data(session_data)
    if not ok:
        log.error(f"❌ Invalid session_data missing fields: {missing}")
        return False

    uid = session_data.get("uid", "")
    main_id = f"{uid}_{session_id}" if uid else session_id

    data_hash = _calculate_data_hash(session_data)

    # Convert to compact JSON (no spaces) to save bytes
    session_text = json.dumps(session_data, separators=(",", ":"), ensure_ascii=False)

    # Split JSON text into byte-safe chunks (no overlap for JSON integrity)
    chunks = _split_text_into_byte_chunks(session_text, max_bytes=14000, overlap_chars=0)
    num_chunks = len(chunks)

    # Prepare shared metadata (includes tenant_id if provided)
    base_meta = {
        "session_id": session_id,
        "timestamp": session_data.get("timestamp", ""),
        "uid": uid,
        "resume_url": session_data.get("resume_url", ""),
        "status": session_data.get("status", "in_progress"),
        "data_hash": data_hash,
        **(metadata or {}),
    }
    # Tenant isolation: add tenant_id to metadata if provided
    base_meta = _add_tenant_to_metadata(base_meta, tenant_id)

    if num_chunks == 1:
        # Single-chunk: keep main id as {uid}_{session_id} when uid present
        doc = chunks[0]
        doc_size = len(doc.encode("utf-8"))
        chat_sessions_collection.upsert(
            ids=[main_id],
            documents=[doc],
            metadatas=[_normalize_metadata({**base_meta, "doc_size": doc_size})],
        )
        log.info(f"✅ Chat session {main_id} stored (single chunk, {doc_size} bytes)")
        return True

    # Multi-chunk: first chunk is main doc in chat_sessions; others go to session_chunks
    main_doc = chunks[0]
    main_size = len(main_doc.encode("utf-8"))

    chat_sessions_collection.upsert(
        ids=[main_id],
        documents=[main_doc],
        metadatas=[_normalize_metadata({**base_meta, "doc_size": main_size, "num_chunks": num_chunks})],
    )

    # Store remaining chunks
    expanded_ids: list[str] = []
    expanded_docs: list[str] = []
    expanded_metas: list[dict] = []

    for i, chunk in enumerate(chunks[1:], start=2):  # chunk_2..N (1 is main)
        cid = f"{main_id}_chunk_{i}"
        expanded_ids.append(cid)
        expanded_docs.append(chunk)
        expanded_metas.append(_normalize_metadata({
            **base_meta,
            "source_id": main_id,
            "chunk_num": i,
            "num_chunks": num_chunks,
            "priority": "MEDIUM",
        }))

    if expanded_ids:
        session_chunks_collection.add(
            ids=expanded_ids,
            documents=expanded_docs,
            metadatas=expanded_metas,
        )
    log.info(f"✅ Chat session {main_id} stored with {num_chunks} chunks (main+{num_chunks-1} extras)")
    return True

def _compress_session_data(session_data: dict) -> dict:
    """
    Aggressively compress session data to fit within Chroma's 16KB limit.
    
    Args:
        session_data: Original session data
        
    Returns:
        Heavily compressed session data
    """
    compressed = {}
    
    # Keep only essential top-level fields
    essential_top_level = ["uid", "timestamp", "status"]
    for field in essential_top_level:
        if field in session_data:
            compressed[field] = session_data[field]
    
    # PRIORITY: Always include structured_resume but compress it heavily
    if "structured_resume" in session_data:
        structured = session_data["structured_resume"]
        if isinstance(structured, dict):
            # Ultra-compressed structured resume - only the most essential fields
            compressed_resume = {
                "Name": structured.get("Name", ""),
                "Skills": structured.get("Skills", [])[:5] if isinstance(structured.get("Skills"), list) else [],  # Limit to 5 skills
                "WorkExperience": structured.get("WorkExperience", [])[:2] if isinstance(structured.get("WorkExperience"), list) else []  # Limit to 2 experiences
            }
            compressed["structured_resume"] = compressed_resume
        else:
            # If structured_resume is not a dict, keep it as is
            compressed["structured_resume"] = structured
    
    # ULTRA-AGGRESSIVE agent compression to make room for structured_resume
    agent_compression_map = {
        "personal_info_parser": ["name"],  # Only name, no contact details
        "education_parser": ["education"],
        "experience_parser": ["work_experience"],  # No certifications
        "skills_parser": ["skills"],
        "resume_assembler": ["status"],
        "resume_score": ["ResumeScore"],  # Only overall score
        "assessment_evaluator": ["assessment_results"],
        "interest_filler": ["user_interests"],
        "assessment_question_generator": ["generated_questions"],  # Generated questions
        "report_generator": ["report"],
        "career_advisor": ["raw_skill_gap_analysis_output"],
        "market_and_course_recommender": ["market_insights"],  # Only insights
        "assessment_recommender": ["assessment_plan"]  # Only plan
    }
    
    for agent_name, essential_fields in agent_compression_map.items():
        if agent_name in session_data:
            agent_data = session_data[agent_name]
            if isinstance(agent_data, dict):
                compressed_agent = {}
                for field in essential_fields:
                    if field in agent_data:
                        value = agent_data[field]
                        # ULTRA-AGGRESSIVE compression of large arrays/objects
                        if field in ["skills", "education", "work_experience", "certifications"] and isinstance(value, list):
                            compressed_agent[field] = value[:3]  # Limit arrays to 3 items
                        elif field in ["assessment_results", "generated_questions", "raw_skill_gap_analysis_output"] and isinstance(value, dict):
                            # Keep only essential parts of complex objects
                            if field == "assessment_results":
                                compressed_agent[field] = {
                                    "total_score": value.get("total_score", 0),
                                    "section_scores": value.get("section_scores", {})
                                }
                            elif field == "generated_questions":
                                # Keep essential question data but limit to first few questions
                                questions_data = value.get("questions", {}).get("multi", {}).get("questions", [])
                                compressed_agent[field] = {
                                    "ok": value.get("ok", False),
                                    "total_questions": value.get("total_questions", 0),
                                    "questions": {"multi": {"questions": questions_data[:3]}} if questions_data else {}
                                }
                            else:
                                compressed_agent[field] = value
                        else:
                            compressed_agent[field] = value
                
                if compressed_agent:  # Only add if we have data
                    compressed[agent_name] = compressed_agent
    
    return compressed

def update_chat_session(session_id: str, session_data: dict, metadata: dict = None, tenant_id: str = None):
    """
    Update chat session with delete-and-rewrite chunking.
    IDs now include uid: main id = "{uid}_{session_id}".
    
    Args:
        session_id: Session identifier
        session_data: Session data dictionary
        metadata: Optional additional metadata
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    import json
    
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"update_chat_session({session_id})")

    # Ensure session_data is JSON-serializable (guard against accidental coroutines, etc.)
    session_data = _make_json_safe(session_data)

    ok, missing = _validate_session_data(session_data)
    if not ok:
        log.error(f"❌ Invalid session_data missing fields: {missing}")
        return False

    uid = session_data.get("uid", "")
    main_id = f"{uid}_{session_id}" if uid else session_id
    
    # Tenant isolation: prepare history metadata with tenant_id
    history_meta = {
        "source_id": main_id,
        "change_type": "update",
        "timestamp": session_data.get("timestamp", ""),
    }
    history_meta = _add_tenant_to_metadata(history_meta, tenant_id)

    # Write history snapshot (best-effort)
    try:
        prev = chat_sessions_collection.get(ids=[main_id])
        prev_doc = (prev.get("documents") or [None])[0]
        if prev_doc:
            history_meta["prev_hash"] = (prev.get("metadatas") or [{}])[0].get("data_hash", "")
            session_history_collection.add(
                ids=[f"{main_id}_hist_{int(__import__('time').time()*1000)}"],
                documents=[prev_doc],
                metadatas=[_normalize_metadata(history_meta)],
            )
    except Exception:
        pass

    # Delete old chunks for this session main_id
    try:
        session_chunks_collection.delete(where={"source_id": main_id})
    except Exception:
        pass

    # Re-insert via chunking (pass tenant_id)
    return insert_chat_session(session_id, session_data, metadata, tenant_id)

def _parse_session_doc(doc_text: str, session_id: str, uid: str | None) -> dict | None:
    """Parse session document JSON; on failure (e.g. truncated/chunked doc) try reconstruction."""
    import json
    try:
        return json.loads(doc_text)
    except json.JSONDecodeError as e:
        # Document may be first chunk only (multi-chunk session) -> "Unterminated string"
        log.warning(
            "Chat session document parse failed (may be chunked); trying reconstruction | session_id=%s uid=%s err=%s",
            session_id, uid or "", str(e)[:200]
        )
        return get_session_with_reconstruction(session_id, uid)
    return None


def get_chat_session(session_id: str, uid: str | None = None, tenant_id: str = None):
    """
    Retrieve chat session data by ID.
    Backward compatible: tries provided id directly; if uid is given, also tries `{uid}_{session_id}`;
    if not found, queries by metadata `session_id` (and uid if provided) to discover the stored id.
    On JSON parse error (e.g. unterminated string from single-chunk read of multi-chunk session),
    falls back to get_session_with_reconstruction to reassemble from chunks.
    
    Args:
        session_id: Session identifier
        uid: Optional user identifier
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"get_chat_session({session_id})")
    
    if chat_sessions_collection is None:
        log.debug("ChromaDB unavailable, cannot retrieve chat session")
        return None
    
    def _try_session_doc(results, sid, u):
        """Parse doc from get() result; if session is multi-chunk, use reconstruction and avoid parse warning."""
        if not results.get('ids'):
            return None
        doc = (results.get('documents') or [None])[0]
        if not doc:
            return None
        metas = results.get('metadatas') or []
        num_chunks = int(metas[0].get('num_chunks', 1)) if metas else 1
        if num_chunks > 1:
            return get_session_with_reconstruction(sid, u)
        return _parse_session_doc(doc, sid, u)

    try:
        # 1) Direct by given id (works if caller passes full stored id)
        results = chat_sessions_collection.get(ids=[session_id])
        out = _try_session_doc(results, session_id, uid)
        if out is not None:
            return out

        # 2) If uid provided, try composed id
        if uid:
            composed_id = f"{uid}_{session_id}"
            results2 = chat_sessions_collection.get(ids=[composed_id])
            out = _try_session_doc(results2, session_id, uid)
            if out is not None:
                return out

        # 3) Discover by metadata (with optional tenant filter)
        where = {"session_id": session_id}
        if uid:
            where = {"$and": [{"session_id": session_id}, {"uid": uid}]}
        # Tenant isolation: add tenant filter if provided
        where = _build_tenant_where_clause(tenant_id, where) or where
        
        resq = chat_sessions_collection.query(
            query_texts=["session data"], n_results=5, where=where
        )
        ids = resq.get('ids') or []
        if ids:
            if isinstance(ids[0], list) and ids[0]:
                found_id = ids[0][0]
            elif isinstance(ids, list):
                found_id = ids[0]
            else:
                found_id = None
            if found_id:
                res3 = chat_sessions_collection.get(ids=[found_id])
                out = _try_session_doc(res3, session_id, uid)
                if out is not None:
                    return out
        return None
    except Exception as e:
        log.error(f"Error retrieving chat session {session_id}: {e}")
        return None

def query_chat_sessions(query_text: str, top_k: int = 5, where_clause: dict = None, tenant_id: str = None):
    """
    Query chat sessions using semantic search.
    
    Args:
        query_text: Text to search for
        top_k: Number of results to return
        where_clause: Optional metadata filter
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        Query results from ChromaDB
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, "query_chat_sessions")
    
    # Tenant isolation: build combined where clause
    effective_where = _build_tenant_where_clause(tenant_id, where_clause)
    
    if chat_sessions_collection is None:
        log.debug("ChromaDB unavailable, cannot query chat sessions")
        return None
    try:
        results = chat_sessions_collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=effective_where
        )
        return results
    except Exception as e:
        log.error(f"Error querying chat sessions: {e}")
        return None

def find_session_by_uid(uid: str, tenant_id: str = None):
    """
    Find the most recent chat session for a given UID.
    
    Args:
        uid: User ID to search for
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        Session ID if found, None otherwise
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"find_session_by_uid({uid})")
    
    # Tenant isolation: build combined where clause
    uid_filter = {"uid": uid}
    effective_where = _build_tenant_where_clause(tenant_id, uid_filter)
    
    try:
        # Get ALL sessions for this UID to find the most recent one
        results = chat_sessions_collection.query(
            query_texts=["session data"],
            n_results=100,  # Get more results to find the most recent
            where=effective_where
        )
        
        if not results or not results.get('ids') or len(results['ids']) == 0:
            return None
        
        # Get session data for all matching sessions to find the most recent
        session_ids = results['ids']
        if isinstance(session_ids[0], list):
            session_ids = session_ids[0]
        
        most_recent_session_id = None
        most_recent_timestamp = None
        
        for session_id in session_ids:
            session_data = get_chat_session(session_id)
            if session_data and 'timestamp' in session_data:
                session_timestamp = session_data['timestamp']
                # Normalize timestamp for comparison: convert to string if needed
                if isinstance(session_timestamp, (int, float)):
                    # Convert numeric timestamp to ISO string for consistency
                    from datetime import datetime
                    try:
                        session_timestamp = datetime.fromtimestamp(float(session_timestamp)).isoformat()
                    except (ValueError, OSError):
                        # If conversion fails, convert to string for comparison
                        session_timestamp = str(session_timestamp)
                elif not isinstance(session_timestamp, str):
                    session_timestamp = str(session_timestamp)
                
                # Compare timestamps (ISO strings compare lexicographically)
                if most_recent_timestamp is None or session_timestamp > most_recent_timestamp:
                    most_recent_timestamp = session_timestamp
                    most_recent_session_id = session_id
        
        # If no timestamp found, return the first one (fallback)
        if most_recent_session_id is None and session_ids:
            most_recent_session_id = session_ids[0]
        
        return most_recent_session_id
        
    except Exception as e:
        log.error(f"Error finding session by UID {uid}: {e}")
        return None


def list_sessions_by_uid(
    uid: str,
    limit: int = 10,
    exclude_session_id: str | None = None,
) -> list[str]:
    """
    List recent chat session IDs for a UID, ordered by timestamp descending.
    Excludes the current session (and goals pseudo-session) for prior-context loading.

    Args:
        uid: User ID
        limit: Max number of session IDs to return
        exclude_session_id: Session ID to exclude (e.g. current chat)

    Returns:
        List of logical session_id strings (most recent first), excluding
        exclude_session_id and goals session ({uid}_goals).
    """
    if chat_sessions_collection is None:
        return []
    try:
        from datetime import datetime

        res = chat_sessions_collection.get(where={"uid": uid}, include=["metadatas"])
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") or []

        candidates = []
        goals_sid = f"{uid}_goals"
        for i, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                continue
            sid = meta.get("session_id")
            if not sid or (isinstance(sid, str) and sid.endswith("_goals")):
                continue
            if sid == exclude_session_id or sid == goals_sid:
                continue
            ts = meta.get("timestamp")
            doc_id = ids[i] if i < len(ids) else None
            candidates.append((sid, ts, doc_id))

        def _ts_key(t):
            _, ts, _ = t
            if ts is None:
                return ""
            if isinstance(ts, (int, float)):
                try:
                    return datetime.fromtimestamp(float(ts)).isoformat()
                except (ValueError, OSError):
                    return str(ts)
            return str(ts)

        candidates.sort(key=_ts_key, reverse=True)
        return [c[0] for c in candidates[:limit]]
    except Exception as e:
        log.debug(f"list_sessions_by_uid uid={uid}: {e}")
        return []


def fetch_structured_resume(uid: str, tenant_id: str = None, current_session_id: str = None):
    """
    Fetch structured resume from chat session data for a given UID.
    Prioritizes the current session if provided, then falls back to most recent session.
    
    Args:
        uid: User ID to search for
        tenant_id: Optional tenant ID (for compatibility, not used in ChromaDB)
        current_session_id: Optional current session ID to check first
        
    Returns:
        Structured resume data if found, None otherwise
    """
    if chat_sessions_collection is None:
        log.debug("ChromaDB unavailable, cannot fetch structured resume")
        return None
    
    try:
        # NEW: Prefer per-UID resume document in chat_sessions (uid_resume)
        try:
            # Lazy import of helpers to avoid circulars during module import
            get_resume_doc  # type: ignore
        except NameError:
            pass
        try:
            doc = get_resume_doc(uid)  # type: ignore
            if doc and isinstance(doc, dict) and doc.get("structured_resume"):
                return doc["structured_resume"]
        except Exception:
            pass

        # First, try the current session if provided
        if current_session_id:
            session_data = get_chat_session(current_session_id)
            if session_data:
                
                # Extract structured resume from resume_parser data
                if "resume_parser" in session_data and "structured_resume" in session_data["resume_parser"]:
                    return session_data["resume_parser"]["structured_resume"]
                
                # Check resume_assembler data
                if "resume_assembler" in session_data and "structured_resume" in session_data["resume_assembler"]:
                    return session_data["resume_assembler"]["structured_resume"]
                
                # Fallback: check if structured_resume is directly in session_data
                if "structured_resume" in session_data:
                    return session_data["structured_resume"]
        
        # If not found in current session, first check the most recent session by UID
        session_id = find_session_by_uid(uid)
        if session_id:
            if not (current_session_id and session_id == current_session_id):
                session_data = get_chat_session(session_id)
                if session_data:
                    if "resume_parser" in session_data and "structured_resume" in session_data["resume_parser"]:
                        return session_data["resume_parser"]["structured_resume"]
                    if "resume_assembler" in session_data and "structured_resume" in session_data["resume_assembler"]:
                        return session_data["resume_assembler"]["structured_resume"]
                    if "structured_resume" in session_data:
                        return session_data["structured_resume"]

        # As a final fallback, scan all sessions for this UID and return the first with resume data
        # Issue 3.2: Log warning about missing tenant isolation
        log.warning(f"SECURITY: get_structured_resume fallback query without tenant_id for uid={uid}")
        try:
            results = chat_sessions_collection.query(
                query_texts=["session data"],
                n_results=100,
                where={"uid": uid}
            )
            session_ids = results.get('ids') or []
            if session_ids and isinstance(session_ids[0], list):
                session_ids = session_ids[0]
            for sid in session_ids:
                # Skip the one we already inspected as current
                if current_session_id and sid == current_session_id:
                    continue
                data = get_chat_session(sid)
                if not data:
                    continue
                keys = list(data.keys())
                if "resume_parser" in data and "structured_resume" in data["resume_parser"]:
                    return data["resume_parser"]["structured_resume"]
                if "resume_assembler" in data and "structured_resume" in data["resume_assembler"]:
                    return data["resume_assembler"]["structured_resume"]
                if "structured_resume" in data:
                    return data["structured_resume"]
        except Exception as e:
            log.error(f"Error scanning sessions for UID {uid}: {e}")

        return None
        
    except Exception as e:
        log.error(f"Error fetching structured resume for UID {uid}: {e}")
        return None

# -------------------------
# NEW: Per-UID multi-document helpers (chat_sessions only)
# -------------------------

MULTI_DOC_SUFFIXES = {
    "resume": "_resume",
    "gapanalyze": "_gapanalyze",
    "assessments": "_assessments",
    "resume_score": "_resume_score",
}

def _doc_id(uid: str, kind: str) -> str:
    return f"{uid}{MULTI_DOC_SUFFIXES[kind]}"

# Fields to remove from each skill before storing in Chroma (no agent reads them from Chroma; reduces doc size / limit errors)
_SKILL_RATIONALE_KEYS = ("PositiveRationale", "NegativeRationale", "HowToImprove")

def _strip_skill_rationales_for_chroma(resume_data: dict) -> dict:
    """
    Return a copy of resume_data with skill proficiency rationale fields removed from
    Skills/skills. Reduces document size and avoids Chroma limit errors; no downstream
    agent reads these fields from the stored resume.
    """
    if not resume_data or not isinstance(resume_data, dict):
        return resume_data
    out = dict(resume_data)
    for skills_key in ("Skills", "skills"):
        raw = out.get(skills_key)
        if not isinstance(raw, list):
            continue
        out[skills_key] = []
        for s in raw:
            if isinstance(s, dict):
                s_trimmed = {k: v for k, v in s.items() if k not in _SKILL_RATIONALE_KEYS}
                out[skills_key].append(s_trimmed)
            else:
                out[skills_key].append(s)
    return out

# Chunking when a single doc would exceed ChromaDB 16KB limit (e.g. uid_resume -> uid_resume_2, uid_resume_3)
CHROMADB_DOC_SIZE_LIMIT = 16384
CHUNK_PAYLOAD_MAX = 15000  # safe margin; each chunk stays under limit
MAX_CHUNKS = 20  # cap to avoid unbounded ids


def _chunk_doc_id(base_id: str, chunk_index: int) -> str:
    """Id for chunk: base_id for 0, base_id_2 for 1, base_id_3 for 2, ..."""
    if chunk_index <= 0:
        return base_id
    return f"{base_id}_{chunk_index + 1}"


def _split_utf8_safe(data: bytes, max_bytes: int) -> list:
    """Split bytes into chunks that do not cut UTF-8 code points."""
    chunks = []
    start = 0
    while start < len(data):
        end = min(start + max_bytes, len(data))
        chunk = data[start:end]
        while end > start and len(chunk) > 0 and (chunk[-1] & 0xC0) == 0x80:
            end -= 1
            chunk = data[start:end]
        chunks.append(chunk.decode("utf-8", errors="replace"))
        start = end
    return chunks


def _size_compact(data: dict) -> dict:
    """
    Compact a dictionary by trimming large arrays and long strings to reduce size.
    Used to keep documents within ChromaDB quota limits.
    """
    compact = dict(data)
    # Trim large arrays conservatively (keep first N items)
    array_fields = ["WorkExperience", "work_experience", "experience", "projects", 
                   "Education", "education", "Skills", "skills", "certifications"]
    for k in array_fields:
        v = compact.get(k)
        if isinstance(v, list):
            if len(v) > 50:
                compact[k] = v[:50]
            # Also trim individual items if they're dicts with long text fields
            if k in ["WorkExperience", "work_experience", "experience"]:
                for item in compact[k]:
                    if isinstance(item, dict):
                        # Trim long responsibilities/descriptions
                        for field in ["responsibilities", "description", "summary"]:
                            if field in item and isinstance(item[field], list):
                                if len(item[field]) > 5:
                                    item[field] = item[field][:5]
                            elif field in item and isinstance(item[field], str):
                                if len(item[field]) > 500:
                                    item[field] = item[field][:500] + "…"

    # Gap-doc structures (uid_gapanalyze): cap market_and_course_recommender, career_advisor, etc.
    for top_key in ("market_and_course_recommender", "skill_and_career_advisor", "career_advisor"):
        block = compact.get(top_key)
        if not isinstance(block, dict):
            continue
        block = dict(block)
        for list_key in ("market_insights", "course_recommendations", "career_paths", "salary_trends",
                         "trending_skills", "improvement_recommendations", "career_advice"):
            if list_key in block and isinstance(block[list_key], list) and len(block[list_key]) > 30:
                block[list_key] = block[list_key][:30]
        raw = block.get("raw_skill_gap_analysis_output")
        if isinstance(raw, dict):
            raw = dict(raw)
            for rk in ("career_paths", "improvement_recommendations", "career_advice", "skill_gaps"):
                if rk in raw and isinstance(raw[rk], list) and len(raw[rk]) > 25:
                    raw[rk] = raw[rk][:25]
            block["raw_skill_gap_analysis_output"] = raw
        mi = block.get("market_insights")
        if isinstance(mi, list):
            for item in mi:
                if isinstance(item, dict):
                    for f in ("insight", "description", "summary", "title"):
                        if f in item and isinstance(item[f], str) and len(item[f]) > 500:
                            item[f] = item[f][:500] + "…"
        compact[top_key] = block

    # Trim rationales in skills
    v = compact.get("skills") or compact.get("Skills")
    if isinstance(v, list):
        for s in v:
            if isinstance(s, dict):
                # Trim Rationale field
                if isinstance(s.get("Rationale"), str) and len(s["Rationale"]) > 1000:
                    s["Rationale"] = s["Rationale"][:1000] + "…"
                # Trim description if present
                if isinstance(s.get("description"), str) and len(s["description"]) > 500:
                    s["description"] = s["description"][:500] + "…"
    
    # Trim long string fields
    string_fields = ["summary", "Summary", "objective", "Objective", "profile", "Profile", "professional_summary"]
    for field in string_fields:
        if field in compact and isinstance(compact[field], str) and len(compact[field]) > 1000:
            compact[field] = compact[field][:1000] + "…"
    
    return compact

def _upsert_session_doc(doc_id: str, payload: dict, metadata: dict | None = None):
    import json
    # Serialize once; compact if large
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    uid = doc_id.rsplit("_", 1)[0] if "_" in doc_id else (metadata or {}).get("uid", "")
    original_size = len(text.encode("utf-8"))
    if original_size > 15000:
        payload = _size_compact(payload)
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        log.debug(f"chroma.compact doc={doc_id} uid={uid} original_size={original_size} compacted_size={len(text.encode('utf-8'))}")

    text_bytes = text.encode("utf-8")
    total_size = len(text_bytes)

    # Single doc under limit: write as before
    if total_size <= CHUNK_PAYLOAD_MAX:
        try:
            normalized_meta = normalize_metadata({**(metadata or {}), "doc_size": total_size})
            chat_sessions_collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[normalized_meta],
            )
            log.debug(f"chroma.upsert doc={doc_id} uid={uid} size={total_size}")
            return {"id": doc_id, "doc_size": total_size}
        except Exception as _e:
            log.error(f"chroma.upsert_failed doc={doc_id} uid={uid} err={str(_e)}")
            raise

    # Over limit: split into doc_id, doc_id_2, doc_id_3, ... (e.g. uid_gapanalyze, uid_gapanalyze_2; uid_resume, uid_resume_2)
    try:
        old_total = 1
        try:
            res = chat_sessions_collection.get(ids=[doc_id], include=["metadatas"])
            if res.get("metadatas") and res["metadatas"]:
                old_total = int(res["metadatas"][0].get("chroma_total_chunks") or 1)
        except Exception:
            pass

        chunks = _split_utf8_safe(text_bytes, CHUNK_PAYLOAD_MAX)
        num_chunks = min(len(chunks), MAX_CHUNKS)
        if len(chunks) > MAX_CHUNKS:
            log.warning(f"chroma.chunk doc={doc_id} uid={uid} capped to {MAX_CHUNKS} chunks (had {len(chunks)})")

        chunk_ids = [_chunk_doc_id(doc_id, i) for i in range(num_chunks)]
        chunk_docs = [chunks[i] for i in range(num_chunks)]
        chunk_metas = [
            normalize_metadata({
                **(metadata or {}),
                "doc_size": len(chunks[i].encode("utf-8")),
                "chroma_chunk_index": i,
                "chroma_total_chunks": num_chunks,
            })
            for i in range(num_chunks)
        ]
        chat_sessions_collection.upsert(
            ids=chunk_ids,
            documents=chunk_docs,
            metadatas=chunk_metas,
        )
        log.debug(f"chroma.upsert doc={doc_id} uid={uid} chunked size={total_size} chunks={num_chunks}")

        # Remove trailing chunks if we shrank (e.g. was 4, now 2)
        if num_chunks < old_total:
            to_delete = [_chunk_doc_id(doc_id, i) for i in range(num_chunks, old_total)]
            try:
                chat_sessions_collection.delete(ids=to_delete)
                log.debug(f"chroma.chunk deleted trailing ids={to_delete}")
            except Exception as del_e:
                log.warning(f"chroma.chunk delete trailing failed: {del_e}")

        return {"id": doc_id, "doc_size": total_size, "chunks": num_chunks}
    except Exception as _e:
        log.error(f"chroma.upsert_failed doc={doc_id} uid={uid} err={str(_e)}")
        raise

def _get_session_doc(doc_id: str) -> dict | None:
    try:
        res = chat_sessions_collection.get(ids=[doc_id], include=["documents", "metadatas"])
        docs = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        if not docs:
            return None
        meta = metadatas[0] if metadatas else {}
        total_chunks = int(meta.get("chroma_total_chunks") or 1)
        if total_chunks <= 1:
            import json
            return json.loads(docs[0])
        # Chunked: fetch uid_resume_2, uid_resume_3, ... and concatenate in order
        extra_ids = [_chunk_doc_id(doc_id, i) for i in range(1, total_chunks)]
        extra = chat_sessions_collection.get(ids=extra_ids, include=["documents"])
        extra_docs = extra.get("documents") or []
        parts = [docs[0]] + [extra_docs[i] if i < len(extra_docs) else "" for i in range(len(extra_ids))]
        full_text = "".join(parts)
        import json
        return json.loads(full_text)
    except Exception:
        pass
    return None

def upsert_resume_doc(uid: str, data: dict, metadata: dict = None):
    """
    Upsert resume document with timestamp tracking.
    Preserves created_at for existing resumes, adds it for new ones.
    """
    from datetime import datetime
    
    if metadata is None:
        metadata = {}
    
    # Check if resume already exists to preserve created_at
    try:
        doc_id = _doc_id(uid, "resume")
        result = chat_sessions_collection.get(ids=[doc_id])
        if result.get("metadatas") and result["metadatas"]:
            existing_meta = result["metadatas"][0]
            if "created_at" in existing_meta:
                metadata["created_at"] = existing_meta["created_at"]
    except Exception:
        pass
    
    # Add created_at if not present (new resume)
    if "created_at" not in metadata:
        metadata["created_at"] = datetime.utcnow().isoformat()
    
    # Always update updated_at
    metadata["updated_at"] = datetime.utcnow().isoformat()
    
    return _upsert_session_doc(doc_id, data, {"uid": uid, "kind": "resume", **metadata})

def upsert_gap_doc(uid: str, data: dict, metadata: dict = None):
    """Store gap analysis doc. When size exceeds limit, chunks as uid_gapanalyze, uid_gapanalyze_2, ... (same as resume)."""
    return _upsert_session_doc(_doc_id(uid, "gapanalyze"), data, {"uid": uid, "kind": "gapanalyze", **(metadata or {})})


def upsert_resume_score_doc(uid: str, data: dict, metadata: dict = None):
    """Store resume scorer agent output in a dedicated per-UID document (uid_resume_score)."""
    return _upsert_session_doc(_doc_id(uid, "resume_score"), data, {"uid": uid, "kind": "resume_score", **(metadata or {})})


def get_resume_score_doc(uid: str) -> dict | None:
    """Retrieve resume scorer output for a UID from the dedicated document."""
    return _get_session_doc(_doc_id(uid, "resume_score"))


# Assessment status values we store (lowercase for consistent mentor/context reads)
ASSESSMENT_STATUS_COMPLETED = "completed"
ASSESSMENT_STATUS_PENDING = "pending"
ASSESSMENT_STATUS_IN_PROGRESS = "in_progress"


def _normalize_assessment_status(s: object) -> str:
    """Normalize status to lowercase completed|pending|in_progress for consistent storage and mentor reads."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return ASSESSMENT_STATUS_PENDING
    v = str(s).strip().lower()
    if v in (ASSESSMENT_STATUS_COMPLETED, ASSESSMENT_STATUS_PENDING, ASSESSMENT_STATUS_IN_PROGRESS):
        return v
    if v in ("complete", "done", "finished"):
        return ASSESSMENT_STATUS_COMPLETED
    if v in ("progress", "in progress", "started"):
        return ASSESSMENT_STATUS_IN_PROGRESS
    return ASSESSMENT_STATUS_PENDING


def normalize_assessment_status(s: object) -> str:
    """Public alias for consistent status storage/reads. Returns one of: completed, pending, in_progress."""
    return _normalize_assessment_status(s)


def _normalize_assessments_doc_for_storage(data: dict) -> dict:
    """
    Ensure assessments_doc is stored correctly: normalize plan item statuses and cap list sizes.
    Preserves assessment_history and assessment_recommender.assessment_plan structure; mentor reads status consistently.
    """
    if not data:
        return data
    out = dict(data)
    # Normalize assessment_recommender.assessment_plan item statuses
    ar = out.get("assessment_recommender")
    if isinstance(ar, dict):
        ar = dict(ar)
        plan = ar.get("assessment_plan")
        if isinstance(plan, list):
            plan = [
                {**item, "status": _normalize_assessment_status(item.get("status"))}
                if isinstance(item, dict) else item
                for item in plan
            ]
            if len(plan) > 100:
                plan = plan[-100:]
            ar["assessment_plan"] = plan
        out["assessment_recommender"] = ar
    # Cap assessment_history so doc stays within size limits
    history = out.get("assessment_history")
    if isinstance(history, list) and len(history) > 100:
        out["assessment_history"] = history[-100:]
    return out


def _upsert_assessments_store_doc(doc_id: str, payload: dict, metadata: dict | None = None, tenant_id: str = None):
    """Upsert assessments doc into user_assessments; chunk as uid_assessments_2, _3, ... if over limit.
    
    Args:
        doc_id: Document identifier
        payload: Assessment data
        metadata: Optional additional metadata
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    import json
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    uid = doc_id.rsplit("_", 1)[0] if "_" in doc_id else (metadata or {}).get("uid", "")
    text_bytes = text.encode("utf-8")
    total_size = len(text_bytes)
    
    # Tenant isolation: add tenant_id to metadata if provided
    meta_with_tenant = _add_tenant_to_metadata(metadata or {}, tenant_id)

    if total_size <= CHUNK_PAYLOAD_MAX:
        try:
            normalized_meta = normalize_metadata({**meta_with_tenant, "doc_size": total_size})
            assessments_collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[normalized_meta],
            )
            log.debug(f"chroma.assessments.upsert doc={doc_id} uid={uid} size={total_size}")
            return {"id": doc_id, "doc_size": total_size}
        except Exception as e:
            log.error(f"chroma.assessments.upsert_failed doc={doc_id} uid={uid} err={str(e)}")
            raise

    # Chunk: uid_assessments, uid_assessments_2, ...
    try:
        old_total = 1
        try:
            res = assessments_collection.get(ids=[doc_id], include=["metadatas"])
            if res.get("metadatas") and res["metadatas"]:
                old_total = int(res["metadatas"][0].get("chroma_total_chunks") or 1)
        except Exception:
            pass

        chunks = _split_utf8_safe(text_bytes, CHUNK_PAYLOAD_MAX)
        num_chunks = min(len(chunks), MAX_CHUNKS)
        if len(chunks) > MAX_CHUNKS:
            log.warning(f"chroma.assessments.chunk doc={doc_id} uid={uid} capped to {MAX_CHUNKS} chunks")

        chunk_ids = [_chunk_doc_id(doc_id, i) for i in range(num_chunks)]
        chunk_docs = [chunks[i] for i in range(num_chunks)]
        chunk_metas = [
            normalize_metadata({
                **meta_with_tenant,
                "doc_size": len(chunks[i].encode("utf-8")),
                "chroma_chunk_index": i,
                "chroma_total_chunks": num_chunks,
            })
            for i in range(num_chunks)
        ]
        assessments_collection.upsert(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas)
        log.debug(f"chroma.assessments.upsert doc={doc_id} uid={uid} chunked size={total_size} chunks={num_chunks}")

        if num_chunks < old_total:
            to_delete = [_chunk_doc_id(doc_id, i) for i in range(num_chunks, old_total)]
            try:
                assessments_collection.delete(ids=to_delete)
            except Exception as del_e:
                log.warning(f"chroma.assessments.chunk delete trailing failed: {del_e}")

        return {"id": doc_id, "doc_size": total_size, "chunks": num_chunks}
    except Exception as e:
        log.error(f"chroma.assessments.upsert_failed doc={doc_id} uid={uid} err={str(e)}")
        raise


def _get_assessments_store_doc(doc_id: str) -> dict | None:
    """Load assessments doc from user_assessments; reassemble from chunks if chunked."""
    try:
        res = assessments_collection.get(ids=[doc_id], include=["documents", "metadatas"])
        docs = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        if not docs:
            return None
        meta = metadatas[0] if metadatas else {}
        total_chunks = int(meta.get("chroma_total_chunks") or 1)
        if total_chunks <= 1:
            import json
            return json.loads(docs[0])
        extra_ids = [_chunk_doc_id(doc_id, i) for i in range(1, total_chunks)]
        extra = assessments_collection.get(ids=extra_ids, include=["documents"])
        extra_docs = extra.get("documents") or []
        parts = [docs[0]] + [extra_docs[i] if i < len(extra_docs) else "" for i in range(len(extra_ids))]
        full_text = "".join(parts)
        import json
        return json.loads(full_text)
    except Exception:
        pass
    return None


def upsert_assessments_doc(uid: str, data: dict, metadata: dict = None, tenant_id: str = None):
    """Upsert assessment document for a user.
    
    Args:
        uid: User identifier
        data: Assessment data
        metadata: Optional additional metadata
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"upsert_assessments_doc({uid})")
    
    normalized = _normalize_assessments_doc_for_storage(data)
    return _upsert_assessments_store_doc(_doc_id(uid, "assessments"), normalized, {"uid": uid, "kind": "assessments", **(metadata or {})}, tenant_id)

def get_resume_doc(uid: str) -> dict | None:
    return _get_session_doc(_doc_id(uid, "resume"))

def get_gap_doc(uid: str) -> dict | None:
    return _get_session_doc(_doc_id(uid, "gapanalyze"))


def get_assessments_doc(uid: str) -> dict | None:
    """Fetch assessment data and completion status from the dedicated user_assessments Chroma collection (chatbot uses this)."""
    doc_id = _doc_id(uid, "assessments")
    raw = _get_assessments_store_doc(doc_id)
    if not raw and chat_sessions_collection is not None:
        # One-time migration: read from chat_sessions if previously stored there
        raw = _get_session_doc(doc_id)
        if raw:
            _upsert_assessments_store_doc(doc_id, _normalize_assessments_doc_for_storage(raw), {"uid": uid, "kind": "assessments"})
            log.info(f"Migrated assessments doc for uid={uid} from chat_sessions to user_assessments")
    if not raw:
        return None
    return _normalize_assessments_doc_for_storage(raw)

def get_user_profile(uid: str) -> dict:
    resume = get_resume_doc(uid) or {}
    gap = get_gap_doc(uid) or {}
    assess = get_assessments_doc(uid) or {}
    resume_score_doc = get_resume_score_doc(uid) or {}
    return {
        "structured_resume": resume.get("structured_resume") or resume,
        "career_advisor": gap.get("career_advisor", {}),
        "market_and_course_recommender": gap.get("market_and_course_recommender", {}),
        "resume_score": resume_score_doc if isinstance(resume_score_doc, dict) and resume_score_doc else gap.get("resume_score", {}),
        "assessments": assess,
    }

# Retrieval & reconstruction helpers


def get_session_with_reconstruction(session_id: str, uid: str | None = None) -> dict | None:
    """Fetch main + all chunks and reconstruct JSON dict with backward compatibility for IDs."""
    if chat_sessions_collection is None or session_chunks_collection is None:
        log.debug("ChromaDB unavailable, cannot reconstruct session")
        return None
    
    import json
    try:
        # Resolve main id
        probe_ids = [session_id]
        if uid:
            probe_ids.insert(0, f"{uid}_{session_id}")
        main_doc = None
        main_id_resolved = None
        for pid in probe_ids:
            main_res = chat_sessions_collection.get(ids=[pid])
            if main_res.get("ids"):
                main_doc = (main_res.get("documents") or [None])[0]
                main_id_resolved = pid
                break
        if main_doc is None:
            # Try discovery by metadata
            where = {"session_id": session_id}
            if uid:
                where["uid"] = uid
            resq = chat_sessions_collection.query(query_texts=["session data"], n_results=5, where=where)
            ids = resq.get('ids') or []
            found_id = None
            if ids:
                if isinstance(ids[0], list) and ids[0]:
                    found_id = ids[0][0]
                elif isinstance(ids, list):
                    found_id = ids[0]
            if found_id:
                main_res = chat_sessions_collection.get(ids=[found_id])
                if main_res.get("ids"):
                    main_doc = (main_res.get("documents") or [None])[0]
                    main_id_resolved = found_id
        if main_doc is None:
            return None

        # Fetch chunks by resolved main id, sort by chunk_num
        chunk_res = session_chunks_collection.get(where={"source_id": main_id_resolved})
        chunk_docs = chunk_res.get("documents") or []
        chunk_metas = chunk_res.get("metadatas") or []
        items = []
        for d, m in zip(chunk_docs, chunk_metas):
            try:
                items.append((int(m.get("chunk_num", 0)), d))
            except Exception:
                continue
        items.sort(key=lambda x: x[0])
        full_text = main_doc + "".join([d for _, d in items if d])
        return json.loads(full_text)
    except Exception:
        return None

# Optional: simple search across main and chunks and group by source_id

def search_across_chunks(query_text: str, top_k: int = 5, priority_filter: list[str] | None = None, 
                         tenant_id: str = None, *, allow_cross_tenant: bool = False):
    """Search across main chat sessions and chunks collections.
    
    Args:
        query_text: Text to search for
        top_k: Number of results to return
        priority_filter: Optional filter for priority levels
        tenant_id: Optional tenant identifier for isolation
        allow_cross_tenant: If True, suppress security warning when tenant_id is None
    """
    # Issue 3.2: Security warning for missing tenant isolation
    if not tenant_id and not allow_cross_tenant:
        log.warning(f"SECURITY: search_across_chunks called without tenant_id (query={query_text[:50]}...)")
    
    # Build where clause with tenant isolation
    where_clause = _build_tenant_where_clause(tenant_id)
    if priority_filter:
        priority_condition = {"priority": {"$in": priority_filter}}
        if where_clause:
            where_clause = {"$and": [where_clause, priority_condition]}
        else:
            where_clause = priority_condition
    
    main = chat_sessions_collection.query(
        query_texts=[query_text], 
        n_results=top_k,
        where=_build_tenant_where_clause(tenant_id)
    )
    ch = session_chunks_collection.query(
        query_texts=[query_text], 
        n_results=top_k, 
        where=where_clause
    )
    return {"main": main, "chunks": ch}

# Create job_closed collection (add near other collection initializations)
try:
    job_closed_collection = _get_collection("job_closed")
    # Suppress progress bars during count
    import contextlib
    from io import StringIO
    os.environ['TQDM_DISABLE'] = '1'
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        count = job_closed_collection.count()
    log.info(f"✅ job_closed collection ready (count: {count})")
except Exception as e:
    log.error(f"❌ Failed to get/create job_closed collection: {e}")
    raise

def move_job_to_closed(job_id: str, closed_by: str | None = None, tenant_id: str = None):
    """Move a job to the closed collection.
    
    Args:
        job_id: Job identifier
        closed_by: Optional identifier of who closed the job
        tenant_id: Optional tenant identifier for isolation (backward compatible)
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"move_job_to_closed({job_id})")
    
    log.info(f"📦 Attempting to close job: {job_id}")

    result = job_descriptions_collection.get(ids=[job_id])

    if not result["ids"]:
        log.warning(f"❌ Job {job_id} not found in job_descriptions")
        return False

    original_metadata = result["metadatas"][0] if result["metadatas"] else {}

    closed_metadata = {
        **original_metadata,
        "status": "closed",
        "closed_at": datetime.utcnow().isoformat(),
        "previous_collection": "job_descriptions",
        "closed_by": closed_by
    }
    
    # Tenant isolation: ensure tenant_id is preserved (from original or new)
    if tenant_id:
        closed_metadata["tenant_id"] = tenant_id
    elif "tenant_id" not in closed_metadata:
        log.warning(f"Tenant isolation: move_job_to_closed({job_id}) - no tenant_id in original metadata")

    closed_metadata = normalize_metadata(closed_metadata)

    job_closed_collection.upsert(
        ids=[job_id],
        documents=result["documents"],
        metadatas=[closed_metadata]
    )

    job_descriptions_collection.delete(ids=[job_id])

    log.info(f"✅ Job {job_id} moved to job_closed")
    return True


def move_job_to_active(job_id: str, tenant_id: str = None) -> bool:
    """
    Move job from job_closed back to job_descriptions collection (reopen).
    
    Args:
        job_id: Job ID to reopen
        tenant_id: Optional tenant identifier for isolation (backward compatible)
        
    Returns:
        True if successful, False otherwise
    """
    # Tenant isolation: validate (warns if missing, for backward compatibility)
    _validate_tenant_id(tenant_id, f"move_job_to_active({job_id})")
    
    try:
        # Get job data from job_closed
        results = job_closed_collection.get(ids=[job_id])
        if not results['ids']:
            log.warning(f"⚠️ Job {job_id} not found in job_closed collection")
            return False
        
        job_text = results['documents'][0]
        job_data = json.loads(job_text)
        original_metadata = results['metadatas'][0] if results['metadatas'] else {}
        
        # Prepare metadata for job_descriptions (remove closed-specific fields)
        active_metadata = {
            k: v for k, v in original_metadata.items()
            if k not in ["closed_at", "previous_collection", "closed_by"]
        }

        active_metadata["status"] = "Live"
        active_metadata["reopened_at"] = datetime.utcnow().isoformat()
        
        # Tenant isolation: ensure tenant_id is preserved (from original or new)
        if tenant_id:
            active_metadata["tenant_id"] = tenant_id
        elif "tenant_id" not in active_metadata:
            log.warning(f"Tenant isolation: move_job_to_active({job_id}) - no tenant_id in original metadata")

        active_metadata = normalize_metadata(active_metadata)
        
        # Insert back into job_descriptions collection
        job_descriptions_collection.upsert(
            ids=[job_id],
            documents=[job_text],
            metadatas=[active_metadata]
        )
        log.info(f"✅ Job {job_id} moved back to job_descriptions collection")
        
        # Delete from job_closed collection
        job_closed_collection.delete(ids=[job_id])
        log.info(f"✅ Job {job_id} deleted from job_closed collection")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Error reopening job {job_id}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def batch_move_jobs_to_closed(job_ids: list[str]) -> dict:
    """
    Move multiple jobs to closed collection.
    
    Args:
        job_ids: List of job IDs to close
        
    Returns:
        Dictionary with success/failure counts and details
    """
    results = {
        "total": len(job_ids),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for job_id in job_ids:
        try:
            success = move_job_to_closed(job_id)
            if success:
                results["success"] += 1
                results["details"].append({
                    "job_id": job_id,
                    "status": "success",
                    "action": "closed"
                })
            else:
                results["failed"] += 1
                results["details"].append({
                    "job_id": job_id,
                    "status": "failed",
                    "action": "closed",
                    "reason": "Job not found or move failed"
                })
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "job_id": job_id,
                "status": "failed",
                "action": "closed",
                "reason": str(e)
            })
    
    log.info(f"📊 Batch close completed: {results['success']}/{results['total']} successful")
    return results

def batch_move_jobs_to_active(job_ids: list[str]) -> dict:
    """
    Reopen multiple jobs from closed collection.
    
    Args:
        job_ids: List of job IDs to reopen
        
    Returns:
        Dictionary with success/failure counts and details
    """
    results = {
        "total": len(job_ids),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for job_id in job_ids:
        try:
            success = move_job_to_active(job_id)
            if success:
                results["success"] += 1
                results["details"].append({
                    "job_id": job_id,
                    "status": "success",
                    "action": "reopened"
                })
            else:
                results["failed"] += 1
                results["details"].append({
                    "job_id": job_id,
                    "status": "failed",
                    "action": "reopened",
                    "reason": "Job not found or move failed"
                })
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "job_id": job_id,
                "status": "failed",
                "action": "reopened",
                "reason": str(e)
            })
    
    log.info(f"📊 Batch reopen completed: {results['success']}/{results['total']} successful")
    return results

def get_closed_job(job_id: str):
    """
    Retrieve job data from job_closed collection.
    
    Args:
        job_id: Job ID to retrieve
        
    Returns:
        Job data dictionary or None if not found
    """
    try:
        results = job_closed_collection.get(ids=[job_id])
        if results['ids']:
            job_data = json.loads(results['documents'][0])
            return job_data
        return None
    except Exception as e:
        log.error(f"❌ Error retrieving closed job {job_id}: {e}")
        return None

def list_closed_jobs(limit: int = 50) -> list:
    """
    List all closed jobs.
    
    Args:
        limit: Maximum number of jobs to return
        
    Returns:
        List of closed job IDs with metadata
    """
    try:
        # Peek at closed jobs
        results = job_closed_collection.peek(limit=limit)
        
        closed_jobs = []
        if results and results.get('ids'):
            for i, job_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                closed_jobs.append({
                    "job_id": job_id,
                    "job_title": metadata.get("job_title", ""),
                    "closed_at": metadata.get("closed_at", ""),
                    "company": metadata.get("company", "")
                })
        
        return closed_jobs
    except Exception as e:
        log.error(f"❌ Error listing closed jobs: {e}")
        return []