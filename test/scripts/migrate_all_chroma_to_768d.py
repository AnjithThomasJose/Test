#!/usr/bin/env python3
"""
Migrate all ChromaDB collections to a target embedding dimension (default 768-d).

Use this when your embedding model has changed (e.g. to BAAI/bge-base-en-v1.5 = 768-d)
and you need every collection to use the new dimension so new inserts succeed.

Run in two phases:
  1. export  - With OLD env (e.g. 384-d model), export every collection to JSON
  2. migrate - With NEW env (e.g. CHROMA_EMBEDDING_MODEL=bge-base, EMBEDDING_DIMENSION=768),
               delete all collections, recreate with new model, re-insert data

Target dimension: set EMBEDDING_DIMENSION=768 (default) or 784 if your model outputs 784-d.
Standard bge-base-en-v1.5 outputs 768-d.

Collections: all that exist in the DB (resume, job_descriptions, job_closed,
courses_knowledge_base, chat_sessions, session_chunks, session_history,
user_assessments, candidate_job_rankings, job_matcher_rankings, etc.).

Usage:
  # Phase 1: Export (keep current/old model in .env)
  python scripts/migrate_all_chroma_to_768d.py export

  # Phase 2: Update .env for new model, then:
  python scripts/migrate_all_chroma_to_768d.py migrate
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 100
EXPORT_DIR = Path(__file__).resolve().parent / "chroma_migration_export"
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_BACKOFF = 2.0

# Target dimension: 768 for bge-base-en-v1.5; override with EMBEDDING_DIMENSION=784 if needed
TARGET_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Collections that use custom export/reinsert (content lives in metadata or needs insert_*)
SPECIAL_COLLECTIONS = {
    "resume",
    "job_descriptions",
    "job_closed",
    "courses_knowledge_base",
}


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "too many requests" in s or "rate limit" in s


def _retry_on_rate_limit(fn, *args, **kwargs):
    for attempt in range(_RATE_LIMIT_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES - 1:
                backoff = _RATE_LIMIT_BACKOFF ** (attempt + 1)
                log.warning("Rate limit hit, retrying in %.1fs (attempt %d/%d)", backoff, attempt + 1, _RATE_LIMIT_RETRIES)
                time.sleep(backoff)
            else:
                raise


def _list_collection_names():
    """Return names of all collections in the current Chroma DB."""
    from chroma import client
    try:
        colls = client.list_collections()
        return [c.name for c in colls]
    except Exception as e:
        log.warning("Could not list collections: %s", e)
        return []


# ---------------------------------------------------------------------------
# Phase 1: Export
# ---------------------------------------------------------------------------


def export_resume_collection():
    from chroma import _get_collection, parse_resume_from_metadata
    coll = _get_collection("resume")
    all_data = []
    offset = 0
    total = coll.count()
    log.info("Exporting %d resumes...", total)
    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(limit=BATCH_SIZE, offset=o, include=["metadatas"])
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        metas = batch.get("metadatas") or []
        for i, rid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            resume_data = parse_resume_from_metadata(meta)
            if resume_data:
                all_data.append({"id": rid, "data": resume_data})
            else:
                log.warning("Could not parse resume for id=%s", rid)
        offset += len(ids)
        log.info("Exported %d/%d resumes", min(offset, total), total)
    return all_data


def export_job_descriptions():
    from chroma import _get_collection
    coll = _get_collection("job_descriptions")
    all_data = []
    offset = 0
    total = coll.count()
    log.info("Exporting %d job descriptions...", total)
    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"])
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, jid in enumerate(ids):
            doc = docs[i] if i < len(docs) else None
            if doc:
                try:
                    jd_data = json.loads(doc)
                    meta = metas[i] if i < len(metas) else {}
                    all_data.append({"id": jid, "data": jd_data, "metadata": meta or {}})
                except json.JSONDecodeError as e:
                    log.warning("Invalid JSON for job_id=%s: %s", jid, e)
        offset += len(ids)
        log.info("Exported %d/%d job descriptions", min(offset, total), total)
    return all_data


def export_job_closed():
    from chroma import _get_collection
    coll = _get_collection("job_closed")
    all_data = []
    offset = 0
    total = coll.count()
    log.info("Exporting %d closed jobs...", total)
    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"])
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, jid in enumerate(ids):
            doc = docs[i] if i < len(docs) else None
            if doc:
                try:
                    jd_data = json.loads(doc)
                    meta = metas[i] if i < len(metas) else {}
                    all_data.append({"id": jid, "data": jd_data, "metadata": meta or {}})
                except json.JSONDecodeError as e:
                    log.warning("Invalid JSON for closed job_id=%s: %s", jid, e)
        offset += len(ids)
        log.info("Exported %d/%d closed jobs", min(offset, total), total)
    return all_data


def export_courses_knowledge_base():
    from chroma import _get_collection
    coll = _get_collection("courses_knowledge_base")
    all_data = []
    offset = 0
    total = coll.count()
    log.info("Exporting %d courses...", total)
    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"])
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, cid in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            all_data.append({"id": cid, "document": doc or "", "metadata": meta or {}})
        offset += len(ids)
        log.info("Exported %d/%d courses", min(offset, total), total)
    return all_data


def export_generic_collection(name: str):
    """Export a collection as ids, documents, metadatas (for re-embed on reinsert)."""
    from chroma import _get_collection, normalize_metadata
    coll = _get_collection(name)
    all_data = []
    offset = 0
    total = coll.count()
    log.info("Exporting %d items from %s...", total, name)
    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"])
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, id_ in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            all_data.append({
                "id": id_,
                "document": doc if doc is not None else "",
                "metadata": normalize_metadata(meta or {}),
            })
        offset += len(ids)
        log.info("Exported %d/%d from %s", min(offset, total), total, name)
    return all_data


def export_and_save():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = os.getenv("CHROMA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    log.info("=== Phase 1: Export (run with current/old embedding model) ===")
    log.info("CHROMA_EMBEDDING_MODEL: %s", model)

    names = _list_collection_names()
    if not names:
        log.warning("No collections found; nothing to export.")
        return

    for name in names:
        try:
            if name == "resume":
                data = export_resume_collection()
                with open(EXPORT_DIR / "resume.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=0)
            elif name == "job_descriptions":
                data = export_job_descriptions()
                with open(EXPORT_DIR / "job_descriptions.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=0)
            elif name == "job_closed":
                data = export_job_closed()
                with open(EXPORT_DIR / "job_closed.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=0)
            elif name == "courses_knowledge_base":
                data = export_courses_knowledge_base()
                with open(EXPORT_DIR / "courses_knowledge_base.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=0)
            else:
                data = export_generic_collection(name)
                safe_name = name.replace("/", "_").replace("\\", "_")
                with open(EXPORT_DIR / f"{safe_name}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=0)
            log.info("Exported %s: %d items", name, len(data))
        except Exception as e:
            log.error("Export failed for %s: %s", name, e)
            raise

    log.info("Export saved to %s", EXPORT_DIR)
    log.info("Next: set CHROMA_EMBEDDING_MODEL and EMBEDDING_DIMENSION=%s, then run: migrate", TARGET_DIMENSION)


# ---------------------------------------------------------------------------
# Phase 2: Delete + Reinsert
# ---------------------------------------------------------------------------


def delete_all_collections():
    from chroma import client
    names = _list_collection_names()
    for name in names:
        try:
            client.delete_collection(name=name)
            log.info("Deleted collection: %s", name)
        except Exception as e:
            log.warning("Could not delete %s: %s", name, e)


def _refresh_chroma_collection_refs():
    """Recreate collection references so new collections use current (target-dim) embedding model."""
    import chroma
    chroma.collection = chroma._get_collection("resume")
    chroma.chat_sessions_collection = chroma._get_collection("chat_sessions")
    chroma.job_descriptions_collection = chroma._get_collection("job_descriptions")
    chroma.session_chunks_collection = chroma._get_collection("session_chunks")
    chroma.session_history_collection = chroma._get_collection("session_history")
    chroma.assessments_collection = chroma._get_collection("user_assessments")
    chroma.candidate_job_rankings_collection = chroma._get_collection("candidate_job_rankings")
    chroma.job_matcher_rankings_collection = chroma._get_collection("job_matcher_rankings")
    chroma.job_closed_collection = chroma._get_collection("job_closed")
    log.info("Refreshed collection references (new %s-d collections)", TARGET_DIMENSION)


def reinsert_resumes(resumes: list):
    from chroma import insert_resume
    for i, item in enumerate(resumes):
        try:
            insert_resume(item["id"], item["data"], metadata={"uid": item["id"]})
            if (i + 1) % 50 == 0:
                log.info("Re-inserted %d/%d resumes", i + 1, len(resumes))
        except Exception as e:
            log.error("Failed to insert resume %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d resumes", len(resumes))


def reinsert_job_descriptions(jds: list):
    from chroma import insert_job_description
    for i, item in enumerate(jds):
        try:
            insert_job_description(item["id"], item["data"], metadata=item.get("metadata", {}))
            if (i + 1) % 50 == 0:
                log.info("Re-inserted %d/%d job descriptions", i + 1, len(jds))
        except Exception as e:
            log.error("Failed to insert JD %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d job descriptions", len(jds))


def reinsert_job_closed(closed: list):
    from chroma import job_closed_collection, normalize_metadata
    for i, item in enumerate(closed):
        try:
            doc_str = json.dumps(item["data"], separators=(",", ":"), ensure_ascii=False)
            meta = normalize_metadata(item.get("metadata", {}))
            if not meta:
                meta = {"_placeholder": "1"}  # ChromaDB requires non-empty metadata
            job_closed_collection.upsert(ids=[item["id"]], documents=[doc_str], metadatas=[meta])
            if (i + 1) % 50 == 0:
                log.info("Re-inserted %d/%d closed jobs", i + 1, len(closed))
        except Exception as e:
            log.error("Failed to insert closed job %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d closed jobs", len(closed))


def reinsert_courses_knowledge_base(courses: list):
    from chroma import _get_collection, normalize_metadata
    coll = _get_collection("courses_knowledge_base")
    for i, item in enumerate(courses):
        try:
            meta = normalize_metadata(item.get("metadata", {}))
            if not meta:
                meta = {"_placeholder": "1"}  # ChromaDB requires non-empty metadata
            coll.upsert(ids=[item["id"]], documents=[item["document"]], metadatas=[meta])
            if (i + 1) % 100 == 0:
                log.info("Re-inserted %d/%d courses", i + 1, len(courses))
        except Exception as e:
            log.error("Failed to insert course %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d courses", len(courses))


def reinsert_generic_collection(name: str, data: list):
    from chroma import _get_collection, normalize_metadata
    if not data:
        return
    coll = _get_collection(name)
    ids = [x["id"] for x in data]
    documents = [x.get("document", "") or "" for x in data]
    metadatas = []
    for x in data:
        meta = normalize_metadata(x.get("metadata") or {})
        if not meta:
            meta = {"_placeholder": "1"}  # ChromaDB requires non-empty metadata
        metadatas.append(meta)
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]
        coll.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        log.info("Re-inserted %d/%d into %s", min(end, len(ids)), len(ids), name)
    log.info("Re-inserted %d items into %s", len(ids), name)


def delete_and_reinsert():
    if not EXPORT_DIR.exists():
        log.error("Export directory not found: %s. Run 'export' first.", EXPORT_DIR)
        sys.exit(1)

    dim = os.getenv("EMBEDDING_DIMENSION", "")
    model = os.getenv("CHROMA_EMBEDDING_MODEL", "")
    if dim and int(dim) != TARGET_DIMENSION:
        log.warning("EMBEDDING_DIMENSION=%s; script target is %s", dim, TARGET_DIMENSION)
    log.info("=== Phase 2: Migrate to %s-d (run with new embedding model) ===", TARGET_DIMENSION)
    log.info("CHROMA_EMBEDDING_MODEL: %s", model or "(not set)")

    # Load all exported files
    exported = {}
    for f in EXPORT_DIR.glob("*.json"):
        name = f.stem
        with open(f, encoding="utf-8") as fp:
            exported[name] = json.load(fp)
        log.info("Loaded %s: %d items", name, len(exported[name]))

    if not exported:
        log.error("No exported JSON files in %s", EXPORT_DIR)
        sys.exit(1)

    log.info("Deleting all collections...")
    delete_all_collections()

    _refresh_chroma_collection_refs()

    log.info("Re-inserting data...")
    if "resume" in exported and exported["resume"]:
        reinsert_resumes(exported["resume"])
    if "job_descriptions" in exported and exported["job_descriptions"]:
        reinsert_job_descriptions(exported["job_descriptions"])
    if "job_closed" in exported and exported["job_closed"]:
        reinsert_job_closed(exported["job_closed"])
    if "courses_knowledge_base" in exported and exported["courses_knowledge_base"]:
        reinsert_courses_knowledge_base(exported["courses_knowledge_base"])

    for name, data in exported.items():
        if name in SPECIAL_COLLECTIONS or not data:
            continue
        reinsert_generic_collection(name, data)

    log.info("Migration complete. All collections are now %s-d.", TARGET_DIMENSION)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands: export | migrate")
        print("Target dimension: EMBEDDING_DIMENSION (default 768)")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "export":
        export_and_save()
    elif cmd == "migrate":
        delete_and_reinsert()
    else:
        log.error("Unknown command: %s. Use 'export' or 'migrate'.", cmd)
        sys.exit(1)


if __name__ == "__main__":
    main()
