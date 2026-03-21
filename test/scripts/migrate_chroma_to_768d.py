#!/usr/bin/env python3
"""
Migrate ChromaDB collections from 384-d to 768-d embeddings.

Run in two phases:
  1. export  - With OLD env (BAAI/bge-small-en-v1.5), export data to JSON files
  2. migrate - With NEW env (BAAI/bge-base-en-v1.5, EMBEDDING_DIMENSION=768),
               delete collections, recreate with new model, re-insert data

Collections migrated: resume, job_descriptions, job_closed, courses_knowledge_base
Cache collections (candidate_job_rankings, job_matcher_rankings) are deleted
and will repopulate as users use the app.

The script CREATES new collections automatically—no need to pre-create them.
After delete, get_or_create_collection creates fresh 768-d collections.

Usage:
  cd agents
  # Phase 1: Export (keep old model in .env)
  python scripts/migrate_chroma_to_768d.py export

  # Phase 2: Update .env:
  #   CHROMA_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
  #   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
  #   EMBEDDING_DIMENSION=768

  # Phase 3: Delete + reinsert
  python scripts/migrate_chroma_to_768d.py migrate
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure agents/ is on path when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ChromaDB Cloud: use smaller batch to avoid rate limits
BATCH_SIZE = 100
EXPORT_DIR = Path(__file__).resolve().parent / "chroma_migration_export"
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_BACKOFF = 2.0


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
                log.warning(
                    "Rate limit hit, retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt + 1,
                    _RATE_LIMIT_RETRIES,
                )
                time.sleep(backoff)
            else:
                raise


# ---------------------------------------------------------------------------
# Phase 1: Export
# ---------------------------------------------------------------------------


def export_resume_collection():
    """Export all resumes from metadata (structured_resume_json)."""
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
    """Export all job descriptions (documents + metadata)."""
    from chroma import _get_collection

    coll = _get_collection("job_descriptions")
    all_data = []
    offset = 0

    total = coll.count()
    log.info("Exporting %d job descriptions...", total)

    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(
                limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"]
            )
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
    """Export all closed jobs (documents + metadata)."""
    from chroma import _get_collection

    coll = _get_collection("job_closed")
    all_data = []
    offset = 0

    total = coll.count()
    log.info("Exporting %d closed jobs...", total)

    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(
                limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"]
            )
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
    """Export all courses from courses_knowledge_base (documents + metadata)."""
    from chroma import _get_collection

    coll = _get_collection("courses_knowledge_base")
    all_data = []
    offset = 0

    total = coll.count()
    log.info("Exporting %d courses from knowledge base...", total)

    while offset < total:
        batch = _retry_on_rate_limit(
            lambda o=offset: coll.get(
                limit=BATCH_SIZE, offset=o, include=["documents", "metadatas"]
            )
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


def export_and_save():
    """Export all data and save to JSON files."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 1: Export (run with OLD 384-d model) ===")
    model = os.getenv("CHROMA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    log.info("Current CHROMA_EMBEDDING_MODEL: %s", model)

    resumes = export_resume_collection()
    jds = export_job_descriptions()
    closed = export_job_closed()
    courses = export_courses_knowledge_base()

    with open(EXPORT_DIR / "resumes.json", "w", encoding="utf-8") as f:
        json.dump(resumes, f, ensure_ascii=False, indent=0)

    with open(EXPORT_DIR / "job_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(jds, f, ensure_ascii=False, indent=0)

    with open(EXPORT_DIR / "job_closed.json", "w", encoding="utf-8") as f:
        json.dump(closed, f, ensure_ascii=False, indent=0)

    with open(EXPORT_DIR / "courses_knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=0)

    log.info("Exported to %s:", EXPORT_DIR)
    log.info("  - resumes.json: %d items", len(resumes))
    log.info("  - job_descriptions.json: %d items", len(jds))
    log.info("  - job_closed.json: %d items", len(closed))
    log.info("  - courses_knowledge_base.json: %d items", len(courses))
    log.info("Next: Update .env to BAAI/bge-base-en-v1.5 and EMBEDDING_DIMENSION=768")
    log.info("Then run: python scripts/migrate_chroma_to_768d.py migrate")


# ---------------------------------------------------------------------------
# Phase 2: Delete + Reinsert
# ---------------------------------------------------------------------------


def delete_collections():
    """Delete collections so they can be recreated with new dimension."""
    from chroma import client

    names = [
        "resume",
        "job_descriptions",
        "job_closed",
        "courses_knowledge_base",
        "candidate_job_rankings",
        "job_matcher_rankings",
    ]
    for name in names:
        try:
            client.delete_collection(name=name)
            log.info("Deleted collection: %s", name)
        except Exception as e:
            log.warning("Could not delete %s: %s", name, e)


def _refresh_chroma_collections():
    """
    Refresh chroma module's collection references after delete.
    Chroma uses get_or_create_collection, so this creates NEW collections
    with the current embedding model (768-d).
    """
    import chroma

    chroma.collection = chroma._get_collection("resume")
    chroma.job_descriptions_collection = chroma._get_collection("job_descriptions")
    chroma.job_closed_collection = chroma._get_collection("job_closed")
    log.info("Refreshed collection references (new 768-d collections created)")


def reinsert_resumes(resumes: list):
    """Re-insert resumes with new 768-d embeddings."""
    from chroma import insert_resume

    for i, item in enumerate(resumes):
        try:
            insert_resume(
                item["id"],
                item["data"],
                metadata={"uid": item["id"]},
            )
            if (i + 1) % 50 == 0:
                log.info("Re-inserted %d/%d resumes", i + 1, len(resumes))
        except Exception as e:
            log.error("Failed to insert resume %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d resumes", len(resumes))


def reinsert_job_descriptions(jds: list):
    """Re-insert job descriptions with new 768-d embeddings."""
    from chroma import insert_job_description

    for i, item in enumerate(jds):
        try:
            insert_job_description(
                item["id"],
                item["data"],
                metadata=item.get("metadata", {}),
            )
            if (i + 1) % 50 == 0:
                log.info("Re-inserted %d/%d job descriptions", i + 1, len(jds))
        except Exception as e:
            log.error("Failed to insert JD %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d job descriptions", len(jds))


def reinsert_job_closed(closed: list):
    """Re-insert closed jobs directly into job_closed collection."""
    from chroma import job_closed_collection, normalize_metadata

    for i, item in enumerate(closed):
        try:
            doc_str = json.dumps(item["data"], separators=(",", ":"), ensure_ascii=False)
            meta = normalize_metadata(item.get("metadata", {}))
            job_closed_collection.upsert(
                ids=[item["id"]],
                documents=[doc_str],
                metadatas=[meta],
            )
            if (i + 1) % 50 == 0:
                log.info("Re-inserted %d/%d closed jobs", i + 1, len(closed))
        except Exception as e:
            log.error("Failed to insert closed job %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d closed jobs", len(closed))


def reinsert_courses_knowledge_base(courses: list):
    """Re-insert courses with new 768-d embeddings."""
    from chroma import _get_collection, normalize_metadata

    coll = _get_collection("courses_knowledge_base")
    for i, item in enumerate(courses):
        try:
            meta = normalize_metadata(item.get("metadata", {}))
            coll.upsert(
                ids=[item["id"]],
                documents=[item["document"]],
                metadatas=[meta],
            )
            if (i + 1) % 100 == 0:
                log.info("Re-inserted %d/%d courses", i + 1, len(courses))
        except Exception as e:
            log.error("Failed to insert course %s: %s", item["id"], e)
            raise
    log.info("Re-inserted %d courses", len(courses))


def delete_and_reinsert():
    """Delete collections and re-insert from exported JSON files."""
    if not EXPORT_DIR.exists():
        log.error("Export directory not found: %s", EXPORT_DIR)
        log.error("Run 'python scripts/migrate_chroma_to_768d.py export' first")
        sys.exit(1)

    model = os.getenv("CHROMA_EMBEDDING_MODEL", "")
    dim = os.getenv("EMBEDDING_DIMENSION", "")
    if "bge-base" not in model or dim != "768":
        log.warning(
            "Expected CHROMA_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 and EMBEDDING_DIMENSION=768"
        )
        log.warning("Current: CHROMA_EMBEDDING_MODEL=%s, EMBEDDING_DIMENSION=%s", model, dim)

    log.info("=== Phase 2: Migrate (run with NEW 768-d model) ===")

    with open(EXPORT_DIR / "resumes.json", encoding="utf-8") as f:
        resumes = json.load(f)
    with open(EXPORT_DIR / "job_descriptions.json", encoding="utf-8") as f:
        jds = json.load(f)
    with open(EXPORT_DIR / "job_closed.json", encoding="utf-8") as f:
        closed = json.load(f)
    with open(EXPORT_DIR / "courses_knowledge_base.json", encoding="utf-8") as f:
        courses = json.load(f)

    log.info("Deleting collections...")
    delete_collections()

    # Refresh chroma's collection refs so get_or_create makes NEW 768-d collections
    _refresh_chroma_collections()

    log.info("Re-inserting data...")
    reinsert_job_descriptions(jds)
    reinsert_resumes(resumes)
    reinsert_job_closed(closed)
    reinsert_courses_knowledge_base(courses)

    log.info("Migration complete.")
    log.info(
        "Cache collections (candidate_job_rankings, job_matcher_rankings) "
        "will repopulate as users run job_matcher."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands: export | migrate")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "export":
        export_and_save()
    elif cmd == "migrate":
        delete_and_reinsert()
    else:
        log.error("Unknown command: %s. Use 'export' or 'migrate'", cmd)
        sys.exit(1)


if __name__ == "__main__":
    main()
