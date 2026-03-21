import logging
from typing import Optional

from chroma import _get_collection, reindex_resume_for_ranker


log = logging.getLogger(__name__)


def reindex_all_resumes_for_ranker(
    batch_size: int = 100,
    max_resumes: Optional[int] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Reindex all resumes in the ChromaDB `resume` collection so that ranker can
    resolve structured resume JSON from metadata (`structured_resume_json`).

    This is safe to run multiple times; resumes that already have structured
    metadata will simply be upserted again.
    """
    collection = _get_collection("resume")
    if collection is None:
        raise RuntimeError("Could not resolve ChromaDB 'resume' collection")

    total = collection.count()
    if max_resumes is not None:
        total = min(total, max_resumes)

    log.info(
        "Starting resume reindex for ranker (total=%s, batch_size=%s, max_resumes=%s)",
        total,
        batch_size,
        max_resumes,
    )

    processed = 0
    offset = 0

    while offset < total:
        remaining = total - offset
        current_batch_size = min(batch_size, remaining)

        # Fetch a window of ids from the resume collection.
        batch = collection.get(
            limit=current_batch_size,
            offset=offset,
        )

        ids = batch.get("ids") or []
        if not ids:
            break

        for resume_id in ids:
            ok = reindex_resume_for_ranker(resume_id, tenant_id=tenant_id)
            processed += 1
            if not ok:
                log.warning("Failed to reindex resume_id=%s", resume_id)

        log.info(
            "Reindexed %s/%s resumes for ranker (last_offset=%s)",
            processed,
            total,
            offset,
        )

        offset += current_batch_size

    log.info("Completed resume reindex for ranker. Total processed=%s", processed)


def main() -> int:
    # Run with defaults; customize via direct function call if needed.
    reindex_all_resumes_for_ranker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

