#!/usr/bin/env python3
"""
Backfill candidate_domain_primary (and candidate_domains) metadata for all
existing resumes in ChromaDB.

Fetches each resume's full document from chat_sessions via get_resume_doc(),
classifies domain using classify_candidate_domain(), and updates the resume
collection metadata.

Usage:
    cd agents
    python scripts/backfill_candidate_domains.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chroma import collection, get_resume_doc, normalize_metadata
from agents.job_matcher import classify_candidate_domain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


async def backfill_candidate_domains():
    try:
        total = collection.count()
    except Exception as e:
        log.error(f"Failed to get resume collection count: {e}")
        return

    log.info(f"Found {total} resumes in ChromaDB. Starting backfill...")

    if total == 0:
        log.info("No resumes to backfill.")
        return

    batch_size = 100
    updated = 0
    skipped = 0
    errors = 0
    no_doc = 0

    for offset in range(0, total, batch_size):
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )

        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        for i, uid in enumerate(ids):
            try:
                meta = metadatas[i] if i < len(metadatas) else {}

                existing_primary = (meta or {}).get("candidate_domain_primary", "")
                if existing_primary and existing_primary.strip():
                    skipped += 1
                    continue

                doc = get_resume_doc(uid)
                if not doc or not isinstance(doc, dict):
                    no_doc += 1
                    continue

                resume = doc.get("structured_resume") or doc
                if not resume or not isinstance(resume, dict):
                    no_doc += 1
                    continue

                domain = await classify_candidate_domain(resume)
                primary = (domain or "").strip().lower() or "other"
                domains_str = primary

                new_meta = dict(meta or {})
                new_meta["candidate_domain_primary"] = primary
                new_meta["candidate_domains"] = domains_str
                new_meta = normalize_metadata(new_meta)

                collection.update(
                    ids=[uid],
                    metadatas=[new_meta],
                )

                updated += 1
                if updated % 25 == 0:
                    log.info(
                        f"  Progress: {updated} updated, {skipped} skipped, {no_doc} no_doc, {errors} errors"
                    )
                else:
                    log.debug(f"  Updated {uid}: {primary}")

            except Exception as e:
                errors += 1
                log.error(f"  Error updating {uid}: {e}")

    log.info(
        f"Backfill complete: {updated} updated, {skipped} skipped (already had domain), "
        f"{no_doc} no doc, {errors} errors"
    )


if __name__ == "__main__":
    asyncio.run(backfill_candidate_domains())
