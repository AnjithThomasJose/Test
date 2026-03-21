#!/usr/bin/env python3
"""
Backfill job_domain_primary metadata for all existing jobs in ChromaDB.

Fetches all job documents, classifies their domains using classify_jd_domains(),
and updates their metadata with the new job_domain_primary field.

Usage:
    cd agents
    python scripts/backfill_job_domains.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chroma import job_descriptions_collection, normalize_metadata
from agents.job_matcher import classify_jd_domains

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def backfill_job_domains():
    total = job_descriptions_collection.count()
    log.info(f"Found {total} jobs in ChromaDB. Starting backfill...")

    if total == 0:
        log.info("No jobs to backfill.")
        return

    batch_size = 100
    updated = 0
    skipped = 0
    errors = 0

    for offset in range(0, total, batch_size):
        results = job_descriptions_collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for i, job_id in enumerate(ids):
            try:
                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else ""

                existing_primary = (meta or {}).get("job_domain_primary", "")
                if existing_primary and existing_primary != "":
                    skipped += 1
                    continue

                try:
                    jd_data = json.loads(doc) if doc else {}
                except (json.JSONDecodeError, TypeError):
                    jd_data = {}

                domains = classify_jd_domains(jd_data)
                primary = domains[0].lower().strip() if domains else ""
                domains_str = ",".join(domains[:3])

                new_meta = dict(meta or {})
                new_meta["job_domain_primary"] = primary
                new_meta["job_domains"] = domains_str
                new_meta = normalize_metadata(new_meta)

                job_descriptions_collection.update(
                    ids=[job_id],
                    metadatas=[new_meta]
                )

                updated += 1
                title = (meta or {}).get("job_title", "N/A")
                if updated % 25 == 0:
                    log.info(f"  Progress: {updated} updated, {skipped} skipped, {errors} errors")
                else:
                    log.debug(f"  Updated {job_id} ({title}): {primary}")

            except Exception as e:
                errors += 1
                log.error(f"  Error updating {job_id}: {e}")

    log.info(f"Backfill complete: {updated} updated, {skipped} skipped (already had domain), {errors} errors")


if __name__ == "__main__":
    backfill_job_domains()
