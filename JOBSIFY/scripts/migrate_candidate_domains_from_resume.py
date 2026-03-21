#!/usr/bin/env python3
"""
Migrate candidate_domains to a top-level metadata field on the resume collection
— zero LLM/embedding cost.

structured_resume_json in Chroma is trimmed to 8000 bytes, so candidate_domains
(near the end of the JSON) is often truncated. This script:
1. Tries structured_resume_json in resume metadata first
2. Falls back to get_resume_doc (chat_sessions) for the full resume when truncated

Usage:
    cd agents
    python scripts/migrate_candidate_domains_from_resume.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chroma import collection, parse_resume_from_metadata, normalize_metadata, get_resume_doc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def _extract_domains_str(resume: dict) -> str | None:
    """Extract candidate_domains as comma-separated string from resume dict."""
    domains = resume.get("candidate_domains") or resume.get("candidateDomains") or []
    if not domains or not isinstance(domains, list):
        return None
    s = ",".join(str(d).strip().lower() for d in domains[:3] if d and str(d).strip())
    return s if s else None


def migrate_candidate_domains(batch_size: int = 100, dry_run: bool = False):
    try:
        total = collection.count()
    except Exception as e:
        log.error(f"Failed to get resume collection count: {e}")
        return

    log.info(f"Found {total} resumes in ChromaDB. Starting domain metadata migration...")

    if total == 0:
        log.info("No resumes to migrate.")
        return

    updated = 0
    skipped_already = 0
    skipped_no_domains = 0
    skipped_no_resume = 0
    from_chat_sessions = 0
    errors = 0

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
                if not meta or not isinstance(meta, dict):
                    skipped_no_resume += 1
                    continue

                existing = (meta.get("candidate_domains") or "").strip()
                if existing:
                    skipped_already += 1
                    continue

                domains_str = None
                used_chat_sessions = False
                parsed = parse_resume_from_metadata(meta)
                if parsed and isinstance(parsed, dict):
                    domains_str = _extract_domains_str(parsed)

                if not domains_str:
                    doc = get_resume_doc(uid)
                    if doc:
                        resume = doc.get("structured_resume") or doc
                        if resume and isinstance(resume, dict):
                            domains_str = _extract_domains_str(resume)
                            if domains_str:
                                used_chat_sessions = True
                                from_chat_sessions += 1

                if not domains_str:
                    skipped_no_domains += 1
                    continue

                if dry_run:
                    src = "chat_sessions" if used_chat_sessions else "metadata"
                    log.info(f"  [DRY RUN] {uid}: would set candidate_domains={domains_str} (from {src})")
                    updated += 1
                    continue

                new_meta = dict(meta)
                new_meta["candidate_domains"] = domains_str
                new_meta = normalize_metadata(new_meta)

                collection.update(
                    ids=[uid],
                    metadatas=[new_meta],
                )

                updated += 1
                if updated % 50 == 0:
                    log.info(
                        f"  Progress: {updated} updated, {skipped_already} already had, "
                        f"{skipped_no_domains} no domains, {from_chat_sessions} from chat_sessions, "
                        f"{errors} errors"
                    )

            except Exception as e:
                errors += 1
                log.error(f"  Error migrating {uid}: {e}")

    log.info(
        f"Migration complete: {updated} updated, {skipped_already} already had domains, "
        f"{skipped_no_domains} no domains anywhere, {from_chat_sessions} from chat_sessions, "
        f"{errors} errors"
    )


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("Running in DRY RUN mode — no writes will be made")
    migrate_candidate_domains(dry_run=dry)
