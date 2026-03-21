import os
import sys
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

# Ensure project root is on sys.path (so we can import chroma, novu_notification_service, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chroma import client as chroma_client, fetch_structured_resume  # type: ignore
from novu_notification_service import NovuClient, Settings, ContextFilter  # type: ignore


# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

NOVU_WORKFLOW_ID = "unlock-your-edge-campaign"

BATCH_SIZE = 100
BATCH_SLEEP_SECONDS = 3
MAX_CONCURRENCY_PER_BATCH = 10

# Default to DRY RUN for safety unless explicitly disabled
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Optional: limit to a small test cohort.
# You can either:
# - set TARGET_EMAILS env var: "a@b.com,c@d.com"
# - or edit this set directly during manual tests.
_env_target_emails = os.getenv("TARGET_EMAILS", "")
TARGET_EMAILS: Set[str] = {
    e.strip().lower() for e in _env_target_emails.split(",") if e.strip()
}

logger = logging.getLogger("unlock_edge_broadcast")


# --------------------------------------------------------------------------------------
# MODELS
# --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    uid: str
    name: str
    email: str
    subscriber_id: str


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------


def is_valid_email(email: Optional[str]) -> bool:
    if not email:
        return False
    email = email.strip()
    if "@" not in email or "." not in email:
        return False
    if " " in email:
        return False
    return True


def extract_name(structured_resume: dict, fallback_name: Optional[str] = None) -> str:
    if not isinstance(structured_resume, dict):
        return fallback_name or "Candidate"

    name = structured_resume.get("Name") or structured_resume.get("name") or fallback_name or ""

    # Some parsed resumes store the name as a list (e.g. ["Anjith Thomas Jose"]).
    # Normalize to a clean string so templates don't see Python list repr like "['...']".
    if isinstance(name, (list, tuple, set)):
        # Prefer the first non-empty element, fall back to joining all.
        seq = [str(part).strip() for part in name if str(part).strip()]
        if seq:
            name = seq[0]
        else:
            name = ""

    name = str(name).strip()
    return name or "Candidate"


def extract_email(structured_resume: dict) -> Optional[str]:
    if not isinstance(structured_resume, dict):
        return None

    contact = (
        structured_resume.get("ContactDetails")
        or structured_resume.get("contact_details")
        or {}
    )
    if not isinstance(contact, dict):
        return None

    email = contact.get("Email") or contact.get("email")
    if not email:
        return None
    email = str(email).strip()
    return email or None


# --------------------------------------------------------------------------------------
# DATA LOADING FROM CHROMA
# --------------------------------------------------------------------------------------


async def load_candidates_from_chroma() -> Tuple[List[Candidate], int]:
    """
    Load candidates from the Chroma 'resume' collection for database prod-jobsify-agent.

    - Uses metadata.uid as the primary id.
    - Pulls full structured_resume via fetch_structured_resume(uid).
    - Extracts candidate_name and email from structured_resume.
    - Uses novu_subscriber_id if present in metadata, otherwise uid.
    - Optionally filters to TARGET_EMAILS for safe testing.
    """
    logger.info("Fetching candidates from Chroma 'resume' collection...")

    resume_collection = chroma_client.get_collection("resume")

    results = resume_collection.get(include=["metadatas"])

    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []

    candidates: List[Candidate] = []
    skipped_invalid_email = 0

    for idx, meta in enumerate(metadatas):
        if idx >= len(ids):
            continue

        uid = str(meta.get("uid") or ids[idx])

        try:
            structured_resume = fetch_structured_resume(uid)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping uid=%s: failed to fetch structured_resume from Chroma: %s",
                uid,
                e,
            )
            continue

        if not structured_resume or not isinstance(structured_resume, dict):
            logger.debug(
                "Skipping uid=%s: structured_resume missing or not a dict", uid
            )
            continue

        email = extract_email(structured_resume)
        if not is_valid_email(email):
            skipped_invalid_email += 1
            logger.debug(
                "Skipping uid=%s: invalid or missing email in structured_resume (%s)",
                uid,
                email,
            )
            continue

        # Optional test cohort filter by email
        if TARGET_EMAILS and email.lower() not in TARGET_EMAILS:
            continue

        name = extract_name(structured_resume, fallback_name=None)
        subscriber_id = (
            meta.get("novu_subscriber_id") or meta.get("subscriber_id") or uid
        )

        candidates.append(
            Candidate(
                uid=uid,
                name=name,
                email=email,  # type: ignore[arg-type]
                subscriber_id=str(subscriber_id),
            )
        )

    logger.info(
        "Loaded %d candidates from Chroma (skipped %d invalid emails)",
        len(candidates),
        skipped_invalid_email,
    )
    if TARGET_EMAILS:
        logger.info(
            "TARGET_EMAILS filter active (%d emails); broadcasting only to this subset.",
            len(TARGET_EMAILS),
        )
    return candidates, skipped_invalid_email


# --------------------------------------------------------------------------------------
# NOVU SENDER
# --------------------------------------------------------------------------------------


async def send_to_candidate(
    novu_client: NovuClient,
    candidate: Candidate,
    semaphore: asyncio.Semaphore,
) -> Tuple[bool, Optional[str]]:
    """
    Trigger the Novu workflow for a single candidate.

    Returns (success, error_message).
    """
    async with semaphore:
        if DRY_RUN:
            logger.info(
                "[DRY_RUN] Would trigger '%s' for %s <%s> (subscriber_id=%s)",
                NOVU_WORKFLOW_ID,
                candidate.name,
                candidate.email,
                candidate.subscriber_id,
            )
            return True, None

        try:
            payload = {"candidate_name": candidate.name}

            await novu_client.trigger_workflow(
                workflow_id=NOVU_WORKFLOW_ID,
                subscriber_id=candidate.subscriber_id,
                email=candidate.email,
                first_name=None,
                last_name=None,
                payload=payload,
                event_id=None,
            )

            logger.info(
                "Triggered '%s' for %s <%s> (subscriber_id=%s)",
                NOVU_WORKFLOW_ID,
                candidate.name,
                candidate.email,
                candidate.subscriber_id,
            )
            return True, None
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to trigger '%s' for uid=%s email=%s: %s",
                NOVU_WORKFLOW_ID,
                candidate.uid,
                candidate.email,
                e,
                exc_info=True,
            )
            return False, str(e)


async def process_batch(
    novu_client: NovuClient,
    batch_candidates: List[Candidate],
) -> Tuple[int, int]:
    """
    Process a single batch with concurrency limit and one retry for failures.

    Returns (successful, failed_after_retry).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY_PER_BATCH)

    # First attempt
    tasks = [
        asyncio.create_task(send_to_candidate(novu_client, c, semaphore))
        for c in batch_candidates
    ]
    results = await asyncio.gather(*tasks)

    failed_candidates: List[Candidate] = []
    for candidate, (ok, _) in zip(batch_candidates, results):
        if not ok:
            failed_candidates.append(candidate)

    # Retry once for failed
    if failed_candidates:
        logger.warning(
            "Retrying %d failed users in this batch for workflow '%s'",
            len(failed_candidates),
            NOVU_WORKFLOW_ID,
        )
        retry_tasks = [
            asyncio.create_task(send_to_candidate(novu_client, c, semaphore))
            for c in failed_candidates
        ]
        retry_results = await asyncio.gather(*retry_tasks)

        still_failed = 0
        for (ok, _), c in zip(retry_results, failed_candidates):
            if not ok:
                still_failed += 1
                logger.error(
                    "User permanently failed after retry: uid=%s email=%s",
                    c.uid,
                    c.email,
                )

        successful = len(batch_candidates) - still_failed
        return successful, still_failed

    # No failures at all
    return len(batch_candidates), 0


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------


async def main() -> None:
    start_time = time.time()

    settings = Settings()
    novu_client = NovuClient(settings.novu_api_key, settings.novu_api_url)
    await novu_client.initialize()
    logger.info(
        "Novu client initialized (workflow_id=%s, DRY_RUN=%s)",
        NOVU_WORKFLOW_ID,
        DRY_RUN,
    )

    try:
        candidates, skipped_invalid_email = await load_candidates_from_chroma()

        total_sent = 0
        total_failed = 0

        if not candidates:
            logger.warning("No candidates found to send broadcast to. Exiting.")
        else:
            for i in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[i : i + BATCH_SIZE]
                batch_index = i // BATCH_SIZE + 1
                logger.info(
                    "Processing batch %d/%d (size=%d)",
                    batch_index,
                    (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE,
                    len(batch),
                )

                sent, failed = await process_batch(novu_client, batch)
                total_sent += sent
                total_failed += failed

                if i + BATCH_SIZE < len(candidates):
                    logger.info(
                        "Sleeping %d seconds before next batch...",
                        BATCH_SLEEP_SECONDS,
                    )
                    await asyncio.sleep(BATCH_SLEEP_SECONDS)

        duration = time.time() - start_time
        logger.info("=== Broadcast Summary (%s) ===", NOVU_WORKFLOW_ID)
        logger.info("DRY_RUN: %s", DRY_RUN)
        logger.info("Total candidates considered: %d", len(candidates))
        logger.info("Total sent (or would send in DRY_RUN): %d", total_sent)
        logger.info("Total failed (after retry): %d", total_failed)
        logger.info("Total skipped (invalid/missing email): %d", skipped_invalid_email)
        logger.info("Duration: %.2f seconds", duration)

    finally:
        try:
            await novu_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())

