# ChromaDB Storage Classification

This document classifies all ChromaDB collections by what is stored as **embeddings**, **documents**, and **metadata**.

---

## Important: How Embeddings Work

**Embeddings are not field-level.** Chroma stores a single vector (embedding) per record, computed from the **entire document text**. The embedding model (e.g. `BAAI/bge-small-en-v1.5`) converts the full input string into a dense vector. Individual fields are not stored as separate embeddings.

---

## Classification Summary

| Category | Collections | Vector Search Used? |
|----------|-------------|---------------------|
| **Semantic search** | `resume`, `job_descriptions`, `job_closed` | Yes |
| **Document storage** | `candidate_job_rankings`, `job_matcher_rankings`, `chat_sessions`, `session_chunks`, `session_history`, `user_assessments` | No (embeddings computed but not queried) |

---

## 1. Collections Used for Semantic (Vector) Search

### `resume`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Entire `resume_data` JSON serialized to string (compact JSON). Includes: `Name`, `Skills`, `Education`, `WorkExperience`, `ContactDetails`, `projects`, `certifications`, etc. — **all fields combined** as one vector. |
| **Document** | Placeholder only (`RESUME_DOC_PLACEHOLDER`) — not the actual resume. |
| **Metadata** | `uid`, `name`, `location`, `skills_count`, `education_count`, `total_experience_years`, `candidate_domains`, `doc_size`, `embedding_model`, `content_hash`, `structured_resume_json` (full resume JSON), `created_at`, `updated_at`, `tenant_id` |

**Embedding input:** `json.dumps(resume_data, separators=(",", ":"), ensure_ascii=False)` — the complete structured resume.

---

### `job_descriptions`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Entire `jd_data` JSON serialized to string. Chroma auto-computes embedding from the document. Includes: `jobTitle`, `location`, `experience`, `educationRequired`, `jobType`, `company`, `requiredSkills`, `jobDescription`, `jobDomains`, etc. — **all fields combined** as one vector. |
| **Document** | Full JD JSON text (`json.dumps(jd_data, indent=2)`). |
| **Metadata** | `job_id`, `job_title`, `location`, `experience`, `education_required`, `job_type`, `company`, `skills_count`, `job_domains`, `job_domain_primary`, `file_url`, `embedding_model`, `tenant_id` |

**Embedding input:** Same as document — full job description JSON.

---

### `job_closed`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Same as `job_descriptions` — full JD JSON. Chroma auto-computes from document. |
| **Document** | Same JD text as `job_descriptions` (moved when job is closed). |
| **Metadata** | Same as `job_descriptions` plus: `status`, `closed_at`, `previous_collection`, `closed_by`, `tenant_id` |

**Embedding input:** Same as document — full job description JSON.

---

## 2. Collections Used for Document Storage (Embeddings Auto-Computed, Not Queried)

These collections pass `documents` only; Chroma computes embeddings internally, but the app does **not** use vector search on them. They are keyed by ID and retrieved by `get(ids=[...])`.

### `candidate_job_rankings`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Entire document text (match result JSON). Chroma auto-computes. |
| **Document** | Match result JSON: `match_score`, `similar_jobs`, `job_id`, `uid`, `confidence_score`, etc. Main doc + optional `similar_jobs` chunks. |
| **Metadata** | `uid`, `job_id`, `stored_at`, `resume_hash` |

**Embedding input:** `main_payload` (match result) + `similar_jobs` JSON — entire stored payload.

---

### `job_matcher_rankings`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Entire document text (match result JSON). Chroma auto-computes. |
| **Document** | Match result JSON: `top_matches`, `matched_jobs`, `total_matches_found`, `job_matcher_status`, `confidence_score`, `processing_time_seconds`, `total_jobs_matched`, `semantic_filter_threshold`, `processing_method`, `batches_processed`, `message` |
| **Metadata** | `uid`, `stored_at` |

**Embedding input:** Full payload from `_JOB_MATCHER_CACHE_KEYS` — entire stored payload.

---

### `chat_sessions`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Entire session JSON. Chroma auto-computes. |
| **Document** | Session JSON: `uid`, `timestamp`, `status`, `structured_resume`, agent outputs (`personal_info_parser`, `education_parser`, `experience_parser`, `skills_parser`, `resume_assembler`, etc.), `resume_url`, etc. |
| **Metadata** | `session_id`, `uid`, `timestamp`, `resume_url`, `status`, `data_hash`, `doc_size`, `num_chunks`, `tenant_id` |

**Embedding input:** Full session JSON — entire session data.

---

### `session_chunks`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Chunked session text. Chroma auto-computes. |
| **Document** | Chunked session JSON (continuation of main session when over limit). |
| **Metadata** | `source_id`, `chunk_num`, `num_chunks`, `priority`, `session_id`, `uid`, `timestamp`, etc. |

**Embedding input:** Chunk text — same as document.

---

### `session_history`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Previous session document. Chroma auto-computes. |
| **Document** | Previous session doc (compressed JSON). |
| **Metadata** | History metadata (session_id, uid, timestamp, etc.) |

**Embedding input:** Same as document.

---

### `user_assessments`

| Storage Type | Content |
|--------------|---------|
| **Embedding** | Entire assessment JSON. Chroma auto-computes. |
| **Document** | Assessment JSON: plans, answers, reports, generated questions, etc. Chunked when over size limit. |
| **Metadata** | `uid`, `doc_size`, `chroma_chunk_index`, `chroma_total_chunks`, `tenant_id` |

**Embedding input:** Full assessment payload — same as document.

---

## 3. Fields Embedded (What Gets Turned Into Vectors)

Embeddings are computed from the **entire document string**, not per field. The following are the **source inputs** that get embedded:

| Collection | Embedding source (full text) |
|------------|-----------------------------|
| **resume** | Entire `resume_data` JSON: `Name`, `Skills`, `Education`, `WorkExperience`, `ContactDetails`, `projects`, `certifications`, `summary`, `location`, etc. |
| **job_descriptions** | Entire `jd_data` JSON: `jobTitle`, `location`, `experience`, `educationRequired`, `jobType`, `company`, `requiredSkills`, `jobDescription`, `jobDomains`, etc. |
| **job_closed** | Same as `job_descriptions` |
| **candidate_job_rankings** | Match result JSON: `match_score`, `similar_jobs`, `job_id`, `confidence_score`, etc. |
| **job_matcher_rankings** | Match result JSON: `top_matches`, `matched_jobs`, `total_matches_found`, `confidence_score`, etc. |
| **chat_sessions** | Session JSON: `uid`, `timestamp`, `status`, `structured_resume`, all agent outputs |
| **session_chunks** | Chunked session text |
| **session_history** | Previous session doc |
| **user_assessments** | Assessment JSON: plans, answers, reports, questions |

---

## 4. Embedding Model

- **Model:** `BAAI/bge-small-en-v1.5` (configurable via `CHROMA_EMBEDDING_MODEL`)
- **Type:** SentenceTransformer (dense vector)
- **Usage:** All collections with an embedding function use this model for the document → vector conversion.
