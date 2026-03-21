# KA Agents — Cost Optimizations Summary

**Last Updated:** March 2026  
**Scope:** All cost-related optimizations implemented across the KA AI agent infrastructure.

**Pricing (2026):** Gemini 2.5 Flash $0.30/1M input, $2.50/1M output; gemini-embedding-001 $0.15/1M tokens. Exchange rate: **1 USD = 92 INR**.

---

## Executive Summary

| Category | Optimizations Done | Est. Savings |
|----------|-------------------|--------------|
| **Embeddings** | Persistent cache, in-process cache, content dedup, batching | **20–50%** fewer embedding API calls |
| **LLM** | invoke_llm routing, LLM cache, parallel calls | **15–30%** via cache hits + observability |
| **Infrastructure** | Auto-rerank removal, bounded dicts, token estimation | Variable (avoids waste) |
| **Combined** | All above | **~25–40%** overall cost reduction (conservative) |

---

## 1. Embedding Optimizations

### 1.1 Gemini Embedding Cache (Persistent)

| Attribute | Detail |
|-----------|--------|
| **File** | `core/gemini_embedding_cache.py` |
| **Type** | SQLite-backed persistent cache |
| **TTL** | 24 hours (configurable) |
| **Scope** | `job_matcher.py`, `ranker.py` |
| **Expected hit rate** | **>90%** for resume/JD text across restarts |

**Where used:**
- Domain classification (resume + JD)
- Semantic similarity (resume vs job)
- Experience embeddings for tier computation

**Savings:** Eliminates re-embedding identical texts. Resume and JD text rarely changes between rerank runs → high cache hit rate.

---

### 1.2 In-Process Embedding Cache

| Attribute | Detail |
|-----------|--------|
| **File** | `core/embedding_cache.py` |
| **Type** | LRU in-memory cache |
| **Default** | Enabled (`ENABLE_EMBEDDING_CACHE=true`) |
| **TTL** | 1 hour |
| **Max entries** | 10,000 |

**Savings:** **20–50%** fewer embedding API calls (per SYSTEM_REVIEW_REPORT.md).

---

### 1.3 ChromaDB Content Deduplication

| Attribute | Detail |
|-----------|--------|
| **File** | `chroma.py` |
| **Mechanism** | `content_hash` check before embedding inserts |
| **Effect** | Skips re-embedding when resume content is unchanged |

**Savings:** Avoids duplicate embedding work on re-indexing; reduces redundant ChromaDB writes.

---

### 1.4 Batched Embeddings

| Attribute | Detail |
|-----------|--------|
| **Scope** | `job_matcher.py`, `ranker.py` |
| **Domain classification** | Batch embed resume/JD + all domain descriptions in one call |
| **Semantic similarity** | Batch embed resume + JD together |
| **Experience embeddings** | Up to 20 texts per batch (`_EMBED_BATCH_SIZE`) |

**Savings:** Fewer API round-trips; embedding API often charges per request, so batching reduces total requests.

---

## 2. LLM Optimizations

### 2.1 invoke_llm Routing (Ranker + Job Matcher)

| Attribute | Detail |
|-----------|--------|
| **Scope** | `analyze_candidate_batch_with_llm` (ranker), job_matcher batch analysis |
| **Effect** | Internal cache, quota tracking, observability |

**Savings:** Cache hits on repeat JDs/candidates; no direct cost reduction per call, but **15–25%** potential savings when same/similar prompts are re-used within 30 min TTL.

---

### 2.2 LLM Response Cache

| Attribute | Detail |
|-----------|--------|
| **File** | `models/llm_invoker.py` |
| **Backend** | SQLite or Redis |
| **TTL** | 30 min (configurable) |
| **Key** | provider + model + temperature + prompt hash |

**Savings:** Repeat calls with identical prompts return cached response → **0 cost** for cache hits.

---

### 2.3 Parallel LLM Calls

| Attribute | Detail |
|-----------|--------|
| **File** | `conversational_mentor.py` |
| **Mechanism** | `asyncio.gather()` for independent LLM calls |

**Savings:** **30–50%** latency reduction (per SYSTEM_REVIEW_REPORT.md); indirectly reduces cost by avoiding timeouts and retries.

---

### 2.4 System Instruction / Context Caching

| Attribute | Detail |
|-----------|--------|
| **Scope** | Ranker, Job Matcher |
| **Mechanism** | Static prompts (JD, ORDERED_ANALYSIS, SCORING_GUIDELINES, CRITICAL_RULES) in `system_instruction` |

**Savings:** **15–25%** via Gemini context caching (cached tokens billed at lower rate).

---

## 3. Infrastructure Optimizations

### 3.1 Auto-Rerank Removal

| Attribute | Detail |
|-----------|--------|
| **Removed** | `daily_rerank_all_jds_handler`, `test_daily_rerank` endpoint |
| **Effect** | No scheduled rerank jobs |

**Savings:** Eliminates periodic embedding + LLM calls that were triggered by cron/scheduler.

---

### 3.2 Bounded Tenant Dicts

| Attribute | Detail |
|-----------|--------|
| **File** | `core/supervisor_agent.py` |
| **Class** | `BoundedTenantDict` |
| **Config** | 500 max entries, 2 hr TTL |

**Savings:** Prevents unbounded memory growth (~100KB–10MB/day); reduces risk of OOM and restarts.

---

### 3.3 Token Estimation

| Attribute | Detail |
|-----------|--------|
| **File** | `models/llm_invoker.py` |
| **Change** | Character-based estimation (≈4 chars/token text, ≈3.5 code) vs `word_count * 1.3` |

**Savings:** More accurate cost tracking and quota management; avoids over/under-estimation.

---

### 3.4 Quota Manager & Cost Alerts

| Attribute | Detail |
|-----------|--------|
| **File** | `core/quota_manager.py` |
| **Features** | Usage tracking, cost alerts, batch request optimization |

**Savings:** Prevents runaway costs; enables proactive budget control.

---

## 4. Expected Savings by Scenario

### 4.1 Ranker (30 requests/day, typical)

| Metric | Before | After (est.) | Savings |
|--------|--------|--------------|---------|
| **Daily cost** | $3.00 (~₹276) | ~$2.16 (~₹199) | **~28%** |
| **Monthly cost** | ~$90 (~₹8,280) | ~$65 (~₹5,980) | **~₹2,300/month** |

*Assumptions: invoke_llm cache hits, embedding cache hits, system_instruction caching.*

---

### 4.2 Ranker (50 requests/day, heavy)

| Metric | Before | After (est.) | Savings |
|--------|--------|--------------|---------|
| **Monthly cost** | ~$150 (~₹13,800) | ~$108 (~₹9,936) | **~₹3,864/month** |

---

### 4.3 Job Matcher (50 career-flow requests/day)

| Metric | Before | After (est.) | Savings |
|--------|--------|--------------|---------|
| **Daily cost** | $5.75 (~₹529) | ~$2.70 (~₹248) | **~53%** (with batching + truncation) |
| **Monthly cost** | ~$173 (~₹15,916) | ~$81 (~₹7,452) | **~₹8,464/month** |

*Note: Job Matcher savings assume batch_size increase and JD truncation (recommendations not yet fully implemented).*

---

### 4.4 Embedding Cost (Both Agents)

| Scenario | Est. Embedding Cost | With Cache (90% hit) |
|----------|---------------------|------------------------|
| **Per ranker session** | ~$0.002 | ~$0.0002 |
| **Per job_matcher compare** | ~$0.0005 | ~$0.00005 |

**Embedding savings:** **~90%** when cache hit rate is high (resume/JD text stable).

---

## 5. Summary Table

| Optimization | Type | Est. Savings | Status |
|--------------|------|--------------|--------|
| Gemini embedding cache | Embeddings | >90% hit rate | ✅ Done |
| In-process embed cache | Embeddings | 20–50% fewer calls | ✅ Done |
| ChromaDB content_hash | Embeddings | Skips duplicates | ✅ Done |
| Batched embeddings | Embeddings | Fewer API round-trips | ✅ Done |
| invoke_llm routing | LLM | Cache + observability | ✅ Done |
| LLM response cache | LLM | 0 cost on cache hit | ✅ Done |
| Parallel LLM calls | LLM | 30–50% latency ↓ | ✅ Done |
| System instruction | LLM | 15–25% (Gemini cache) | ✅ Done |
| Auto-rerank removal | Infrastructure | Eliminates scheduled calls | ✅ Done |
| Bounded tenant dicts | Memory | Prevents leaks | ✅ Done |
| Token estimation | Observability | Accurate tracking | ✅ Done |
| Quota manager | Observability | Cost alerts | ✅ Done |

---

## 6. Gemini Pricing Reference (2026)

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| **Gemini 2.5 Flash** | $0.30/1M tokens | $2.50/1M tokens | Primary for job_matcher, ranker |
| **Gemini 2.0 Flash-Lite** | $0.075/1M | $0.30/1M | Cheaper alternative for simple tasks |
| **gemini-embedding-001** | $0.15/1M tokens | — | Domain classification, semantic similarity |

*Source: [Google AI Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing). Model registry (`core/model_registry.py`) uses $1.00/1M for internal tracking; actual API costs differ.*

---

## 7. Configuration Reference

| Env Variable | Default | Purpose |
|--------------|---------|---------|
| `ENABLE_GEMINI_EMBED_CACHE` | `true` | Toggle persistent embedding cache |
| `GEMINI_EMBED_CACHE_TTL_SECONDS` | `86400` (24h) | Embedding cache TTL |
| `GEMINI_EMBED_CACHE_PATH` | `/tmp/gemini_embed_cache.sqlite` | SQLite path |
| `ENABLE_EMBEDDING_CACHE` | `true` | Toggle in-process embedding cache |
| `EMBEDDING_CACHE_TTL_SECONDS` | `3600` | In-process cache TTL |
| `USE_LLM_CACHE` | `true` | Toggle LLM response cache |
| `LLM_CACHE_TTL_SECONDS` | `1800` | LLM cache TTL |

---

## 8. Caveats

1. **Pricing:** USD estimates use Model Registry ($1.00/1M) for internal tracking. Actual Gemini 2.5 Flash: $0.30/1M input, $2.50/1M output.
2. **Cache hit rates:** Depend on workload; repeat JDs/candidates → higher hits.
3. **Exchange rate:** 92 INR/USD (2026).
4. **Job Matcher batching:** Full 53% savings assumes batch_size=3 and JD truncation (partially done).
