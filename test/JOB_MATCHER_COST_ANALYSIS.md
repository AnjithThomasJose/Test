# Job Matcher Agent — Token Usage & Cost Analysis

**Analysis Date:** March 11, 2025  
**Scope:** `agents/agents/job_matcher.py`  
**Environment:** QA (uses `gemini-2.5-flash` per `settings.py`)  
**Region:** India (Google AI / Vertex AI)

---

## 1. Token Usage Breakdown

### 1.1 LLM Call Sites

| Call Site | Purpose | Input Tokens (est.) | Output Tokens (est.) | Frequency per Session |
|-----------|---------|---------------------|----------------------|------------------------|
| `classify_candidate_domain()` | Domain classification (SOFTWARE_ENGINEERING, AI_ML, etc.) | 2,000–3,500 | 80–120 | 1× |
| `analyze_job_batch_with_llm()` | Per-job contextual scoring + rationale | 3,500–8,000/job | 800–1,500/job | **N×** (N = 10–35 jobs) |
| `llm_filter_and_rank_jobs()` | Legacy: filter/rank up to 50 jobs | 8,000–15,000 | 1,500–3,000 | 0× (career coach only) |
| `generate_llm_rationale_for_job()` | Per-job rationale (30–85% match band) | 2,500–5,000 | 200–400 | 0× (different flow) |

### 1.2 Per-Call Token Composition

**`classify_candidate_domain`:**
- Anonymized resume JSON: ~1,500–3,000 chars → ~400–750 tokens
- Domain definitions (35 domains): ~2,800 chars → ~700 tokens
- Instructions: ~500 chars → ~125 tokens  
- **Total input:** ~2,000–3,500 tokens  
- **Output:** JSON with `primary_domain`, `confidence`, `reasoning` → ~80–120 tokens

**`analyze_job_batch_with_llm` (batch_size=1):**
- `ORDERED_ANALYSIS_STEPS_UNIFIED`: ~900 chars → ~225 tokens
- `SCORING_GUIDELINES_0_100`: ~1,200 chars → ~300 tokens
- `CRITICAL_RULES`: ~2,000 chars → ~500 tokens
- Candidate context block: ~200 tokens
- **Full resume JSON** (sanitized): 2,000–6,000 chars → 500–1,500 tokens
- **1 job** (full JD): 1,500–4,000 chars → 400–1,000 tokens
- Instructions + JSON schema: ~800 tokens  
- **Total input per call:** ~3,500–8,000 tokens  
- **Output:** JSON array (1 job) with skills_matched, match_score, rationales → ~800–1,500 tokens

### 1.3 Multimodal Inputs

**None.** All inputs are text (resume JSON, job descriptions, prompts). No images or code files.

### 1.4 Embedding Usage (Gemini)

- **Model:** `models/gemini-embedding-001`
- **Usage:** `_calculate_semantic_similarity_resume_job()` (compare flow only)
- **Calls:** 2 per compare (resume text + job text)
- Embedding tokens are billed separately from generative tokens; typically ~$0.00001/1K tokens.

---

## 2. Cost Calculation

### 2.1 Pricing Assumptions

| Source | Input ($/1M tokens) | Output ($/1M tokens) | Notes |
|--------|---------------------|----------------------|-------|
| **Model Registry** (internal) | $1.00 | $1.00 | `core/model_registry.py` |
| **Gemini 2.5 Flash** (actual 2026) | $0.30/1M | $2.50/1M | Google AI pricing |
| **gemini-embedding-001** | $0.15/1M tokens | — | Embedding model |

**Assumption:** Model Registry used for internal tracking. Actual Gemini 2.5 Flash: Input $0.30/1M, Output $2.50/1M.

**Exchange rate:** 1 USD = **92 INR** (2026).

### 2.2 Per-Session Cost (Career Flow)

**Typical session:** 25 jobs analyzed (midpoint of 10–35).

| Component | Input Tokens | Output Tokens | Cost (USD) |
|-----------|--------------|---------------|------------|
| Domain classification | 2,500 | 100 | $0.0026 |
| Batch analysis (25×) | 87,500 | 25,000 | $0.1125 |
| **Total** | **90,000** | **25,100** | **$0.115** |

**Cost in INR:** 0.115 × 92 ≈ **₹10.60 per session**

### 2.3 Compare Flow (1 candidate vs 1 job)

| Component | Input Tokens | Output Tokens | Cost (USD) |
|-----------|--------------|---------------|------------|
| Domain classification | 2,500 | 100 | $0.0026 |
| Batch analysis (1×) | 4,000 | 1,000 | $0.005 |
| Embeddings (2×) | ~500 equiv. | — | ~$0.0005 |
| **Total** | **~7,000** | **~1,100** | **~$0.008** |

**Cost in INR:** ~**₹0.74 per compare**

### 2.4 Tier Clarification

- **Google AI Studio (API key):** Uses `GOOGLE_API_KEY`; free tier has limits; pay-as-you-go above.
- **Vertex AI:** Different SKU; may have region surcharges; India typically no extra surcharge.
- **Model registry** values may not match official Google pricing; verify against current docs.

---

## 3. Context Efficiency

### 3.1 Largest Token Consumers

| Consumer | Est. Tokens | % of Total | Notes |
|----------|-------------|-----------|-------|
| **Full resume (×N)** | 25 × 1,200 = 30,000 | ~33% | Same resume sent N times |
| **Static prompts (×N)** | 25 × 1,025 = 25,625 | ~28% | ORDERED_ANALYSIS + SCORING + CRITICAL_RULES |
| **Job descriptions (×N)** | 25 × 700 = 17,500 | ~19% | One JD per call |
| **Output (rationales)** | 25,000 | ~22% | Required for UX |

### 3.2 Redundancy & Waste

1. **Resume repeated N times**  
   - Same full resume JSON sent in every `analyze_job_batch_with_llm` call.  
   - **Waste:** ~(N−1) × 1,200 = 24 × 1,200 ≈ **28,800 tokens/session**

2. **Static prompts repeated N times**  
   - `ORDERED_ANALYSIS_STEPS_UNIFIED`, `SCORING_GUIDELINES_0_100`, `CRITICAL_RULES` (~1,025 tokens) sent per job.  
   - **Waste:** (N−1) × 1,025 ≈ **24,600 tokens/session**

3. **No system instruction / prompt caching**  
   - `invoke_llm` supports `system_instruction` for Gemini caching, but job_matcher does not use it.  
   - Static content could be cached once per session.

4. **batch_size = 1**  
   - One job per LLM call for “ChatGPT-quality” scoring.  
   - Increases total calls and repeated context.

5. **Full JD in every prompt**  
   - `fullJobDescription` + responsibilities sent per job.  
   - Some JDs are 2,000+ chars; truncation or summarization could reduce size.

---

## 4. Cost Reduction Recommendations

### 4.1 High Impact

| Recommendation | Est. Savings | Implementation | Latency Impact |
|----------------|-------------|----------------|----------------|
| **Use Gemini context caching** | 15–25% | Move static prompts (ORDERED_ANALYSIS, SCORING_GUIDELINES, CRITICAL_RULES) to `system_instruction`; cache for session | Lower (cached tokens cheaper) |
| **Batch 3–4 jobs per call** | 30–50% | Increase `batch_size` from 1 to 3–4; validate score quality | Slight increase per call |
| **Resume in system/cache once** | 20–30% | Put resume in cached system block; per-call prompt = jobs only | Lower |
| **Truncate JD description** | 10–15% | Cap `fullJobDescription` at 800–1,200 chars; keep skills/requirements | None |

### 4.2 Medium Impact

| Recommendation | Est. Savings | Implementation | Latency Impact |
|----------------|-------------|----------------|----------------|
| **Use Gemini Flash Lite for domain** | 5–10% | Route `classify_candidate_domain` to `gemini-2.5-flash-lite` | Slightly faster |
| **Cache domain classification** | 2–5% | Store `_candidate_domain` on resume; skip LLM when present | Faster |
| **Reduce `MAX_JOBS_FOR_LLM`** | Variable | Lower from 35 to 25; rely more on vector pre-filtering | Faster, fewer tokens |

### 4.3 Lower Impact

| Recommendation | Est. Savings | Implementation | Latency Impact |
|----------------|-------------|----------------|----------------|
| **Cap `max_output_tokens`** | 2–5% | Already 6,000 for batch; consider 4,000 if rationales are shorter | None |
| **Resume summarization** | 10–20% | Send 500-char summary instead of full JSON for low-priority jobs | Possible quality loss |

---

## 5. Projected Savings

### 5.1 Scenario: 50 Career-Flow Requests/Day

**Current (estimated):**
- 50 × $0.115 = **$5.75/day** ≈ **₹529/day**
- Monthly: **~$173** ≈ **₹15,916**

**After optimizations (conservative):**
- Context caching: −20%
- Batch size 3: −35%
- JD truncation: −10%  
- **Combined (multiplicative):** ~1 − (0.8 × 0.65 × 0.9) ≈ **53% reduction**

**Projected:**
- 50 × $0.115 × 0.47 ≈ **$2.70/day** ≈ **₹248/day**
- Monthly: **~$81** ≈ **₹7,452**

**Savings:** ~**₹8,464/month** (~53%)

### 5.2 Scenario: 100 Requests/Day

**Current:** ~₹15,916/month  
**Projected (same 53%):** ~₹7,452/month  
**Savings:** ~**₹8,464/month**

---

## 6. Latency Considerations

| Change | Cost | Latency |
|--------|------|---------|
| Context caching | ↓ | ↓ (fewer tokens to process) |
| Batch size 3–4 | ↓↓ | ↑ (larger prompts, more output) |
| Flash Lite for domain | ↓ | ↓ (faster model) |
| Fewer jobs (MAX 25) | ↓ | ↓ (fewer calls) |
| JD truncation | ↓ | ≈ (same call count) |

For an interactive coding/UX agent, batching 3–4 jobs may add ~2–5 s per batch; parallel batching (5 concurrent) helps. Domain caching and Flash Lite improve latency.

---

## 7. Assumptions & Caveats

1. **Pricing:** Model registry uses $1.00/1M for both input/output. Actual Google AI / Vertex AI pricing may differ.
2. **Region:** No explicit India surcharge in docs; billing typically in USD.
3. **Exchange rate:** 92 INR/USD (2026).
4. **Token estimation:** Uses `_estimate_tokens` (≈4 chars/token for natural language, ≈3.5 for JSON).
5. **Session mix:** Assumes career flow dominates; compare flow and career coach flows have different profiles.
6. **Cache hit rate:** Internal LLM cache (30 min TTL) can reduce repeat calls for similar prompts.

---

## 8. Quick Wins (Priority Order)

1. ~~Add `system_instruction` with static prompts for Gemini context caching.~~ **DONE** (Mar 2025): Resume + static prompts (ORDERED_ANALYSIS, SCORING_GUIDELINES, CRITICAL_RULES) moved to `system_instruction`; per-call prompt contains only jobs. Enables Gemini implicit caching.
2. Increase `batch_size` from 1 to 3 and validate score quality.
3. Truncate `fullJobDescription` to 1,000 chars in `analyze_job_batch_with_llm`.
4. Use `_get_candidate_domain_from_resume` / stored domain to skip `classify_candidate_domain` when possible.
5. Route `classify_candidate_domain` to `gemini-2.5-flash-lite` for cost and latency.
