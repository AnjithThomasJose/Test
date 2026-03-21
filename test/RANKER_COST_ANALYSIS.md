# Ranker Agent — Token Usage & Cost Analysis

**Analysis Date:** March 11, 2025  
**Scope:** `agents/agents/ranker.py`  
**Environment:** QA (uses `gemini-2.5-flash` per `settings.py`)  
**Region:** India (Google AI / Vertex AI)

---

## 1. Token Usage Breakdown

### 1.1 LLM Call Sites

| Call Site | Purpose | Input Tokens (est.) | Output Tokens (est.) | Frequency per Session |
|-----------|---------|---------------------|----------------------|------------------------|
| `analyze_candidate_batch_with_llm()` | Batch candidate scoring + rationale | 8,000–18,000/batch | 2,500–5,000/batch | **B×** (B = ceil(candidates/8)) |
| `_classify_domain_with_llm()` | JD domain classification | 3,000–5,000 | 80–120 | **0×** (DEPRECATED; embeddings used) |
| `generate_llm_rationale()` | Per-candidate rationale | 2,500–5,000 | 200–400 | **0×** (not used in main flow) |

### 1.2 Per-Batch Token Composition (`analyze_candidate_batch_with_llm`)

**Config:** `LLM_BATCH_SIZE = 8` candidates per call, `LLM_PARALLEL_BATCHES = 5` concurrent.

**Input per batch:**
- Job details (title, skills, experience, education, full description, responsibilities): 2,000–5,000 chars → 500–1,250 tokens
- Job domain section: ~100 tokens
- **8 candidates** (skills, experience, education, certs, projects, location): ~400 chars each → 3,200 chars → ~800 tokens
- `ORDERED_ANALYSIS_STEPS_UNIFIED`: ~225 tokens
- `SCORING_GUIDELINES_0_100`: ~300 tokens
- `CRITICAL_RULES`: ~500 tokens
- `SKILLS_MATCH_INSTRUCTIONS`: ~150 tokens
- Recruiter Q&A (optional): ~300 tokens
- Instructions + JSON schema: ~600 tokens  

**Total input per batch:** ~8,000–18,000 tokens (depends on JD length)

**Output per batch:** 8 JSON objects with skills_matched, match_score, rationale, positive_rationale, negative_rationale, summary, concerns → ~2,500–5,000 tokens

### 1.3 Non-LLM (Embedding) Usage

| Call Site | Purpose | Model | Frequency |
|-----------|---------|-------|-----------|
| `_classify_jd_domain_with_embeddings()` | JD domain classification | gemini-embedding-001 | 1× per JD |
| `_get_domain_embedding()` | Domain description embedding | gemini-embedding-001 | 1× per ranker run |
| `_get_text_embedding()` | Per-experience embedding | gemini-embedding-001 | N× (N = total work experiences across tier candidates) |
| `_calculate_domain_relevant_experience()` | Domain-relevant years | Uses above embeddings | Per candidate in tier batch |

**Note:** Tier classification (`compute_tiers_for_candidates`) uses embeddings only—no LLM. Domain classification uses embeddings, not the deprecated `_classify_domain_with_llm`.

### 1.4 Multimodal Inputs

**None.** All inputs are text (JD, candidate summaries). No images.

---

## 2. Cost Calculation

### 2.1 Pricing Assumptions

| Source | Input ($/1M tokens) | Output ($/1M tokens) |
|--------|---------------------|----------------------|
| **Model Registry** (internal) | $1.00 | $1.00 |
| **Gemini 2.5 Flash** (actual 2026) | $0.30/1M input | $2.50/1M output |
| **Embeddings** (gemini-embedding-001) | $0.15/1M tokens | — |

**Exchange rate:** 1 USD = **92 INR** (2026).

### 2.2 Per-Session Cost (Typical)

**Typical session:** 50 candidates sent to LLM (after tier filter).

| Component | Batches | Input Tokens | Output Tokens | Cost (USD) |
|-----------|---------|--------------|---------------|------------|
| Batch analysis (50 candidates) | 7 | 70,000 | 28,000 | $0.098 |
| Embeddings (JD domain + tier) | — | ~2,000 equiv. | — | ~$0.002 |
| **Total** | **7** | **~72,000** | **~28,000** | **~$0.10** |

**Cost in INR:** ~**₹9.20 per session** (at 92 INR/USD)

### 2.3 Per-Session Cost (Heavy: 150 candidates)

**Heavy session:** `MAX_CANDIDATES_TO_PROCESS = 150` (hard cap).

| Component | Batches | Input Tokens | Output Tokens | Cost (USD) |
|-----------|---------|--------------|---------------|------------|
| Batch analysis (150 candidates) | 19 | 228,000 | 76,000 | $0.304 |
| Embeddings | — | ~5,000 equiv. | — | ~$0.005 |
| **Total** | **19** | **~233,000** | **~76,000** | **~$0.31** |

**Cost in INR:** ~**₹28.50 per session** (at 92 INR/USD)

### 2.4 Tier Clarification

- **Google AI Studio** (API key): Uses `GOOGLE_API_KEY`
- **Vertex AI:** Different SKU; verify current pricing
- **Note:** `analyze_candidate_batch_with_llm` is routed through `invoke_llm` (enables internal cache, quota tracking, observability).

---

## 3. Context Efficiency

### 3.1 Largest Token Consumers

| Consumer | Est. Tokens/Batch | % of Total | Notes |
|----------|-------------------|-----------|-------|
| **JD (×B batches)** | 1,000 × B | ~12% | Same JD sent in every batch |
| **8 candidate summaries** | 800 × 8 = 6,400 | ~45% | Candidate data |
| **Static prompts (×B)** | 1,675 × B | ~23% | ORDERED_ANALYSIS + SCORING + CRITICAL + SKILLS |
| **Output (8 rationales)** | 3,200 | ~20% | Required for UX |

### 3.2 Redundancy & Waste

1. **JD repeated B times**  
   - Same full job description sent in every batch.  
   - **Waste:** (B−1) × 1,000 ≈ **6,000 tokens** (7 batches) to **18,000 tokens** (19 batches)

2. **Static prompts repeated B times**  
   - ORDERED_ANALYSIS, SCORING_GUIDELINES, CRITICAL_RULES, SKILLS_MATCH (~1,675 tokens) sent per batch.  
   - **Waste:** (B−1) × 1,675 ≈ **10,000–30,000 tokens/session**

3. **No system instruction / prompt caching**  
   - Static content could be cached via `system_instruction`; not used.

4. **invoke_llm routing**  
   - `analyze_candidate_batch_with_llm` uses `invoke_llm`; internal cache, quota tracking, and observability enabled.

5. **Full JD in every batch**  
   - `fullJobDescription` + responsibilities sent per batch; some JDs are 3,000+ chars.

---

## 4. Cost Reduction Recommendations

### 4.1 High Impact

| Recommendation | Est. Savings | Implementation | Latency Impact |
|----------------|-------------|----------------|----------------|
| **Use Gemini context caching** | 15–25% | Move static prompts to `system_instruction`; cache for session | Lower |
| **Route through invoke_llm** | ✅ DONE | `analyze_candidate_batch_with_llm` uses `invoke_llm` | None |
| **JD in system/cache once** | 10–15% | Put JD in cached block; per-batch prompt = candidates only | Lower |
| **Truncate JD description** | 10–15% | Cap `jd_responsibilities` at 1,200 chars | None |

### 4.2 Medium Impact

| Recommendation | Est. Savings | Implementation | Latency Impact |
|----------------|-------------|----------------|----------------|
| **Increase batch size** | 15–25% | Test `LLM_BATCH_SIZE=10` or 12; validate quality | Slight ↑ per call |
| **Reduce MAX_CANDIDATES_TO_PROCESS** | Variable | Lower from 150 to 100; rely on tier filter | Faster, fewer tokens |
| **Tighten TIER_LLM_CUTOFF** | Variable | Send only Tiers 1–2 to LLM (skip Tier 3) | Faster, possible quality trade-off |

### 4.3 Lower Impact

| Recommendation | Est. Savings | Implementation | Latency Impact |
|----------------|-------------|----------------|----------------|
| **Cap max_output_tokens** | 2–5% | Already 6,000; consider 5,000 if rationales are shorter | None |
| **Summarize candidate blocks** | 10–15% | Truncate project descriptions, limit certs to top 3 | Possible quality loss |

---

## 5. Projected Savings

### 5.1 Scenario: 30 Ranker Requests/Day

**Current (estimated):**
- 30 × $0.10 = **$3.00/day** ≈ **₹276/day**
- Monthly: **~$90** ≈ **₹8,280**

**After optimizations (conservative):**
- Context caching: −20%
- JD truncation: −10%
- Route through invoke_llm: +cache hits on repeat JDs  
- **Combined:** ~1 − (0.8 × 0.9) ≈ **28% reduction**

**Projected:**
- 30 × $0.10 × 0.72 ≈ **$2.16/day** ≈ **₹199/day**
- Monthly: **~$65** ≈ **₹5,980**

**Savings:** ~**₹2,300/month** (~28%)

### 5.2 Scenario: 50 Requests/Day (Heavy)

**Current:** ~$150/month (~₹13,800)  
**Projected (28%):** ~$108/month (~₹9,936)  
**Savings:** ~**₹3,864/month**

---

## 6. Latency Considerations

| Change | Cost | Latency |
|--------|------|---------|
| Context caching | ↓ | ↓ |
| Route through invoke_llm | ≈ | ≈ (+cache possible) |
| Batch size 10–12 | ↓↓ | ↑ (larger prompts) |
| Fewer candidates (MAX 100) | ↓ | ↓ |
| JD truncation | ↓ | ≈ |

---

## 7. Assumptions & Caveats

1. **Pricing:** Model registry $1.00/1M; actual Google pricing may differ.
2. **Token tracking:** `analyze_candidate_batch_with_llm` uses `invoke_llm`—usage is tracked in model registry / observability.
3. **Candidates per session:** 50 typical, 150 max; depends on ChromaDB results and tier filter.
4. **generate_llm_rationale:** Defined but not called in main ranker flow; may be dead code.
5. **Embeddings:** Tier + domain use embeddings; cost is small vs. LLM.

---

## 8. Quick Wins (Priority Order)

1. ~~**Route `analyze_candidate_batch_with_llm` through `invoke_llm`**~~ ✅ DONE — already uses `invoke_llm`.
2. Add `system_instruction` with static prompts for Gemini context caching.
3. Truncate `jd_responsibilities` to 1,200 chars.
4. Test `LLM_BATCH_SIZE=10` and validate score quality.
5. Consider lowering `MAX_CANDIDATES_TO_PROCESS` from 150 to 100.

---

## 9. Comparison with Job Matcher

| Metric | Job Matcher (Career) | Ranker |
|--------|----------------------|--------|
| **Primary unit** | 1 candidate vs. N jobs | 1 job vs. N candidates |
| **Batch size** | 1 job/call | 8 candidates/call |
| **Typical calls/session** | 25 | 7 |
| **Cost/session (typical)** | ~$0.115 | ~$0.10 |
| **Largest redundancy** | Resume × N | JD × B |
| **Uses invoke_llm** | Yes (mostly) | Yes |
