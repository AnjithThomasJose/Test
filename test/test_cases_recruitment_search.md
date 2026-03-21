# Recruitment Candidate Search - Test Cases

## Overview

Tests for `POST /search-candidate`: query parsing, normalization, vector search, name post-filtering, fallback logic, and API contract.

---

## 1. Name Search

### TC-NAME-001: Partial name (first name only)
**Request:**
```json
{ "query": "Ashwin" }
```
**Expected:**
- `parsed_query.parsed_name` = `"Ashwin"` (LLM or rule-based)
- `search_text` includes "Ashwin"; no name metadata filter
- **Post-filter:** keep only candidates whose `candidate_name` contains "ashwin" (case-insensitive)
- `"Ashwin Nair"` included; `"John Doe"` excluded
- `total_found` ≥ 0; `matches[].candidate_name` contains "Ashwin"

### TC-NAME-002: Full name
**Request:**
```json
{ "query": "Ashwin Nair" }
```
**Expected:**
- `parsed_name` = `"Ashwin Nair"`
- Post-filter: `"ashwin nair" in candidate_name.lower()` → include
- Only candidates with "Ashwin Nair" in name

### TC-NAME-003: Name + role (combined)
**Request:**
```json
{ "query": "Ashwin Nair Python developer" }
```
**Expected:**
- `parsed_name` = `"Ashwin Nair"` (or similar), `parsed_role` set
- `search_text` = name + role (order: name, experience, role, skills)
- Name post-filter applied; role/skills affect vector recall only (no role filter in current design)

### TC-NAME-004: Single-word name (rule fallback)
**Request:**
```json
{ "query": "Ashwin" }
```
**Note:** Rule-based parser requires 2–4 words for `name`; single word yields `parsed_name=None` there. LLM can return `"Ashwin"`.
**Expected:**
- If LLM used: `parsed_name` = `"Ashwin"`, post-filter runs
- If rules used: `parsed_name` = `None`, no name post-filter; results are vector-only

### TC-NAME-005: Name with no matches
**Request:**
```json
{ "query": "XyzNonexistentName123" }
```
**Expected:**
- `total_found` = 0, `matches` = []
- No 500; `processing_time_ms` present
- `parsed_query` populated; `filters_applied` as applicable

---

## 2. Role & Skills Search

### TC-ROLE-001: Role only
**Request:**
```json
{ "query": "Senior Python Developer" }
```
**Expected:**
- `parsed_role` = "Python Developer" (or similar), `parsed_experience` = "Senior"
- `search_text` includes role and experience
- `normalized["seniority"]` = "senior" → filter `seniority_level` = "senior" if metadata exists
- `matches` have `match_score`; `candidate_skills` and `candidate_name` populated when available

### TC-ROLE-002: Skills only
**Request:**
```json
{ "query": "Python Django AWS" }
```
**Expected:**
- `parsed_skills` = ["Python", "Django", "AWS"] (or equivalents)
- `search_text` includes skills
- `normalized["skills"]` may expand (e.g. "Python" → ["Python", "Python3", "Py"])
- No skills metadata filter in current design; skills affect vector search only

### TC-ROLE-003: Role + skills + location
**Request:**
```json
{ "query": "Senior Python Developer in Dubai with Django" }
```
**Expected:**
- `parsed_role`, `parsed_location` = "Dubai", `parsed_skills` contains "Django", `parsed_experience` = "Senior"
- `normalized["location"]` (e.g. "Dubai") → filter `current_city` = "Dubai"
- `normalized["seniority"]` → filter `seniority_level`
- `search_text` = experience + role + skills (and possibly normalized location if ever added to search_text)
- Fallback: if 0 results, retry without location; then without all filters

---

## 3. Location & Filters

### TC-LOC-001: Location only
**Request:**
```json
{ "query": "candidates in Dubai" }
```
**Expected:**
- `parsed_location` = "Dubai" (or similar)
- `normalized["location"]` → filter `current_city`
- `search_text` from role/skills or `raw_query` if nothing else parsed
- If 0 with filter: fallback without location

### TC-LOC-002: Normalized location (NYC, DXB)
**Request:**
```json
{ "query": "Python developer in NYC" }
```
**Expected:**
- `parsed_location` = "NYC"
- `normalized["location"]` = "New York" (or similar via Groq)
- Filter `current_city` = "New York"
- Normalizer failure → fallback to `location.title()`

### TC-LOC-003: Remote
**Request:**
```json
{ "query": "Remote React developer" }
```
**Expected:**
- `parsed_location` = "Remote"
- `normalized["location"]` = "Remote" (or equivalent)
- Filter on `current_city`; depends on how "Remote" is stored in metadata

---

## 4. Seniority & Education

### TC-SEN-001: Seniority variants
**Request:**
```json
{ "query": "Sr Python developer" }
```
**Expected:**
- `parsed_experience` = "Sr" or "Senior"
- `normalized["seniority"]` = "senior"
- Filter `seniority_level` = "senior" when metadata exists

### TC-EDU-001: Education in query
**Request:**
```json
{ "query": "Python developer with Master's degree" }
```
**Expected:**
- `parsed_education` = "Master's" (or similar)
- `normalized["education"]` = "masters"
- Filter `education_level` = "masters"
- Only applied if collection has `education_level` metadata

---

## 5. API & Request Validation

### TC-API-001: Minimal valid request
**Request:**
```json
{ "query": "Python developer" }
```
**Expected:**
- 200; `query`, `parsed_query`, `matches`, `total_found`, `processing_time_ms`, `search_strategy`, `filters_applied`, `avg_match_score`, `top_match_score`
- `top_k` defaults to `RECRUITMENT_DEFAULT_TOP_K` when omitted
- `matches[].candidate_id`, `candidate_name`, `match_score`, `candidate_skills`, `total_experience_years`, `current_location`, `seniority_level`, `education_level`

### TC-API-002: With top_k
**Request:**
```json
{ "query": "Python", "top_k": 10 }
```
**Expected:**
- `len(matches)` ≤ 10
- `top_k` in [1, RECRUITMENT_MAX_TOP_K]; 400 if out of range

### TC-API-003: Empty query
**Request:**
```json
{ "query": "" }
```
**Expected:**
- 400; validation error (query required / cannot be empty)

### TC-API-004: Query too long
**Request:**
```json
{ "query": "< 501 character string>" }
```
**Expected:**
- 400 if length > 500; otherwise 200

### TC-API-005: With callback_url
**Request:**
```json
{ "query": "Python", "callback_url": "https://valid-domain/path" }
```
**Expected:**
- 200; if callback is valid, results also sent to callback (payload with `node: "recruitment_search"`, `status: "completed"`, `output` with `matches`, `parsed_query`, etc.)
- Response body still contains full result

### TC-API-006: Invalid or missing GenAI token
**Request:**
```json
{ "query": "Python" }
```
**Headers:** Missing or invalid auth token as per `verify_request_token`
**Expected:**
- 401

---

## 6. Fallback Search Behavior

### TC-FB-001: Zero results with location filter
**Setup:** Query that applies `current_city` filter and returns 0 from Chroma.
**Request:**
```json
{ "query": "Python developer in CityWithNoCandidates" }
```
**Expected:**
- First run: `where` includes `current_city`; 0 results
- Second run: `where` without `current_city`; return any non-zero result
- Log: "No results with location filter, trying without location filter..."

### TC-FB-002: Zero results with all filters
**Setup:** Query with location + seniority (or education) such that first and second tries yield 0.
**Expected:**
- Third run: `filters=[]`, `search_text` unchanged
- Log: "No results with filters, trying pure vector search..."
- Final `matches` from vector-only run if any

### TC-FB-003: No fallback when results found
**Request:**
```json
{ "query": "Python developer in Dubai" }
```
**Expected:**
- First search with filters returns > 0 → that result used; no fallback

---

## 7. Response Shape & Post-Filtering

### TC-RESP-001: Matches include candidate_name and candidate_skills
**Request:**
```json
{ "query": "Python" }
```
**Expected:**
- Every `match` has: `candidate_id`, `candidate_name` (string, or "N/A"), `candidate_skills` (list), `match_score`, `total_experience_years`, `current_location`, `seniority_level`, `education_level`
- No `rationale`, `highlights`, `concerns`, `matched_skills`, `unmatched_skills`

### TC-RESP-002: Name post-filter excludes non-matching
**Request:**
```json
{ "query": "Ashwin" }
```
**Expected:**
- Every `match.candidate_name` contains "Ashwin" (case-insensitive)
- No candidate with name "John Doe" only

### TC-RESP-003: parsed_query and filters_applied
**Request:**
```json
{ "query": "Senior Python Developer in Dubai" }
```
**Expected:**
- `parsed_query`: `raw_query`, `parsed_role`, `parsed_location`, `parsed_experience`, `parsed_skills`, `parsed_education`, `parsed_company_type`, `parsed_name`, `query_id`, `timestamp`
- `filters_applied`: includes `location` and `seniority` when normalized

### TC-RESP-004: Deduplication by candidate
**Note:** Chroma may return multiple chunks per candidate; adapter currently maps by `result.id` (chunk id). If multiple chunks per candidate appear, each may be a separate match.
**Expected:**
- Document behavior: either one match per `candidate_id` or explicitly allow multiple rows per candidate (to be defined). Tests should assert chosen behavior.

---

## 8. Edge Cases

### TC-EDGE-001: Very short query
**Request:**
```json
{ "query": "Dev" }
```
**Expected:**
- Parser may set `parsed_role` or leave mostly empty; `search_text` = "Dev" or `raw_query`
- 200; `matches` from vector search; may be broad

### TC-EDGE-002: Special characters
**Request:**
```json
{ "query": "C++ .NET developer" }
```
**Expected:**
- No 500; parser and normalizer handle gracefully
- `search_text` includes "C++", ".NET" (or normalized forms)

### TC-EDGE-003: Multiple names (ambiguous)
**Request:**
```json
{ "query": "John or Smith" }
```
**Expected:**
- Parser may treat as role/skills or single name; implementation-defined
- 200; consistent behavior (no crash)

### TC-EDGE-004: top_k = 1
**Request:**
```json
{ "query": "Python", "top_k": 1 }
```
**Expected:**
- `len(matches)` ≤ 1; 200

### TC-EDGE-005: top_k at maximum
**Request:**
```json
{ "query": "developer", "top_k": 100 }
```
**Expected:**
- 200 if 100 ≤ RECRUITMENT_MAX_TOP_K; 400 if > max

---

## 9. Normalization (Unit-Level)

### TC-NORM-001: Location
- "NYC" → "New York"; "DXB" → "Dubai"; "Bay Area" → "San Francisco"
- Groq failure → `location.title()`

### TC-NORM-002: Skills
- "react" → ["React", "ReactJS", "React.js"]; "k8s" → ["Kubernetes", "K8s"]
- Multiple skills: flatten and deduplicate

### TC-NORM-003: Seniority
- "Sr", "Lead", "Principal" → "senior"; "Entry-level" → "junior"
- Output in {"junior","mid","senior","lead"}

### TC-NORM-004: Education
- "MS", "MSc" → "masters"; "BSc" → "bachelors"; "PhD" → "phd"
- Output in {"high_school","bachelors","masters","phd"}

---

## 10. Query Parser (Unit-Level)

### TC-PARSE-001: LLM returns valid JSON
- All fields present as null or value; `RecruitmentQuery` builds; `query_id` is UUID.

### TC-PARSE-002: LLM timeout or bad JSON
- Fallback to `_parse_with_rules`; `RecruitmentQuery` still returned.

### TC-PARSE-003: Rule-based name (2–4 capitalized words, no role keywords)
- "John Smith" → `parsed_name` = "John Smith"
- "John" only → `parsed_name` = None in rules (LLM can still set it)

### TC-PARSE-004: Rule-based role/location/skills
- "Python developer in Dubai" → role from keywords, location "Dubai", skills ["Python"] if in list.

---

## Summary Table

| Category    | Test IDs        | Focus                                      |
|------------|------------------|--------------------------------------------|
| Name       | TC-NAME-001–005  | Partial/full name, combined, no match      |
| Role/Skills| TC-ROLE-001–003  | Role, skills, role+skills+location         |
| Location   | TC-LOC-001–003   | City, normalized, Remote                   |
| Seniority  | TC-SEN-001       | Sr → senior                                |
| Education  | TC-EDU-001       | Master's → masters, filter                 |
| API        | TC-API-001–006   | Validation, top_k, callback, auth          |
| Fallback   | TC-FB-001–003    | Relax location, then all filters           |
| Response   | TC-RESP-001–004  | Schema, name post-filter, parsed_query     |
| Edge       | TC-EDGE-001–005  | Short, special chars, top_k bounds         |
| Normalize  | TC-NORM-001–004  | location, skills, seniority, education     |
| Parser     | TC-PARSE-001–004 | LLM, fallback, rules for name/role/loc     |
