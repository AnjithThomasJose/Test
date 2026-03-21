# Compare Flow: Graph & Callback Nodes (resume_url in payload)

When `resume_url` is present in the payload and the request is a compare flow (`/compare-candidate-job`, `request_type=candidate_job_match`), the graph runs the following path. **Bold** = node sends output to callback; *italic* = no callback (passthrough/internal).

---

## Flow Diagram

```
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                     COMPARE FLOW (resume_url)                    │
                                    └─────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  dispatcher      │  (entry)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ validate_resume  │  ✅ CALLBACK
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ groq_resume_     │  ✅ CALLBACK
  │ parser           │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ interest_filler  │  ✅ CALLBACK
  └────────┬─────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
  ┌──────────┐  ┌────────────────────────┐
  │ resume_  │  │ skill_proficiency_       │  ✅ CALLBACK
  │ summary  │  │ analyzer                │
  └──────────┘  └────────────┬────────────┘
  (no callback)               │
                             ▼
                    ┌──────────────────┐
                    │ resume_scorer     │  ✅ CALLBACK
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ resume_assembler  │  ❌ NO CALLBACK (passthrough)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ job_matcher_        │  ❌ NO CALLBACK (preprocessor)
                    │ preprocessor (1st)   │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ job_matcher       │  ✅ CALLBACK (single-job match)
                    │ (1st run)         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ career_advisor_   │  ❌ NO CALLBACK (passthrough)
                    │ entry             │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ enhanced_role_fit │  ✅ CALLBACK
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ career_flow_      │  ❌ NO CALLBACK (passthrough)
                    │ fan_out           │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
  ┌─────────────────────┐     ┌──────────────────────┐
  │ career_advisor       │     │ job_matcher_         │  ❌ NO CALLBACK
  │                      │     │ preprocessor (2nd)   │
  └──────────┬──────────┘     └──────────┬────────────┘
             │                           │
             ▼                           ▼
  ┌─────────────────────────┐  ┌──────────────────┐
  │ market_and_course_       │  │ job_matcher       │  ✅ CALLBACK (multi-job)
  │ recommender              │  │ (2nd run)         │
  └──────────┬──────────────┘  └──────────┬────────┘
             │                           │
             ▼                           │
  ┌─────────────────────┐               │
  │ assessment_          │               │
  │ recommender          │               │
  └──────────┬──────────┘               │
             │                           │
             └───────────┬───────────────┘
                         │ (both → assessment_validator; fan-in guard skips until both ready)
                         ▼
              ┌──────────────────────┐
              │ assessment_validator  │  ✅ CALLBACK
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ end                  │  (final aggregated callback)
              └──────────────────────┘
```

---

## Nodes That Send Callbacks (in execution order)

| # | Node | Sends Callback |
|---|------|----------------|
| 1 | `validate_resume` | ✅ Yes |
| 2 | `groq_resume_parser` | ✅ Yes |
| 3 | `interest_filler` | ✅ Yes |
| 4 | `resume_summary` | ❌ No (passthrough) |
| 5 | `skill_proficiency_analyzer` | ✅ Yes |
| 6 | `resume_scorer` | ✅ Yes |
| 7 | `resume_assembler` | ❌ No (passthrough) |
| 8 | `job_matcher_preprocessor` (1st) | ❌ No (preprocessor) |
| 9 | `job_matcher` (1st – single job) | ✅ Yes |
| 10 | `career_advisor_entry` | ❌ No (passthrough) |
| 11 | `enhanced_role_fit` | ✅ Yes |
| 12 | `career_flow_fan_out` | ❌ No (passthrough) |
| 13 | `career_advisor` | ✅ Yes |
| 14 | `job_matcher_preprocessor` (2nd) | ❌ No (preprocessor) |
| 15 | `market_and_course_recommender` | ✅ Yes |
| 16 | `assessment_recommender` | ✅ Yes |
| 17 | `job_matcher` (2nd – multi-job) | ✅ Yes |
| 18 | `assessment_validator` | ✅ Yes (fan-in guard skips until both branches complete) |

---

## Summary: Callback Nodes (12 total)

1. **validate_resume** – resume validation result  
2. **groq_resume_parser** – parsed structured resume  
3. **interest_filler** – user interests  
4. **skill_proficiency_analyzer** – skill analysis  
5. **resume_scorer** – resume score (runs before resume_assembler)  
6. **job_matcher** (1st) – single-job match  
7. **enhanced_role_fit** – role fit analysis  
8. **career_advisor** – career advice  
9. **market_and_course_recommender** – market/course recommendations  
10. **assessment_recommender** – assessment plan  
11. **job_matcher** (2nd) – multi-job matches (`matched_jobs`, `top_matches`)  
12. **assessment_validator** – final validation and merged result  

Plus the final aggregated callback from the `end` node (full state with `candidate_job_match_result`, `matched_jobs`, `top_matches`, etc.).

**Note:** After job_matcher (1st), the flow is unified with normal resume: career_advisor_entry → enhanced_role_fit → career_flow_fan_out → [career_advisor + job_matcher_preprocessor] (parallel) → assessment_validator.
