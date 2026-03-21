"""
Unified matching logic shared by ranker and job_matcher.

All three agents use the same:
- ORDERED_ANALYSIS_STEPS_UNIFIED (neutral phrasing for consistent scores across flows)
- Scoring Guidelines (0-100 scale: 85+ excellent, 75-84 good, 60-74 moderate, below 60 poor)
- Critical rules for skills, education, certifications

This ensures consistent candidate-job matching regardless of which agent runs the analysis.
"""

# ---------------------------------------------------------------------------
# ORDERED ANALYSIS STEPS (same sequence for all agents)
# UNIFIED: Neutral phrasing for consistent scoring across ranker and job_matcher.
# CANDIDATE_CENTRIC / JOB_CENTRIC: Kept for backward compatibility; prefer UNIFIED for score consistency.
# ---------------------------------------------------------------------------

ORDERED_ANALYSIS_STEPS_UNIFIED = """
   **ORDERED ANALYSIS STEPS** (follow this sequence):
   1. **Domain fit**: Evaluate alignment between the candidate's professional domain and the job's domain. Cross-domain should score lower unless skills strongly overlap.
   2. **Domain-relevant experience**: Focus on experience IN THIS JOB'S DOMAIN, not just total years. Check the job's experience requirement (e.g., "3-5 years", "Senior", "5+ years"). If the candidate's relevant experience is far below the minimum (e.g., <1 year vs 3+ required), this is a MAJOR GAP that must significantly reduce the score. For Senior/Lead roles, also expect evidence of ownership, mentorship, or technical leadership—lack of these indicators should reduce the score.
   3. **Skills match**: 
      - **CRITICAL**: Must-have skills (if specified) - candidate MUST have most/all for high score (75+)
      - Additional required skills (from JD parsing)
      - Preferred skills (bonus points)
      - Consider certifications as skill evidence
   4. **Education and certifications**: Compare candidate's education with job requirements. Consider relevant certifications.
   5. **Final match_score**: Combine all factors with appropriate weights. Return an integer 0-100."""

ORDERED_ANALYSIS_STEPS_CANDIDATE_CENTRIC = """
   **ORDERED ANALYSIS STEPS** (follow this sequence):
   1. **Domain fit**: Does the candidate's professional domain align with this job's domain? Cross-domain candidates should score lower unless skills strongly overlap.
   2. **Domain-relevant experience**: Focus on experience IN THIS JOB'S DOMAIN, not just total years. Check the job's experience requirement (e.g., "3-5 years", "Senior", "5+ years"). If the candidate's relevant experience is far below the minimum (e.g., <1 year vs 3+ required), this is a MAJOR GAP that must significantly reduce the score. For Senior/Lead roles, also expect evidence of ownership, mentorship, or technical leadership—lack of these indicators should reduce the score.
   3. **Skills match**: 
      - **CRITICAL**: Must-have skills (if specified) - candidate MUST have most/all for high score
      - Additional required skills (from JD parsing)
      - Preferred skills (bonus points)
      - Consider certifications as skill evidence
   4. **Education and certifications**: Compare candidate's education with job requirements. Consider relevant certifications.
   5. **Final match_score**: Combine all factors with appropriate weights. Return an integer 0-100."""

ORDERED_ANALYSIS_STEPS_JOB_CENTRIC = """
   **ORDERED ANALYSIS STEPS** (follow this sequence for EACH job):
   1. **Domain fit**: Does this job's domain align with the candidate's professional domain? Cross-domain jobs should score lower unless skills strongly overlap.
   2. **Domain-relevant experience**: Focus on experience IN THIS JOB'S DOMAIN, not just total years. Check the job's experience requirement (e.g., "3-5 years", "Senior", "5+ years"). If the candidate's relevant experience is far below the minimum (e.g., <1 year vs 3+ required), this is a MAJOR GAP that must significantly reduce the score. For Senior/Lead roles, also expect evidence of ownership, mentorship, or technical leadership—lack of these indicators should reduce the score.
   3. **Skills match**: 
      - **CRITICAL**: Must-have skills (if specified) - candidate MUST have most/all for 75+
      - Additional required skills (from JD parsing)
      - Preferred skills (bonus points)
      - Consider certifications as skill evidence
   4. **Education and certifications**: Compare candidate's education with job requirements. Consider relevant certifications.
   5. **Final match_score**: Combine all factors with appropriate weights. Return an integer 0-100."""

# ---------------------------------------------------------------------------
# SCORING GUIDELINES (0-100 scale) - identical for all agents
# ---------------------------------------------------------------------------

SCORING_GUIDELINES_0_100 = """
   **Scoring Guidelines (0-100 scale, unified across ranker and job_matcher):**
   - 85-100: Excellent match - candidate has most required skills and strong domain alignment
   - 75-84: Good match - candidate has key skills and domain alignment
   - 60-74: Moderate match - some skills but significant gaps
   - Below 60: Poor match
   - If must-have skills are specified: Candidate MUST have most/all must-have skills for 75+
   - Missing must-have skills significantly reduces score
   - Domain mismatch should reduce score significantly
   - Preferred skills add bonus points but are not required
   - match_score MUST reflect ALL factors: domain fit, domain-relevant experience, required skills, preferred skills, education, and certifications. No single factor alone determines the score.
   - EXPERIENCE GATE (seniority-aware):
     - Senior/Lead/Principal/Staff roles: If candidate has <50% of required experience, cap score at 74 (moderate). Experience is critical at this level.
     - Mid-level roles: If candidate has <50% of required experience but strong skills (≥70% match), allow up to 79 (good). Skills partially compensate.
     - Junior/Entry roles or no seniority specified: No hard experience cap. Weight skills more heavily than experience.
   - SKILL COVERAGE GATE: If skill match is below 50% of required skills, cap the score at 74 (moderate) maximum.
   - CONSISTENCY RULE: match_score MUST align with positive_rationale and negative_rationale. If negative_rationale describes a "primary gap", "major gap", or "significant shortfall", the score MUST be in the moderate (60-74) or poor (<60) range—never excellent (85+)."""

# ---------------------------------------------------------------------------
# CRITICAL RULES (skills, education, certifications) - same for all agents
# ---------------------------------------------------------------------------

CRITICAL_RULES = """
CRITICAL RULES:
- ⚠️ PRIORITY MATCHING: If "Must-Have Skills" are specified, prioritize matching those first
- ⚠️ ONLY return skills that appear in the "Required Skills" list above
- ⚠️ DO NOT add any skills that are not in the required skills list
- ⚠️ DO NOT include education degrees/majors in skills_matched - education is matched separately
- Preferred skills are bonus points but not required
- Be generous with matching - include variations (e.g., "React" matches "React.js", "ReactJS", "React Native")
- Consider related skills (e.g., if "React" is required and candidate has "React.js", include "React" in matched)
- Consider the ENTIRE resume: skills section, projects, certifications, work experience
- Weight projects and achievements heavily - they demonstrate skills in action
- ⚠️ EDUCATION MATCHING (separate from skills):
  - Compare candidate's education from the resume with job's "Education Required" field
  - Match flexibly: "B.Com" = "BCom" = "Bachelor of Commerce", "M.Com" = "MCom" = "Master of Commerce"
  - If major/field is specified, check if candidate's major matches (e.g., "Computer Science", "Commerce", "Taxation")
  - Consider education match when calculating match_score (e.g., if job requires "B.Com" and candidate has "B.Com", this is a positive factor)
- ⚠️ CERTIFICATIONS: If the candidate has certifications listed, treat them as a strength when relevant. Do NOT say "no certifications" or "no relevant certifications" if the candidate has any.
- Use the EXACT skill name from the required skills list in your response
- match_score: integer 0-100 (85+ excellent, 75-84 good, 60-74 moderate, below 60 poor)
- ⚠️ SCORE CAPS (seniority-aware experience + universal skill gate):
  - SKILL GAP: If skill match is <50% of required skills, score cannot exceed 74 (regardless of seniority).
  - EXPERIENCE GAP for Senior/Lead/Principal/Staff roles: If candidate has <50% of required experience, score cannot exceed 74.
  - EXPERIENCE GAP for Mid-level roles: If candidate has <50% of required experience but strong skills (≥70% match), score can go up to 79. Otherwise cap at 74.
  - EXPERIENCE GAP for Junior/Entry or unspecified seniority: No hard experience cap. Skills matter most—let skill match drive the score.
  - If BOTH experience AND skills are major gaps, score should be <60 (poor).
- ⚠️ SENIORITY: For jobs with "Senior", "Lead", "Principal", or "Staff" in the title, expect 4+ years relevant experience AND evidence of ownership/leadership. Missing these → reduce score by at least 15 points.
- ⚠️ RATIONALE-SCORE ALIGNMENT: Before finalizing, verify that your match_score is consistent with what you wrote in positive_rationale and negative_rationale. If you described a "primary gap" or "major concern", the score must reflect it (≤74)."""

# ---------------------------------------------------------------------------
# Skills matching instructions (for "skills_matched" / "preferred_skills_matched")
# ---------------------------------------------------------------------------

SKILLS_MATCH_INSTRUCTIONS = """1. **skills_matched**: Skills from the "Required Skills" list above that the candidate has (check skills, projects, certifications, experience)
   - ⚠️ PRIORITY: Must match skills from "Must-Have Skills" section first (if present)
   - ⚠️ Then consider "Additional Required Skills" (from JD parsing)
   - ⚠️ DO NOT include education degrees/majors in skills_matched
   - ⚠️ ONLY include technical/professional skills (e.g., "Python", "React", "AWS", "Agile")
   - Education will be checked separately below
2. **preferred_skills_matched**: Skills from the "Preferred Skills" list that the candidate has (bonus points)"""


def normalize_match_score_0_100_to_0_1(raw: float) -> float:
    """Convert LLM match_score to 0.0-1.0. Accepts 0-100 or already 0-1."""
    if not isinstance(raw, (int, float)):
        return 0.0
    v = float(raw)
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))
