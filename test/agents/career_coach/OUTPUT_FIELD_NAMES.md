# Career Coach Chatbot - Output Field Names Reference

## Standard Output Format

All function tools return data in a consistent format:

```python
{
    "status": "completed" | "error",
    "node": "<function_name>",
    "output": {
        # Main data fields (varies by function)
    },
    # Metadata fields (consistent across all functions)
    "processing_time": <float>,  # NOT processing_time_seconds
    "request_id": <string>,
    "analysis_method": <string>,  # e.g., "existing_data", "llm", "resume_analyzer"
    "confidence_score": <float>,  # 0.0 to 1.0
    "success": <boolean>,  # true/false
    "message": <string>,  # Optional human-readable message
    
    # Function-specific top-level fields (for backward compatibility)
    # These duplicate data from "output" for easy access
}
```

## Function-Specific Output Fields

### 1. `get_course_recommendations`
**Output fields:**
- `course_recommendations`: List of course objects
- `market_insights`: List of market insight objects
- `salary_trends`: Dict with current_level, next_level, future_expectations
- `career_paths`: List of career path objects
- `skill_demand_analysis`: Dict of skill demand levels
- `raw_skill_gap_analysis_output`: Full skill gap analysis

**Course object fields:**
- `course`: Course title (primary field name)
- `title`: Course title (fallback, also accepted)
- `platform`: Platform/provider name (primary field name)
- `provider`: Platform/provider name (fallback, also accepted)
- `url`: Course URL
- `difficulty`: "Easy" | "Medium" | "Hard" | "All Levels"
- `duration`: Duration string (e.g., "6 weeks")
- `relevance_score`: Float (0.0 to 1.0)
- `description`: Course description
- `target_skill`: Target skill name
- `type`: "course" | "book" | "paper" | "video" | "blog" | "webpage"
- `author`: Author name (for books/papers)

### 2. `get_assessment_recommendations`
**Output fields:**
- `assessment_plan`: List of assessment objects
- `assessment_needs`: Dict with assessment needs analysis

**Assessment object fields:**
- `topic`: Assessment topic name
- `difficulty`: "easy" | "medium" | "hard" | "expert"
- `assessment_time_minutes`: Integer (assessment duration)
- `status`: "pending" | "completed"
- `assessment_id`: String (if assessment exists)
- `completed_at`: ISO date string (if completed)
- `score`: Float (if completed)

### 3. `create_career_goal` / `get_career_goals` / `update_career_goal`
**Output fields:**
- `goals`: List of goal objects
- `total_goals`: Integer

**Goal object fields:**
- `goal_id`: String (unique identifier)
- `title`: String (goal title)
- `description`: String (goal description)
- `status`: "active" | "completed" | "paused" | "cancelled"
- `progress`: Float (0.0 to 1.0)
- `priority`: "low" | "medium" | "high"
- `target_date`: ISO date string (YYYY-MM-DD)
- `created_at`: ISO datetime string
- `updated_at`: ISO datetime string
- `milestones`: List of milestone objects
- `tasks`: List of task objects

**Milestone object fields:**
- `milestone_id`: String
- `title`: String
- `due_date`: ISO date string
- `status`: "not_started" | "in_progress" | "completed"
- `progress`: Float (0.0 to 1.0)
- `tasks`: List of task objects

**Task object fields:**
- `task_id`: String
- `title`: String
- `status`: "not_started" | "in_progress" | "completed"
- `priority`: "low" | "medium" | "high"
- `due_date`: ISO date string
- `effort_hours`: Float (estimated hours)
- `difficulty`: "easy" | "medium" | "hard" | "expert"
- `type`: "course" | "assessment" | "project" | "study"
- `notes`: String
- `milestone_id`: String (if attached to milestone)

### 4. `identify_relevant_jobs`
**Output fields:**
- `top_matches`: List of job match objects
- `total_matches_found`: Integer
- `matching_method`: String
- `job_matcher_status`: "success" | "fallback" | "error"

**Job match object fields:**
- `rank`: Integer
- `job_id`: String
- `job_title`: String
- `company`: String
- `location`: String
- `match_score`: Float (0.0 to 1.0)
- `skill_match_percentage`: Float (0.0 to 100.0)
- `skill_match_count`: Integer
- `skills_matched`: List of strings
- `skills_unmatched`: List of strings
- `rationale`: String (match explanation)
- `vector_similarity`: Float

### 5. `get_resume_optimization_suggestions` (NEW)
**Output fields:**
- `suggestions`: List of suggestion objects
- `total_suggestions`: Integer
- `high_priority_count`: Integer
- `resume_analysis`: Dict with analysis metrics

**Suggestion object fields:**
- `category`: "skills" | "achievements" | "ats_optimization" | "education" | "projects" | "summary" | "contact"
- `priority`: "high" | "medium" | "low"
- `issue`: String (description of the issue)
- `suggestion`: String (recommendation text)
- `action`: String (actionable step)

**Resume analysis fields:**
- `has_skills`: Boolean
- `skills_count`: Integer
- `experience_count`: Integer
- `quantified_experience_count`: Integer
- `has_education`: Boolean
- `has_projects`: Boolean
- `has_summary`: Boolean

### 6. `get_salary_insights` (NEW)
**Output fields:**
- `salary_insights`: Dict with salary information
- `data_source`: String ("market_and_course_recommender")

**Salary insights fields:**
- `current_level`: Dict with:
  - `salary_range`: String (e.g., "₹X,00,000 - ₹Y,00,000")
  - `rationale`: String (explanation)
- `next_level`: Dict with:
  - `salary_range`: String
  - `rationale`: String
  - `growth_potential`: String ("20-40% increase")
- `future_expectations`: Dict with:
  - `salary_range`: String
  - `rationale`: String
  - `timeline`: String ("5+ years")
  - `growth_potential`: String ("50-100% increase")
- `market_insights`: List of market insight objects (top 3)

### 7. `get_learning_path` (NEW)
**Output fields:**
- `learning_path`: Dict with learning path structure
- `summary`: Dict with summary metrics

**Learning path fields:**
- `topic`: String (learning path topic)
- `stages`: List of stage objects
- `total_estimated_duration`: String (e.g., "14 weeks")
- `total_courses`: Integer
- `total_assessments`: Integer

**Stage object fields:**
- `stage`: Integer (1, 2, 3)
- `level`: "beginner" | "intermediate" | "advanced"
- `name`: String (stage name)
- `description`: String
- `courses`: List of course objects (top 3)
- `assessments`: List of assessment objects (top 2)
- `estimated_duration`: String (e.g., "2-4 weeks")
- `prerequisites`: String (optional)
- `next_stage`: String (optional)

**Summary fields:**
- `total_stages`: Integer
- `total_courses`: Integer
- `total_assessments`: Integer
- `estimated_duration`: String

## Field Name Consistency Rules

1. **Processing Time:**
   - Always use `processing_time` (NOT `processing_time_seconds`)
   - Type: Float (seconds)

2. **Error Responses:**
   - Always include `node` field
   - Always include `status: "error"`
   - Always include `error` field with error message
   - Always include `processing_time`
   - Always include `request_id` (if available)
   - Always include `success: false`

3. **Success Responses:**
   - Always include `status: "completed"`
   - Always include `node` field
   - Always include `output` dict with main data
   - Always include `processing_time`
   - Always include `request_id`
   - Always include `analysis_method`
   - Always include `confidence_score` (default: 0.8)
   - Always include `success: true`
   - Optionally include `message` for user-friendly feedback

4. **Course Field Names:**
   - Primary: `course` (title), `platform` (provider)
   - Fallback: `title` (also accepted), `provider` (also accepted)
   - Code should handle both for compatibility

5. **Date Formats:**
   - All dates: ISO format (YYYY-MM-DD)
   - All datetimes: ISO format (YYYY-MM-DDTHH:MM:SS)

## Missing Fields Check

### ✅ All functions now include:
- `status` (completed/error)
- `node` (function name)
- `output` (main data)
- `processing_time` (NOT processing_time_seconds)
- `request_id`
- `analysis_method`
- `confidence_score`
- `success` (boolean)
- `message` (optional, for user feedback)

### ✅ Error cases include:
- `node` field
- `error` field
- `processing_time`
- `request_id`
- `success: false`

## Notes

- All outputs match the format of native agents (market_and_course_recommender, assessment_recommender, job_matcher)
- Field names are consistent across all functions
- Backward compatibility fields are included where needed
- UI can render outputs without transformation
