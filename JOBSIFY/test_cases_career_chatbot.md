# Career Coach Chatbot - Test Cases

## Test Case Categories

### 1. Resume-Related Questions (NEW FEATURE)
**Purpose:** Verify chatbot can answer questions about candidate's resume using the RESUME DETAIL section.

#### TC-RES-001: List Skills
**User Message:** "What skills do I have?"
**Expected Behavior:**
- Chatbot lists actual skills from resume (e.g., "Python, SQL, React, JavaScript, ...")
- Should NOT just say "You have X skills" - should list them
- Should reference skills naturally in response

#### TC-RES-002: Summarize Experience
**User Message:** "Can you summarize my work experience?"
**Expected Behavior:**
- Lists job titles, companies, dates from resume
- Mentions key responsibilities if available
- Conversational tone, not just bullet dump

#### TC-RES-003: Education Background
**User Message:** "What's my educational background?"
**Expected Behavior:**
- Lists degrees, majors, universities
- Includes graduation years if available
- Natural language response

#### TC-RES-004: Projects Inquiry
**User Message:** "What projects have I worked on?"
**Expected Behavior:**
- Lists project names from resume
- Includes brief descriptions
- May suggest how to highlight them in interviews

#### TC-RES-005: Specific Role Details
**User Message:** "What did I do at [Company Name]?"
**Expected Behavior:**
- Finds matching company in experience
- Describes role, dates, responsibilities
- If not found, politely says so

#### TC-RES-006: Skills Count vs List
**User Message:** "How many skills do I have?"
**Expected Behavior:**
- Provides count AND may list top skills
- Should reference actual skills, not just number

#### TC-RES-007: Resume Summary Request
**User Message:** "Give me a summary of my resume"
**Expected Behavior:**
- Combines name, experience, education, key skills
- Natural paragraph format
- Highlights strengths

---

### 2. Course Recommendations

#### TC-COURSE-001: General Course Request
**User Message:** "Show me courses to improve my skills"
**Expected Behavior:**
- Calls `get_course_recommendations` tool
- Returns personalized courses from existing data (if available)
- Lists course titles, providers, URLs
- Explains why courses are relevant

#### TC-COURSE-002: Topic-Specific Course Request
**User Message:** "I want to learn Python, can you recommend courses?"
**Expected Behavior:**
- Calls `get_course_recommendations` with `topics=["Python"]` and `courses_only=true`
- Uses fast path (no market analysis)
- Returns Python-specific courses
- Conversational response

#### TC-COURSE-003: Multiple Topics
**User Message:** "Find courses for AWS and Docker"
**Expected Behavior:**
- Calls tool with `topics=["AWS", "Docker"]`
- Returns courses for both topics
- May suggest learning order

#### TC-COURSE-004: Refresh Request
**User Message:** "Show me new course recommendations"
**Expected Behavior:**
- Calls tool with `force_refresh=true`
- Triggers full market_and_course_recommender agent
- Returns fresh recommendations

#### TC-COURSE-005: General Learning Question (No Tool Call)
**User Message:** "How can I learn machine learning?"
**Expected Behavior:**
- Does NOT call tool (general question)
- Answers from knowledge/context
- May mention courses in context if available
- Provides learning path advice

---

### 3. Assessment Recommendations

#### TC-ASSESS-001: General Assessment Request
**User Message:** "I want to take some assessments"
**Expected Behavior:**
- Calls `get_assessment_recommendations` with `suggest_new_topics=true`
- Returns assessment plan with topics
- Explains what each assessment covers
- May create tasks for assessments

#### TC-ASSESS-002: Topic-Specific Assessment
**User Message:** "I want to test my Python skills"
**Expected Behavior:**
- Calls tool with `topics=["Python"]`
- Creates assessment plan for Python
- Returns assessment details (difficulty, time, topics)

#### TC-ASSESS-003: Multiple Topics
**User Message:** "Assessments for React and Node.js please"
**Expected Behavior:**
- Calls tool with `topics=["React", "Node.js"]`
- Creates assessments for both topics
- May suggest taking them in order

#### TC-ASSESS-004: Suggest New Topics
**User Message:** "What assessments should I take?"
**Expected Behavior:**
- Calls tool with `suggest_new_topics=true`
- Analyzes profile, skill gaps, career paths
- Suggests 3-5 relevant topics
- Only creates assessments for topics without existing plans

#### TC-ASSESS-005: Refresh Assessments
**User Message:** "Give me updated assessment recommendations"
**Expected Behavior:**
- Calls tool with `force_refresh=true`
- Regenerates assessment plan
- May remove completed assessments

---

### 4. Job Matching

#### TC-JOB-001: Find Jobs (Explicit Search Request)
**User Message:** "What jobs are available for me?"
**Expected Behavior:**
- Calls `identify_relevant_jobs` tool
- Returns top 3-4 job matches (chatbot limit)
- Includes job titles, companies, match reasons
- Conversational presentation

#### TC-JOB-002: Search with Query
**User Message:** "Find me software engineer jobs in Bangalore"
**Expected Behavior:**
- Calls tool with `search_query="software engineer Bangalore"`
- Returns filtered results
- Top 3-4 matches only

#### TC-JOB-003: General Career Question (No Tool Call)
**User Message:** "What careers should I pursue?"
**Expected Behavior:**
- Does NOT call tool
- Uses `career_paths` from context
- Provides advice based on profile
- Conversational guidance

#### TC-JOB-004: Role Suitability (No Tool Call)
**User Message:** "What roles am I qualified for?"
**Expected Behavior:**
- Does NOT call tool
- Uses `role_fit_suggestions` from context
- Explains qualifications
- May reference skills/experience

#### TC-JOB-005: Job Search with Location
**User Message:** "Show me jobs in Mumbai"
**Expected Behavior:**
- Calls tool with location filter
- Returns location-specific matches
- Top 3-4 results

---

### 5. Goal/Milestone/Task Management

#### TC-GOAL-001: Create Goal Explicitly
**User Message:** "I want to become a data scientist. Set a goal for me."
**Expected Behavior:**
- Calls `create_career_goal` tool
- Creates goal with title, description
- May suggest milestones (courses, assessments, projects)
- Sets realistic target_date

#### TC-GOAL-002: Create Goal After Course Recommendation
**User Message:** "I want to learn Python" → [Chatbot recommends courses] → "Yes, create a goal for this"
**Expected Behavior:**
- After course recommendations, offers to create goal
- If user agrees, calls `create_career_goal`
- Links recommended courses as tasks
- Sets milestones (e.g., "Complete Python basics course")

#### TC-GOAL-003: View Goals
**User Message:** "What are my current goals?"
**Expected Behavior:**
- Calls `get_career_goals` tool
- Lists all active goals
- Shows progress, milestones, tasks
- Conversational summary

#### TC-GOAL-004: Update Goal Progress
**User Message:** "I completed the Python course, update my goal"
**Expected Behavior:**
- Calls `update_career_goal` tool
- Marks task as completed
- Updates goal progress
- Congratulates user

#### TC-GOAL-005: Add Task to Goal
**User Message:** "Add a task to practice Python daily"
**Expected Behavior:**
- Calls `update_career_goal` with `add_tasks`
- Creates task with type, difficulty, effort_hours
- Sets realistic due_date
- Confirms addition

#### TC-GOAL-006: Goal Status Check (Proactive)
**User Message:** "How am I doing on my goals?"
**Expected Behavior:**
- Calls `get_career_goals` tool
- Reviews all goals, milestones, tasks
- Identifies overdue or stalled items
- Gently nudges on pending tasks
- Suggests next steps

#### TC-GOAL-007: Intelligent Date Setting
**User Message:** "I want to learn React by next month"
**Expected Behavior:**
- Creates goal with `target_date` = next month
- Breaks down into tasks with due dates
- Estimates effort based on task type
- Sets realistic milestones

#### TC-GOAL-008: Task Completion
**User Message:** "Mark the 'Complete Python basics' task as done"
**Expected Behavior:**
- Calls `update_career_goal` with `complete_tasks=["task_id"]`
- Updates task status
- Updates milestone progress if applicable
- Updates goal progress

---

### 6. Career Advice (No Tool Calls)

#### TC-ADVICE-001: Career Path Question
**User Message:** "What career paths are suitable for me?"
**Expected Behavior:**
- Does NOT call any tool
- Uses `career_paths` from context
- Explains each path
- References user's skills/experience
- May suggest next steps

#### TC-ADVICE-002: Skill Gap Inquiry
**User Message:** "What skills am I missing?"
**Expected Behavior:**
- Uses `skill_gaps` from context
- Lists missing skills
- Explains why they're important
- May suggest courses/assessments

#### TC-ADVICE-003: Improvement Recommendations
**User Message:** "How can I improve my profile?"
**Expected Behavior:**
- Uses `improvement_recommendations` from context
- Provides actionable advice
- May reference courses, assessments, goals
- Personalized suggestions

#### TC-ADVICE-004: Salary Expectations
**User Message:** "What salary should I expect?"
**Expected Behavior:**
- Uses `market_insights` from context (if available)
- References experience level
- May mention current_level, next_level ranges
- Explains factors affecting salary

---

### 7. Proactive Mentoring

#### TC-MENTOR-001: Check on Pending Assessments
**User Message:** "Hi" (after assessments were recommended but not completed)
**Expected Behavior:**
- Loads context with `pending_assessments`
- Gently mentions pending assessments
- Asks if user wants to take them
- Offers to help get started

#### TC-MENTOR-002: Nudge on Overdue Tasks
**User Message:** "What should I do next?" (with overdue tasks)
**Expected Behavior:**
- Reviews goals/tasks from context
- Identifies overdue items
- Gently reminds about them
- Suggests prioritizing overdue tasks
- Offers to help break them down

#### TC-MENTOR-003: Suggest Next Steps
**User Message:** "I'm not sure what to focus on"
**Expected Behavior:**
- Analyzes profile, goals, skill gaps
- Suggests 2-3 priority actions
- May propose creating a goal
- References career paths

#### TC-MENTOR-004: Goal Progress Check
**User Message:** "How am I progressing?" (with active goals)
**Expected Behavior:**
- Calls `get_career_goals` tool
- Reviews progress on all goals
- Highlights completed milestones
- Identifies next tasks
- Encourages continued progress

---

### 8. Conversation Flow & Follow-ups

#### TC-CONV-001: Clickable Follow-up Questions
**User Message:** "Show me Python courses"
**Expected Behavior:**
- Returns course recommendations
- Generates 2-4 clickable follow-up questions
- Questions phrased as USER asking (e.g., "Can you show me assessments for Python?")
- NOT chatbot offering (e.g., NOT "Would you like assessments?")

#### TC-CONV-002: Multi-Turn Conversation
**User Message Sequence:**
1. "What skills do I have?"
2. "Which of these should I focus on?"
3. "Show me courses for [selected skill]"
4. "Create a goal for this"
**Expected Behavior:**
- Maintains context across turns
- References previous conversation
- Builds on earlier responses
- Natural flow

#### TC-CONV-003: Context Switching
**User Message:** "Actually, I'm more interested in data science now"
**Expected Behavior:**
- Acknowledges change in interest
- Updates recommendations accordingly
- May suggest new goals
- Doesn't force old recommendations

#### TC-CONV-004: Clarification Request
**User Message:** "What do you mean by that?"
**Expected Behavior:**
- References previous response
- Provides clearer explanation
- Uses simpler language
- May give examples

---

### 9. Edge Cases & Error Handling

#### TC-EDGE-001: No Resume Data
**User Message:** "What skills do I have?" (no resume uploaded)
**Expected Behavior:**
- Gracefully handles missing resume
- Explains that resume is needed
- Suggests uploading resume
- Doesn't crash or show errors

#### TC-EDGE-002: Empty Skills List
**User Message:** "What skills do I have?" (resume has no skills)
**Expected Behavior:**
- Acknowledges no skills found
- Suggests ways to identify skills
- May recommend skill assessment
- Helpful, not dismissive

#### TC-EDGE-003: Ambiguous Request
**User Message:** "Help me"
**Expected Behavior:**
- Asks clarifying questions
- Offers options (courses, assessments, jobs, goals)
- Doesn't assume intent
- Friendly and helpful

#### TC-EDGE-004: Tool Call Failure
**User Message:** "Show me courses" (tool fails)
**Expected Behavior:**
- Handles error gracefully
- Explains issue in user-friendly way
- Suggests alternatives
- Doesn't expose technical errors

#### TC-EDGE-005: Very Long Conversation
**User Message:** 20+ message exchange
**Expected Behavior:**
- Maintains context
- Doesn't repeat information unnecessarily
- References earlier conversation
- Remains responsive

#### TC-EDGE-006: Rapid Fire Questions
**User Message:** Multiple questions in quick succession
**Expected Behavior:**
- Handles each independently
- Maintains session state
- Doesn't mix contexts
- Clear responses

---

### 10. Integration & Data Consistency

#### TC-INT-001: Course Recommendations Format
**User Message:** "Show me courses"
**Expected Behavior:**
- Returns format matching native `market_and_course_recommender` output
- Includes: title, url, provider, description, relevance_score
- UI can render without transformation

#### TC-INT-002: Assessment Recommendations Format
**User Message:** "Recommend assessments"
**Expected Behavior:**
- Returns format matching native `assessment_recommender` output
- Includes: topic, difficulty, time, status, assessment_id
- UI can render without transformation

#### TC-INT-003: Job Matches Format
**User Message:** "Find jobs for me"
**Expected Behavior:**
- Returns top 3-4 matches (chatbot limit)
- Format matches native `job_matcher` output
- Includes: job_title, company, location, match_score, rationale
- UI can render without transformation

#### TC-INT-004: Goal Format Consistency
**User Message:** "Create a goal to learn Python"
**Expected Behavior:**
- Returns goal in standard format
- Includes: goal_id, title, description, milestones, tasks, status, progress
- Matches format used by goal management system
- UI can render without transformation

---

### 11. Streaming & Status Updates

#### TC-STREAM-001: Streaming Response
**User Message:** "Tell me about my career options"
**Expected Behavior:**
- Response streams in chunks
- Frontend receives SSE events
- User sees text appearing in real-time
- No long wait before response starts

#### TC-STREAM-002: Tool Execution Status
**User Message:** "Show me courses" (triggers tool call)
**Expected Behavior:**
- Sends `tool_start` status event before tool execution
- Frontend shows loading indicator
- Sends `tool_complete` status event after tool execution
- User knows system is working

#### TC-STREAM-003: Long Tool Execution
**User Message:** "Find me jobs" (job_matcher takes 10+ seconds)
**Expected Behavior:**
- Status updates keep user informed
- Streaming continues after tool completes
- No silent gaps
- User sees progress

---

### 12. Intelligent Tool Parameterization

#### TC-PARAM-001: Topic Extraction
**User Message:** "I want courses for Python and machine learning"
**Expected Behavior:**
- Extracts topics: ["Python", "machine learning"]
- Calls `get_course_recommendations` with `topics=["Python", "machine learning"]`
- Sets `courses_only=true` for fast path
- Doesn't trigger full market analysis

#### TC-PARAM-002: Force Refresh Detection
**User Message:** "Give me fresh course recommendations"
**Expected Behavior:**
- Detects "fresh" keyword
- Sets `force_refresh=true`
- Triggers full agent run
- Returns new recommendations

#### TC-PARAM-003: Job Search Query Extraction
**User Message:** "Find software engineer jobs in remote locations"
**Expected Behavior:**
- Extracts search_query: "software engineer remote"
- Calls `identify_relevant_jobs` with search_query
- Returns filtered results

#### TC-PARAM-004: Assessment Topic Extraction
**User Message:** "I want to test my React and TypeScript skills"
**Expected Behavior:**
- Extracts topics: ["React", "TypeScript"]
- Calls `get_assessment_recommendations` with `topics=["React", "TypeScript"]`
- Creates assessments for both topics

---

### 13. Resume Optimization Suggestions (NEW FEATURE)

#### TC-RESUME-OPT-001: General Resume Improvement Request
**User Message:** "How can I improve my resume?"
**Expected Behavior:**
- Calls `get_resume_optimization_suggestions` tool
- Returns prioritized suggestions (high/medium/low)
- Includes specific issues and actionable recommendations
- Covers: skills, achievements, ATS keywords, education, projects, summary, contact

#### TC-RESUME-OPT-002: ATS Optimization Request
**User Message:** "Can you help me optimize my resume for ATS?"
**Expected Behavior:**
- Calls `get_resume_optimization_suggestions` tool
- Returns suggestions focused on ATS compatibility
- Includes keyword optimization recommendations
- Mentions industry-standard keywords

#### TC-RESUME-OPT-003: Resume Feedback Request
**User Message:** "Give me feedback on my resume"
**Expected Behavior:**
- Calls `get_resume_optimization_suggestions` tool
- Returns comprehensive analysis
- Identifies missing sections
- Provides specific improvement actions

#### TC-RESUME-OPT-004: Achievement Quantification
**User Message:** "My resume lacks impact, how can I improve it?"
**Expected Behavior:**
- Calls `get_resume_optimization_suggestions` tool
- Highlights missing quantified achievements
- Suggests adding metrics, percentages, numbers
- Provides examples of quantified statements

#### TC-RESUME-OPT-005: Skills Section Enhancement
**User Message:** "Should I add more skills to my resume?"
**Expected Behavior:**
- Calls `get_resume_optimization_suggestions` tool
- Analyzes current skills count
- Recommends target number of skills (10-15)
- Suggests relevant skills based on target roles

#### TC-RESUME-OPT-006: Missing Resume Data
**User Message:** "How can I improve my resume?" (no resume uploaded)
**Expected Behavior:**
- Calls `get_resume_optimization_suggestions` tool
- Returns error message gracefully
- Suggests uploading resume first
- Doesn't crash or show technical errors

---

### 14. Salary Insights (NEW FEATURE)

#### TC-SALARY-001: General Salary Question
**User Message:** "What salary should I expect?"
**Expected Behavior:**
- Calls `get_salary_insights` tool
- Returns current level, next level, and future expectations
- Includes salary ranges with rationales
- Mentions growth potential percentages

#### TC-SALARY-002: Current Level Salary
**User Message:** "How much should I be earning at my current level?"
**Expected Behavior:**
- Calls `get_salary_insights` tool
- Focuses on current_level salary range
- Provides rationale based on experience, role, skills
- References market conditions

#### TC-SALARY-003: Future Salary Expectations
**User Message:** "What salary can I expect in 5 years?"
**Expected Behavior:**
- Calls `get_salary_insights` tool
- Highlights future_expectations range
- Explains career path and skill development needed
- Mentions 50-100% growth potential

#### TC-SALARY-004: Next Level Salary
**User Message:** "What salary should I target for my next role?"
**Expected Behavior:**
- Calls `get_salary_insights` tool
- Emphasizes next_level salary range
- Explains skills/experience needed to achieve it
- Mentions 20-40% increase potential

#### TC-SALARY-005: Fresh Salary Data Request
**User Message:** "Give me updated salary insights"
**Expected Behavior:**
- Calls `get_salary_insights` with `force_refresh=true`
- Triggers fresh market analysis if needed
- Returns latest salary data
- May take longer but provides current market rates

#### TC-SALARY-006: No Salary Data Available
**User Message:** "What salary should I expect?" (no market analysis done)
**Expected Behavior:**
- Calls `get_salary_insights` tool
- Returns error message gracefully
- Suggests completing profile analysis first
- May trigger market_and_course_recommender if force_refresh

---

### 15. Learning Path Visualization (NEW FEATURE)

#### TC-LEARN-001: General Learning Path Request
**User Message:** "Show me a learning path"
**Expected Behavior:**
- Calls `get_learning_path` tool
- Returns structured path with stages (beginner → intermediate → advanced)
- Includes courses and assessments for each stage
- Shows estimated durations and prerequisites

#### TC-LEARN-002: Topic-Specific Learning Path
**User Message:** "Create a learning path for Python"
**Expected Behavior:**
- Calls `get_learning_path` with `topic="Python"`
- Filters courses and assessments for Python
- Organizes by difficulty level
- Returns Python-specific progression

#### TC-LEARN-003: Learning Roadmap Request
**User Message:** "I want a structured learning plan for data science"
**Expected Behavior:**
- Calls `get_learning_path` with `topic="data science"`
- Creates multi-stage learning path
- Links courses and assessments in logical order
- Provides timeline estimates

#### TC-LEARN-004: Skill Progression Path
**User Message:** "How should I progress in React development?"
**Expected Behavior:**
- Calls `get_learning_path` with `topic="React"`
- Returns beginner → intermediate → advanced stages
- Each stage includes relevant courses and assessments
- Shows prerequisites between stages

#### TC-LEARN-005: Learning Path with No Courses
**User Message:** "Show me a learning path" (no courses/assessments available)
**Expected Behavior:**
- Calls `get_learning_path` tool
- Returns empty or minimal path gracefully
- Suggests getting course recommendations first
- Doesn't crash or show errors

#### TC-LEARN-006: Multi-Topic Learning Path
**User Message:** "I want to learn both Python and machine learning, show me a path"
**Expected Behavior:**
- May call `get_learning_path` for each topic or combined
- Returns structured progression
- May suggest learning order (e.g., Python first, then ML)
- Provides integrated learning plan

---

## Test Execution Checklist

### Pre-conditions for Testing
- [ ] Candidate has uploaded resume (structured_resume in Chroma)
- [ ] Context aggregator can load resume data
- [ ] Some career analysis data exists (skill gaps, career paths)
- [ ] Chat session is active
- [ ] Streaming is enabled

### Test Execution
1. **Resume Questions:** Test TC-RES-001 through TC-RES-007
2. **Course Recommendations:** Test TC-COURSE-001 through TC-COURSE-005
3. **Assessment Recommendations:** Test TC-ASSESS-001 through TC-ASSESS-005
4. **Job Matching:** Test TC-JOB-001 through TC-JOB-005
5. **Goal Management:** Test TC-GOAL-001 through TC-GOAL-008
6. **Career Advice:** Test TC-ADVICE-001 through TC-ADVICE-004
7. **Proactive Mentoring:** Test TC-MENTOR-001 through TC-MENTOR-004
8. **Conversation Flow:** Test TC-CONV-001 through TC-CONV-004
9. **Edge Cases:** Test TC-EDGE-001 through TC-EDGE-006
10. **Integration:** Test TC-INT-001 through TC-INT-004
11. **Streaming:** Test TC-STREAM-001 through TC-STREAM-003
12. **Parameterization:** Test TC-PARAM-001 through TC-PARAM-004
13. **Resume Optimization:** Test TC-RESUME-OPT-001 through TC-RESUME-OPT-006
14. **Salary Insights:** Test TC-SALARY-001 through TC-SALARY-006
15. **Learning Path:** Test TC-LEARN-001 through TC-LEARN-006

### Success Criteria
- ✅ All tool calls use correct parameters
- ✅ Resume questions return actual data (not just counts)
- ✅ Follow-up questions are clickable (user-voiced)
- ✅ Streaming works without gaps
- ✅ Status updates appear during tool execution
- ✅ Output formats match native agent formats
- ✅ Error handling is graceful
- ✅ Conversation context is maintained
- ✅ Proactive mentoring works (nudges, suggestions)

---

## Sample Test Script

```python
# Example test script structure
test_cases = [
    {
        "id": "TC-RES-001",
        "name": "List Skills",
        "user_message": "What skills do I have?",
        "expected_tool_calls": [],
        "expected_content": ["Python", "SQL", "React"],  # Actual skill names
        "should_not_contain": ["X skills identified"]  # Not just count
    },
    {
        "id": "TC-COURSE-002",
        "name": "Topic-Specific Course Request",
        "user_message": "I want to learn Python, can you recommend courses?",
        "expected_tool_calls": [
            {
                "tool": "get_course_recommendations",
                "args": {"topics": ["Python"], "courses_only": True}
            }
        ],
        "expected_content": ["Python", "course", "Coursera", "Udemy"]
    },
    {
        "id": "TC-RESUME-OPT-001",
        "name": "General Resume Improvement Request",
        "user_message": "How can I improve my resume?",
        "expected_tool_calls": [
            {
                "tool": "get_resume_optimization_suggestions",
                "args": {"uid": "test_uid"}
            }
        ],
        "expected_content": ["suggestions", "priority", "high", "medium", "low"],
        "should_contain": ["action", "issue", "suggestion"]
    },
    {
        "id": "TC-SALARY-001",
        "name": "General Salary Question",
        "user_message": "What salary should I expect?",
        "expected_tool_calls": [
            {
                "tool": "get_salary_insights",
                "args": {"uid": "test_uid", "force_refresh": False}
            }
        ],
        "expected_content": ["current_level", "next_level", "future_expectations", "salary_range", "rationale"]
    },
    {
        "id": "TC-LEARN-001",
        "name": "General Learning Path Request",
        "user_message": "Show me a learning path",
        "expected_tool_calls": [
            {
                "tool": "get_learning_path",
                "args": {"uid": "test_uid"}
            }
        ],
        "expected_content": ["stages", "beginner", "intermediate", "advanced", "courses", "assessments", "estimated_duration"]
    },
    {
        "id": "TC-LEARN-002",
        "name": "Topic-Specific Learning Path",
        "user_message": "Create a learning path for Python",
        "expected_tool_calls": [
            {
                "tool": "get_learning_path",
                "args": {"uid": "test_uid", "topic": "Python"}
            }
        ],
        "expected_content": ["Python", "stages", "beginner", "intermediate", "advanced"]
    },
    # ... more test cases
]
```

---

## Notes

- **Resume Questions:** The chatbot should now have access to full resume detail (skills list, experience entries, education, projects) via the RESUME DETAIL section in the prompt.
- **Tool Calls:** Verify that tools are only called when explicitly requested, not for general questions.
- **Format Consistency:** All tool outputs should match native agent formats to avoid UI conflicts.
- **Streaming:** Verify SSE events are sent correctly and frontend receives them.
- **Proactive Mentoring:** Chatbot should check goal/task status and nudge appropriately.
- **Resume Optimization:** New feature provides actionable, prioritized suggestions for resume improvement (ATS, keywords, achievements, format).
- **Salary Insights:** New feature extracts salary data from market analysis (current, next level, future expectations) with rationales.
- **Learning Path:** New feature creates structured learning paths (beginner → intermediate → advanced) from courses and assessments, with topic filtering support.
