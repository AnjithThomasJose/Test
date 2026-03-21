# Growth Plan Page - Implementation Specification

## Overview

The **Growth Plan** page is a dedicated UI for career goal management that works **in sync with the AI Mentor**. Users can create goals from either the chatbot or this page, and all data stays synchronized.

---

## Table of Contents

1. [Core Concept](#1-core-concept)
2. [Data Flow & Sync Architecture](#2-data-flow--sync-architecture)
3. [Data Models](#3-data-models)
4. [API Endpoints](#4-api-endpoints)
5. [Plan Generation Agent](#5-plan-generation-agent)
6. [Recommended Goals](#6-recommended-goals)
7. [Task Types & Platform Integration](#7-task-types--platform-integration)
8. [UI/UX Wireframes](#8-uiux-wireframes)
9. [Implementation Phases](#9-implementation-phases)
10. [Test Cases](#10-test-cases)

---

## 1. Core Concept

### What It Does

- **Create Goals**: User types a career goal (e.g., "I want to become a Cloud Architect") OR selects from AI-recommended goals
- **AI Generates Plan**: System analyzes user profile and creates a complete plan with milestones and tasks
- **Track Progress**: Visual progress tracking with checkable tasks and milestone completion
- **Sync with Mentor**: Goals created in chatbot appear here; goals created here are known to the mentor

### Key Features

| Feature | Description |
|---------|-------------|
| **Goal Input** | Free-text input field for custom goals |
| **Recommended Goals** | AI-suggested goals based on profile, skills, and career paths |
| **Plan Generation** | Automatic milestone + task breakdown using AI |
| **Progress Tracking** | Visual progress bars, task checklists |
| **Task Linking** | Tasks link to courses, assessments, jobs, interviews |
| **Mentor Sync** | Bidirectional sync with AI Mentor chatbot |

---

## 2. Data Flow & Sync Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           JOBSIFY AI                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐                    ┌─────────────────┐           │
│   │   AI MENTOR     │◄──────────────────►│  GROWTH PLAN    │           │
│   │   (Chatbot)     │      REAL-TIME     │     PAGE        │           │
│   │                 │        SYNC        │                 │           │
│   │ Tools:          │                    │ Features:       │           │
│   │ - create_goal   │                    │ - Goal input    │           │
│   │ - get_goals     │                    │ - Recommended   │           │
│   │ - update_goal   │                    │ - Progress view │           │
│   └────────┬────────┘                    └────────┬────────┘           │
│            │                                      │                     │
│            │         SHARED DATA LAYER            │                     │
│            └──────────────┬───────────────────────┘                     │
│                           ▼                                             │
│            ┌─────────────────────────────┐                              │
│            │     ChromaDB / Firestore    │                              │
│            │     ─────────────────────   │                              │
│            │     goals collection        │                              │
│            │     (uid + session_id)      │                              │
│            └─────────────────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Sync Scenarios

| Scenario | Flow |
|----------|------|
| User creates goal in **Mentor** | Mentor → `create_career_goal` tool → DB → Page shows new goal |
| User creates goal in **Page** | Page → `/goals/generate-plan` API → DB → Mentor can reference it |
| User completes task in **Page** | Page → `/goals/.../task/.../complete` → DB → Mentor knows progress |
| User asks Mentor about progress | Mentor → `get_career_goals` tool → Shows current status |
| Mentor suggests next task | Mentor reads goals → Proactively nudges user |

---

## 3. Data Models

### 3.1 Goal Model

```json
{
  "goal_id": "uuid-v4",
  "goal": "Become a Cloud Solutions Architect",
  "description": "Transition from backend developer to cloud architecture role",
  "status": "in_progress",
  "progress": 0.45,
  "priority": "high",
  "target_date": "2026-09-01",
  "estimated_duration_months": 6,
  "created_at": "2026-01-23T10:00:00Z",
  "updated_at": "2026-01-23T15:30:00Z",
  "created_by": "mentor | page",
  "role_fit_percentage": 85,
  "milestones": [...],
  "metadata": {
    "source_career_path": "Cloud Solutions Architect",
    "skill_gaps_addressed": ["AWS", "Cloud Architecture", "System Design"]
  }
}
```

### 3.2 Milestone Model

```json
{
  "milestone_id": "uuid-v4",
  "title": "Build Foundation",
  "description": "Learn cloud fundamentals and basic certifications",
  "order": 1,
  "status": "in_progress",
  "progress": 0.67,
  "target_date": "2026-03-01",
  "estimated_duration_weeks": 4,
  "tasks": [...],
  "created_at": "2026-01-23T10:00:00Z",
  "completed_at": null
}
```

### 3.3 Task Model

```json
{
  "task_id": "uuid-v4",
  "title": "Complete AWS Cloud Practitioner course",
  "description": "Finish the foundational AWS course to understand cloud basics",
  "order": 1,
  "status": "completed",
  "priority": "high",
  "due_date": "2026-02-15",
  "estimated_hours": 20,
  "task_type": "course",
  "linked_resource": {
    "type": "course",
    "id": "course-aws-practitioner-123",
    "title": "AWS Cloud Practitioner Essentials",
    "url": "https://..."
  },
  "completion_criteria": "Score 80% or higher on course completion quiz",
  "created_at": "2026-01-23T10:00:00Z",
  "completed_at": "2026-02-10T14:00:00Z"
}
```

### 3.4 Status Enums

```python
class GoalStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

class TaskType(str, Enum):
    COURSE = "course"              # Links to course recommendations
    ASSESSMENT = "assessment"      # Links to skill assessments
    PROJECT = "project"            # External portfolio project
    CERTIFICATION = "certification" # External certification
    JOB_APPLICATION = "job_application"  # Links to jobs section
    RESUME_UPDATE = "resume_update"      # Links to resume analysis
    INTERVIEW_PREP = "interview_prep"    # Links to AI interview
    NETWORKING = "networking"      # Networking activities
    LEARNING = "learning"          # General learning (books, videos)
    CUSTOM = "custom"              # User-defined task
```

---

## 4. API Endpoints

### 4.1 Endpoint Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/goals/{uid}` | GET | List all goals with milestones and tasks |
| `/goals/{uid}/recommended` | GET | Get AI-recommended goals based on profile |
| `/goals/{uid}/generate-plan` | POST | Generate full plan from goal text |
| `/goals/{uid}` | POST | Create a goal (with pre-generated plan) |
| `/goals/{uid}/{goal_id}` | GET | Get single goal details |
| `/goals/{uid}/{goal_id}` | PUT | Update goal metadata |
| `/goals/{uid}/{goal_id}` | DELETE | Delete a goal |
| `/goals/{uid}/{goal_id}/milestone/{milestone_id}` | PUT | Update milestone |
| `/goals/{uid}/{goal_id}/task/{task_id}` | PUT | Update task |
| `/goals/{uid}/{goal_id}/task/{task_id}/complete` | POST | Mark task complete |

### 4.2 Endpoint Details

#### GET `/goals/{uid}`

List all goals for a user.

**Response:**
```json
{
  "success": true,
  "goals": [
    {
      "goal_id": "...",
      "goal": "Become a Cloud Solutions Architect",
      "status": "in_progress",
      "progress": 0.45,
      "priority": "high",
      "target_date": "2026-09-01",
      "milestones": [
        {
          "milestone_id": "...",
          "title": "Build Foundation",
          "status": "completed",
          "progress": 1.0,
          "tasks": [...]
        },
        ...
      ]
    }
  ],
  "total_goals": 2,
  "active_goals": 1,
  "completed_goals": 1
}
```

---

#### GET `/goals/{uid}/recommended`

Get AI-recommended goals based on user profile.

**Response:**
```json
{
  "success": true,
  "recommended_goals": [
    {
      "goal": "Transition to Cloud Solutions Architect",
      "description": "Based on your AWS experience and system design skills",
      "role_fit_percentage": 85,
      "estimated_duration_months": 6,
      "key_milestones": [
        "AWS Certification",
        "Cloud Architecture Projects",
        "System Design Mastery"
      ],
      "skill_gaps_to_close": ["Cloud Architecture Patterns", "Multi-cloud Strategy"],
      "source": "career_paths"
    },
    {
      "goal": "Become a DevOps Engineer",
      "description": "Your backend skills align well with DevOps practices",
      "role_fit_percentage": 78,
      "estimated_duration_months": 4,
      "key_milestones": [
        "CI/CD Mastery",
        "Container Orchestration",
        "Infrastructure as Code"
      ],
      "skill_gaps_to_close": ["Kubernetes", "Terraform"],
      "source": "career_paths"
    },
    {
      "goal": "Improve Python Proficiency",
      "description": "Your recent assessment showed room for improvement",
      "role_fit_percentage": null,
      "estimated_duration_months": 2,
      "key_milestones": [
        "Advanced Python Concepts",
        "Python Projects",
        "Assessment Score 80%+"
      ],
      "skill_gaps_to_close": ["Advanced Python", "Python Best Practices"],
      "source": "assessment_gaps"
    }
  ],
  "sources_used": ["career_paths", "role_fit_suggestions", "skill_gaps", "assessment_results"]
}
```

---

#### POST `/goals/{uid}/generate-plan`

Generate a complete plan from a goal description.

**Request:**
```json
{
  "goal_text": "I want to become a Cloud Solutions Architect",
  "target_date": "2026-09-01",
  "priority": "high"
}
```

**Response:**
```json
{
  "success": true,
  "generated_plan": {
    "goal": "Become a Cloud Solutions Architect",
    "description": "Transition from your current role to a Cloud Solutions Architect position",
    "estimated_duration_months": 6,
    "target_date": "2026-09-01",
    "priority": "high",
    "role_fit_percentage": 85,
    "milestones": [
      {
        "title": "Build Cloud Foundation",
        "description": "Learn cloud fundamentals and get entry-level certification",
        "order": 1,
        "estimated_duration_weeks": 4,
        "tasks": [
          {
            "title": "Complete AWS Cloud Practitioner course",
            "task_type": "course",
            "priority": "high",
            "estimated_hours": 20,
            "linked_resource": {
              "type": "course",
              "title": "AWS Cloud Practitioner Essentials",
              "url": "..."
            }
          },
          {
            "title": "Take Cloud Fundamentals assessment",
            "task_type": "assessment",
            "priority": "high",
            "estimated_hours": 2,
            "completion_criteria": "Score 70% or higher"
          },
          {
            "title": "Pass AWS Cloud Practitioner exam",
            "task_type": "certification",
            "priority": "high",
            "estimated_hours": 4
          }
        ]
      },
      {
        "title": "Get Solutions Architect Certified",
        "description": "Prepare for and pass the AWS Solutions Architect Associate exam",
        "order": 2,
        "estimated_duration_weeks": 6,
        "tasks": [
          {
            "title": "Study AWS Solutions Architect materials",
            "task_type": "course",
            "priority": "high",
            "estimated_hours": 40
          },
          {
            "title": "Complete 5 practice exams",
            "task_type": "assessment",
            "priority": "high",
            "estimated_hours": 10
          },
          {
            "title": "Pass AWS Solutions Architect Associate exam",
            "task_type": "certification",
            "priority": "high",
            "estimated_hours": 3
          }
        ]
      },
      {
        "title": "Build Portfolio Projects",
        "description": "Create real-world cloud architecture projects",
        "order": 3,
        "estimated_duration_weeks": 8,
        "tasks": [
          {
            "title": "Design and deploy a scalable web application",
            "task_type": "project",
            "priority": "high",
            "estimated_hours": 30
          },
          {
            "title": "Build a serverless data pipeline",
            "task_type": "project",
            "priority": "medium",
            "estimated_hours": 25
          },
          {
            "title": "Create a multi-region disaster recovery setup",
            "task_type": "project",
            "priority": "medium",
            "estimated_hours": 20
          }
        ]
      },
      {
        "title": "Land the Role",
        "description": "Update profile and apply to target positions",
        "order": 4,
        "estimated_duration_weeks": 6,
        "tasks": [
          {
            "title": "Update resume with cloud architecture focus",
            "task_type": "resume_update",
            "priority": "high",
            "estimated_hours": 3
          },
          {
            "title": "Complete 3 mock system design interviews",
            "task_type": "interview_prep",
            "priority": "high",
            "estimated_hours": 6
          },
          {
            "title": "Apply to 10 Cloud Architect positions",
            "task_type": "job_application",
            "priority": "high",
            "estimated_hours": 5
          }
        ]
      }
    ],
    "skill_gaps_addressed": [
      "Cloud Architecture Patterns",
      "AWS Services",
      "System Design",
      "Multi-region Architecture"
    ],
    "profile_analysis": {
      "current_strengths": ["Python", "Backend Development", "REST APIs"],
      "gaps_to_close": ["Cloud Architecture", "AWS Certification", "System Design"],
      "estimated_effort": "15-20 hours per week for 6 months"
    }
  },
  "auto_save": false
}
```

---

#### POST `/goals/{uid}`

Create a new goal (save a generated plan).

**Request:**
```json
{
  "goal": "Become a Cloud Solutions Architect",
  "description": "...",
  "target_date": "2026-09-01",
  "priority": "high",
  "milestones": [...],
  "created_by": "page"
}
```

**Response:**
```json
{
  "success": true,
  "goal_id": "uuid-v4",
  "message": "Goal created successfully",
  "goal": {...}
}
```

---

#### POST `/goals/{uid}/{goal_id}/task/{task_id}/complete`

Mark a task as complete.

**Request:**
```json
{
  "completed_at": "2026-02-10T14:00:00Z",
  "notes": "Scored 85% on the assessment"
}
```

**Response:**
```json
{
  "success": true,
  "task": {
    "task_id": "...",
    "status": "completed",
    "completed_at": "2026-02-10T14:00:00Z"
  },
  "milestone_progress": 0.67,
  "goal_progress": 0.35,
  "next_task": {
    "task_id": "...",
    "title": "Take practice exam #2"
  }
}
```

---

## 5. Plan Generation Agent

### 5.1 Agent Overview

The **Goal Plan Generator** is an AI agent that creates personalized milestone/task plans based on:

1. User's stated goal
2. User's current profile (resume, skills, experience)
3. User's skill gaps
4. Existing assessment results
5. Available courses and resources

### 5.2 Agent Prompt Template

```python
GOAL_PLAN_GENERATOR_PROMPT = """
You are a career planning expert. Create a detailed, actionable plan to help someone achieve their career goal.

## USER PROFILE
{user_profile}

## CURRENT SKILLS
{current_skills}

## SKILL GAPS (areas to improve)
{skill_gaps}

## ASSESSMENT RESULTS
{assessment_results}

## AVAILABLE COURSES (can be linked to tasks)
{available_courses}

## USER'S GOAL
{goal_text}

## TARGET DATE
{target_date}

## INSTRUCTIONS

Create a comprehensive plan with:

1. **MILESTONES** (3-5 major phases to reach the goal)
   - Each milestone should be a significant achievement
   - Milestones should be sequential (complete one before the next)
   - Estimate duration in weeks

2. **TASKS** (3-6 tasks per milestone)
   - Each task should be specific and actionable
   - Assign task_type from: course, assessment, project, certification, job_application, resume_update, interview_prep, networking, learning, custom
   - Link to actual courses/assessments when available
   - Estimate hours for each task
   - Set priority (high/medium/low)

3. **PERSONALIZATION**
   - Consider user's current skills (don't repeat what they know)
   - Focus on closing skill gaps
   - Use assessment results to prioritize weak areas
   - Link to specific courses from the available list

## OUTPUT FORMAT

Return a JSON object with this structure:
{
  "goal": "...",
  "description": "...",
  "estimated_duration_months": N,
  "milestones": [
    {
      "title": "...",
      "description": "...",
      "order": 1,
      "estimated_duration_weeks": N,
      "tasks": [
        {
          "title": "...",
          "description": "...",
          "task_type": "course|assessment|project|...",
          "priority": "high|medium|low",
          "estimated_hours": N,
          "linked_resource": { "type": "...", "title": "...", "url": "..." } // optional
        }
      ]
    }
  ],
  "skill_gaps_addressed": ["..."],
  "profile_analysis": {
    "current_strengths": ["..."],
    "gaps_to_close": ["..."],
    "estimated_effort": "X hours per week for Y months"
  }
}
"""
```

### 5.3 Agent Implementation

```python
# agents/agents/goal_plan_generator.py

async def generate_goal_plan(
    uid: str,
    goal_text: str,
    target_date: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a complete goal plan with milestones and tasks.
    
    Args:
        uid: User ID
        goal_text: The career goal (e.g., "Become a Cloud Architect")
        target_date: Optional target completion date
        session_id: Optional session ID
        
    Returns:
        Complete plan with milestones and tasks
    """
    # 1. Load user context
    context = await load_comprehensive_context(uid, session_id)
    
    # 2. Get available courses for linking
    courses = await get_relevant_courses(goal_text, context)
    
    # 3. Build prompt
    prompt = GOAL_PLAN_GENERATOR_PROMPT.format(
        user_profile=format_profile(context),
        current_skills=format_skills(context),
        skill_gaps=format_gaps(context),
        assessment_results=format_assessments(context),
        available_courses=format_courses(courses),
        goal_text=goal_text,
        target_date=target_date or "6 months from now"
    )
    
    # 4. Invoke LLM
    response = await invoke_llm(prompt, response_format="json")
    
    # 5. Parse and validate
    plan = parse_plan_response(response)
    
    # 6. Enrich with IDs and timestamps
    plan = enrich_plan(plan, uid)
    
    return plan
```

---

## 6. Recommended Goals

### 6.1 Sources for Recommendations

| Source | Goal Type | Example |
|--------|-----------|---------|
| **Career Paths** | Role transition | "Become a Cloud Solutions Architect" |
| **Role Fit Suggestions** | Best-fit roles | "Transition to DevOps Engineer (85% fit)" |
| **Skill Gaps** | Skill improvement | "Master AWS Services" |
| **Assessment Results** | Weak area improvement | "Improve Python proficiency (scored 45%)" |
| **Job Market Trends** | Emerging skills | "Learn Kubernetes for high-demand roles" |

### 6.2 Recommendation Algorithm

```python
async def get_recommended_goals(uid: str, session_id: Optional[str] = None) -> List[Dict]:
    """
    Generate personalized goal recommendations based on user profile.
    """
    context = await load_comprehensive_context(uid, session_id)
    recommendations = []
    
    # 1. From career paths (highest priority)
    if context.get("career_paths"):
        for path in context["career_paths"][:3]:
            recommendations.append({
                "goal": f"Transition to {path['title']}",
                "description": path.get("description", ""),
                "role_fit_percentage": path.get("match_percentage"),
                "source": "career_paths",
                "priority": "high"
            })
    
    # 2. From role fit suggestions
    if context.get("role_fit_suggestions"):
        for role in context["role_fit_suggestions"][:2]:
            if role.get("fit_percentage", 0) >= 70:
                recommendations.append({
                    "goal": f"Become a {role['role_name']}",
                    "role_fit_percentage": role.get("fit_percentage"),
                    "source": "role_fit",
                    "priority": "high"
                })
    
    # 3. From skill gaps
    if context.get("skill_gaps"):
        top_gaps = context["skill_gaps"][:3]
        for gap in top_gaps:
            recommendations.append({
                "goal": f"Master {gap}",
                "description": f"Close your skill gap in {gap}",
                "source": "skill_gaps",
                "priority": "medium"
            })
    
    # 4. From assessment results (low scores)
    if context.get("assessment_results"):
        for result in context["assessment_results"]:
            score = result.get("total_score", 100)
            if score < 60:
                topic = result.get("assessment_topic", "Unknown")
                recommendations.append({
                    "goal": f"Improve {topic} proficiency",
                    "description": f"Your assessment score was {score}%. Target 80%+",
                    "source": "assessment_gaps",
                    "priority": "medium"
                })
    
    # Deduplicate and rank
    return dedupe_and_rank(recommendations)[:5]
```

---

## 7. Task Types & Platform Integration

### 7.1 Task Type Mapping

| Task Type | Platform Feature | Auto-Link | Completion Tracking |
|-----------|------------------|-----------|---------------------|
| `course` | Course Recommendations | Yes - links to specific courses | Manual (user marks complete) |
| `assessment` | Assessments | Yes - links to specific assessment | Auto (when assessment completed) |
| `project` | External | No | Manual |
| `certification` | External | No | Manual |
| `job_application` | Jobs Section | Yes - links to job search | Manual |
| `resume_update` | Resume Analysis | Yes - links to resume page | Manual |
| `interview_prep` | AI Interview | Yes - links to interview practice | Auto (interview completed) |
| `networking` | External | No | Manual |
| `learning` | External | No | Manual |
| `custom` | None | No | Manual |

### 7.2 Auto-Completion Triggers

```python
# When user completes an assessment, check if it completes any goal tasks
async def on_assessment_completed(uid: str, assessment_topic: str, score: float):
    goals = await get_career_goals(uid)
    for goal in goals:
        for milestone in goal.get("milestones", []):
            for task in milestone.get("tasks", []):
                if (task.get("task_type") == "assessment" and 
                    task.get("status") != "completed" and
                    assessment_topic.lower() in task.get("title", "").lower()):
                    # Auto-complete the task
                    await complete_task(uid, goal["goal_id"], task["task_id"], {
                        "auto_completed": True,
                        "score": score
                    })

# When user completes an interview session
async def on_interview_completed(uid: str, interview_type: str):
    # Similar logic to auto-complete interview_prep tasks
    pass
```

---

## 8. UI/UX Wireframes

### 8.1 Main Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🎯 GROWTH PLAN                                           [+ New Goal]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  💡 What career goal do you want to achieve?                     │   │
│  │  ┌─────────────────────────────────────────────────────────┐     │   │
│  │  │ Type your goal here...                                   │     │   │
│  │  └─────────────────────────────────────────────────────────┘     │   │
│  │                                              [Generate Plan] 🚀   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  📌 RECOMMENDED FOR YOU                                                 │
│                                                                         │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐  │
│  │ ☁️ Cloud Architect │ │ 🔧 DevOps Engineer │ │ 👨‍💼 Tech Lead      │  │
│  │ ━━━━━━━━━━━━━━━━━ │ │ ━━━━━━━━━━━━━━━━━ │ │ ━━━━━━━━━━━━━━━━━ │  │
│  │ 85% role fit      │ │ 78% role fit      │ │ 72% role fit      │  │
│  │ ~6 months         │ │ ~4 months         │ │ ~8 months         │  │
│  │                   │ │                   │ │                   │  │
│  │ Key milestones:   │ │ Key milestones:   │ │ Key milestones:   │  │
│  │ • AWS Cert        │ │ • CI/CD Mastery   │ │ • Leadership      │  │
│  │ • System Design   │ │ • Kubernetes      │ │ • Architecture    │  │
│  │ • Portfolio       │ │ • IaC             │ │ • Mentoring       │  │
│  │                   │ │                   │ │                   │  │
│  │ [Create Plan] ➜   │ │ [Create Plan] ➜   │ │ [Create Plan] ➜   │  │
│  └────────────────────┘ └────────────────────┘ └────────────────────┘  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  📋 YOUR ACTIVE GOALS                                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 🎯 Become a Cloud Solutions Architect                    [Edit] │   │
│  │ ━━━━━━━━━━━━━━━━━━━━━━━━░░░░░░░░░░░░░░░ 45%                     │   │
│  │ 📅 Target: Sep 2026  |  ⭐ Priority: High  |  ⏱️ 6 months       │   │
│  │                                                                 │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ ✅ Milestone 1: Build Foundation              [COMPLETED]   │ │   │
│  │ │    ├─ ✓ Complete AWS Cloud Practitioner course              │ │   │
│  │ │    ├─ ✓ Take Cloud Fundamentals assessment                  │ │   │
│  │ │    └─ ✓ Pass AWS Cloud Practitioner exam                    │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ 🔄 Milestone 2: Get Certified                [IN PROGRESS]  │ │   │
│  │ │    ├─ ✓ Study AWS Solutions Architect materials             │ │   │
│  │ │    ├─ ○ Complete practice exam 1/5         [Start] →        │ │   │
│  │ │    ├─ ○ Complete practice exam 2/5                          │ │   │
│  │ │    ├─ ○ Complete practice exam 3/5                          │ │   │
│  │ │    ├─ ○ Complete practice exam 4/5                          │ │   │
│  │ │    ├─ ○ Complete practice exam 5/5                          │ │   │
│  │ │    └─ ○ Pass AWS Solutions Architect Associate exam         │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ ○ Milestone 3: Build Portfolio                [PENDING]     │ │   │
│  │ │    └─ 3 tasks                                               │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ ○ Milestone 4: Land the Role                  [PENDING]     │ │   │
│  │ │    └─ 3 tasks                                               │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Task Detail Modal

```
┌─────────────────────────────────────────────────────────────┐
│  📝 TASK DETAILS                                       [X]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Complete AWS Cloud Practitioner course                     │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  📋 Description:                                            │
│  Learn cloud computing fundamentals through the official    │
│  AWS Cloud Practitioner course.                             │
│                                                             │
│  🏷️ Type: Course                                            │
│  ⏱️ Estimated: 20 hours                                     │
│  ⭐ Priority: High                                          │
│  📅 Due: Feb 15, 2026                                       │
│                                                             │
│  🔗 Linked Resource:                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  AWS Cloud Practitioner Essentials                   │   │
│  │  Provider: AWS Training                              │   │
│  │  [Open Course] →                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ✅ Completion Criteria:                                    │
│  Score 80% or higher on course completion quiz              │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  📝 Notes (optional):                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│           [Cancel]              [Mark as Complete ✓]        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 Plan Generation Loading State

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    🎯 Generating Your Plan                  │
│                                                             │
│         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60%                 │
│                                                             │
│         ✓ Analyzing your profile...                         │
│         ✓ Identifying skill gaps...                         │
│         ⟳ Creating milestones...                            │
│         ○ Finding relevant courses...                       │
│         ○ Estimating timeline...                            │
│                                                             │
│         This usually takes 10-15 seconds                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Phases

### Phase 1: Backend Foundation (Week 1)

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Create Goal data models | `models/goal.py` |
| 1.2 | Add Chroma storage functions | `chroma.py` |
| 1.3 | Create basic CRUD endpoints | `main.py` |
| 1.4 | Connect to existing chatbot tools | `function_tools.py` |

**Deliverables:**
- `GET /goals/{uid}` - List goals
- `POST /goals/{uid}` - Create goal
- `PUT /goals/{uid}/{goal_id}` - Update goal
- `DELETE /goals/{uid}/{goal_id}` - Delete goal

### Phase 2: Plan Generation Agent (Week 2)

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Create plan generator prompt | `agents/goal_plan_generator.py` |
| 2.2 | Implement plan generation logic | `agents/goal_plan_generator.py` |
| 2.3 | Add course linking | `agents/goal_plan_generator.py` |
| 2.4 | Create generate-plan endpoint | `main.py` |

**Deliverables:**
- `POST /goals/{uid}/generate-plan` - Generate plan from goal text
- Plan includes milestones, tasks, linked resources

### Phase 3: Recommended Goals (Week 3)

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Create recommendation algorithm | `agents/goal_recommender.py` |
| 3.2 | Pull from career paths | `agents/goal_recommender.py` |
| 3.3 | Pull from skill gaps | `agents/goal_recommender.py` |
| 3.4 | Pull from assessment results | `agents/goal_recommender.py` |
| 3.5 | Create recommended endpoint | `main.py` |

**Deliverables:**
- `GET /goals/{uid}/recommended` - Get AI-recommended goals

### Phase 4: Task Management & Auto-Completion (Week 4)

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Task completion endpoint | `main.py` |
| 4.2 | Progress calculation | `function_tools.py` |
| 4.3 | Auto-completion triggers | `hooks/goal_hooks.py` |
| 4.4 | Milestone status updates | `function_tools.py` |

**Deliverables:**
- `POST /goals/{uid}/{goal_id}/task/{task_id}/complete`
- Auto-completion when assessments/interviews completed
- Progress recalculation on task completion

### Phase 5: Frontend Integration (Week 5-6)

| Task | Description |
|------|-------------|
| 5.1 | Goals list page |
| 5.2 | Goal detail view with milestones |
| 5.3 | Task checkbox interactions |
| 5.4 | Goal creation flow |
| 5.5 | Plan generation UI |
| 5.6 | Recommended goals section |

---

## 10. Test Cases

### 10.1 Goal CRUD Tests

```python
# Test: Create goal
def test_create_goal():
    response = client.post("/goals/user123", json={
        "goal": "Become a Cloud Architect",
        "target_date": "2026-09-01",
        "priority": "high",
        "milestones": [...]
    })
    assert response.status_code == 200
    assert response.json()["goal_id"] is not None

# Test: List goals
def test_list_goals():
    response = client.get("/goals/user123")
    assert response.status_code == 200
    assert "goals" in response.json()

# Test: Update goal
def test_update_goal():
    response = client.put("/goals/user123/goal-123", json={
        "status": "in_progress",
        "progress": 0.5
    })
    assert response.status_code == 200

# Test: Delete goal
def test_delete_goal():
    response = client.delete("/goals/user123/goal-123")
    assert response.status_code == 200
```

### 10.2 Plan Generation Tests

```python
# Test: Generate plan for career transition
def test_generate_plan_career_transition():
    response = client.post("/goals/user123/generate-plan", json={
        "goal_text": "I want to become a Cloud Solutions Architect"
    })
    assert response.status_code == 200
    plan = response.json()["generated_plan"]
    assert len(plan["milestones"]) >= 3
    assert all("tasks" in m for m in plan["milestones"])

# Test: Generate plan for skill improvement
def test_generate_plan_skill_improvement():
    response = client.post("/goals/user123/generate-plan", json={
        "goal_text": "Improve my Python skills"
    })
    assert response.status_code == 200
    plan = response.json()["generated_plan"]
    assert "Python" in plan["goal"]

# Test: Plan includes linked courses
def test_plan_includes_linked_courses():
    response = client.post("/goals/user123/generate-plan", json={
        "goal_text": "Learn AWS"
    })
    plan = response.json()["generated_plan"]
    course_tasks = [t for m in plan["milestones"] for t in m["tasks"] if t["task_type"] == "course"]
    assert len(course_tasks) > 0
```

### 10.3 Recommended Goals Tests

```python
# Test: Get recommendations
def test_get_recommended_goals():
    response = client.get("/goals/user123/recommended")
    assert response.status_code == 200
    recs = response.json()["recommended_goals"]
    assert len(recs) > 0
    assert all("goal" in r for r in recs)

# Test: Recommendations include role fit
def test_recommendations_include_role_fit():
    response = client.get("/goals/user123/recommended")
    recs = response.json()["recommended_goals"]
    career_recs = [r for r in recs if r["source"] == "career_paths"]
    assert any(r.get("role_fit_percentage") is not None for r in career_recs)
```

### 10.4 Task Completion Tests

```python
# Test: Complete task
def test_complete_task():
    response = client.post("/goals/user123/goal-123/task/task-456/complete", json={
        "notes": "Completed with 85% score"
    })
    assert response.status_code == 200
    assert response.json()["task"]["status"] == "completed"

# Test: Progress updates on completion
def test_progress_updates_on_completion():
    # Complete a task
    response = client.post("/goals/user123/goal-123/task/task-456/complete")
    assert response.json()["goal_progress"] > 0
    assert response.json()["milestone_progress"] > 0

# Test: Next task suggested
def test_next_task_suggested():
    response = client.post("/goals/user123/goal-123/task/task-456/complete")
    assert "next_task" in response.json()
```

### 10.5 Sync with Mentor Tests

```python
# Test: Goal created via mentor appears in page
def test_mentor_goal_syncs_to_page():
    # Create goal via chatbot tool
    await create_career_goal(uid="user123", goal="Learn Kubernetes", ...)
    
    # Fetch via page API
    response = client.get("/goals/user123")
    goals = response.json()["goals"]
    assert any("Kubernetes" in g["goal"] for g in goals)

# Test: Goal created via page known to mentor
def test_page_goal_known_to_mentor():
    # Create via page
    client.post("/goals/user123", json={"goal": "Master Python", ...})
    
    # Fetch via chatbot tool
    result = await get_career_goals(uid="user123")
    assert any("Python" in g["goal"] for g in result["goals"])
```

---

## Summary

This implementation creates a **Growth Plan page** that:

1. **Syncs with AI Mentor** - Same data store, bidirectional updates
2. **AI-Generated Plans** - Complete milestone/task breakdown from a goal
3. **Personalized Recommendations** - Goals suggested based on profile
4. **Task Linking** - Tasks connect to courses, assessments, jobs, interviews
5. **Progress Tracking** - Visual progress bars, auto-completion triggers
6. **Actionable** - Every task is specific with linked resources

The system makes goal-setting and career planning seamless, whether users prefer chatting with the AI Mentor or using the dedicated page.
