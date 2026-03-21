# Promptfoo Use Guide

A practical guide for testing and iterating on agent prompts using Promptfoo.

---

## What is Promptfoo?

Promptfoo lets you:
- **Test prompts** against real LLM outputs
- **Compare variants** (e.g., before/after prompt changes)
- **Assert on outputs** (JSON structure, required fields, quality checks)
- **Track regressions** when you change prompts

---

## Quick Start

### 1. Prerequisites

- Node.js (for `npx promptfoo`)
- Python 3.x (for providers)
- API keys in `.env` or `.env.development`:
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` (most agents)
  - `GROQ_API_KEY` (resume parser, JD parser)

### 2. Run Your First Eval

From the `agents/` directory:

```bash
# Test the resume parser
npx promptfoo eval -c promptfoo/resume_parser.yaml

# Test the assessment question generator
npx promptfoo eval -c promptfoo/assessment_questions.yaml
```

### 3. View Results

```bash
npx promptfoo view
```

Opens a web UI where you can inspect prompts, outputs, and assertion results.

---

## Common Workflows

### Workflow 1: Test a Prompt Change

1. Edit the prompt file in `promptfoo/prompts/` (e.g., `assessment_questions.txt`)
2. Run the eval:
   ```bash
   npx promptfoo eval -c promptfoo/assessment_questions.yaml
   ```
3. Check the output in the terminal or run `npx promptfoo view` for the full UI
4. If assertions pass and outputs look good, promote the change to production (e.g., update `prompt_generator.py` or the shared prompt file)

### Workflow 2: Add a New Test Case

1. Open the YAML config (e.g., `promptfoo/assessment_questions.yaml`)
2. Add a new entry under `tests:`:

```yaml
tests:
  - vars:
      topic: "Your new topic"
      difficulty: "hard"
      assessment_type: "short_answer"
      breakdown: "- 2 short_answer questions"
      coding_note: ""
    assert:
      - type: javascript
        value: |
          try {
            const p = JSON.parse(output);
            const aq = p.assessment_questions || p;
            const list = aq.short_answer || aq[Object.keys(aq)[0]] || [];
            return Array.isArray(list) && list.length >= 1;
          } catch (e) { return false; }
```

3. Run the eval to verify

### Workflow 3: Compare Two Prompt Versions

1. Create a second prompt file (e.g., `assessment_questions_v2.txt`)
2. Add both prompts to the config:

```yaml
prompts:
  - file://prompts/assessment_questions.txt
  - file://prompts/assessment_questions_v2.txt
```

3. Run the eval — you’ll get outputs for both prompts side by side
4. Use `npx promptfoo view` to compare

### Workflow 4: Run All Agent Evals

```bash
cd agents
for f in promptfoo/*.yaml; do
  echo "Running $f..."
  npx promptfoo eval -c "$f"
done
```

---

## File Structure

| Path | Purpose |
|------|---------|
| `promptfoo/prompts/*.txt` | Prompt templates. Edit these to test changes. |
| `promptfoo/providers/*.py` | Python scripts that call the LLM (Gemini/Groq). |
| `promptfoo/*.yaml` | Eval configs: prompts, providers, test cases, assertions. |

---

## Understanding YAML Configs

Example config:

```yaml
description: "Assessment question generator"
prompts:
  - file://prompts/assessment_questions.txt
providers:
  - file://providers/gemini_provider.py
tests:
  - vars:
      topic: "Python basics"
      difficulty: "easy"
      breakdown: "- 3 mcq questions"
    assert:
      - type: javascript
        value: |
          const p = JSON.parse(output);
          return p.assessment_questions && p.assessment_questions.mcq?.length >= 2;
```

- **vars**: Values substituted into `{{var}}` placeholders in the prompt
- **assert**: Checks run on the LLM output. If any fail, the test fails.
- **providers**: Script that sends the prompt to the LLM and returns the response

---

## Assertion Types

### JavaScript (custom logic)

```yaml
- type: javascript
  value: |
    const p = JSON.parse(output);
    return p.assessment_questions?.mcq?.length >= 2;
```

### Contains / Not contains

```yaml
- type: contains
  value: "expected substring"
- type: not-contains
  value: "```"
```

### LLM-as-judge (if configured)

```yaml
- type: llm-rubric
  value: "The output must be valid JSON with at least 2 questions."
```

---

## Agent → Config Mapping

| Agent | Config File |
|-------|-------------|
| Resume parser | `resume_parser.yaml` |
| JD parser | `jd_parser.yaml` |
| Enhanced role fit | `enhanced_role_fit.yaml` |
| Assessment questions | `assessment_questions.yaml` |
| Prescreening questions | `prescreening_questions.yaml` |
| Skill proficiency | `skill_proficiency_analyzer.yaml` |
| Interview transcript eval | `interview_transcript_evaluator.yaml` |
| Validate resume | `validate_resume.yaml` |
| Mock interview prep | `mock_interview_prep.yaml` |
| Goal plan | `goal_plan_generator.yaml` |
| Resume content | `resume_content_generator.yaml` |
| Interest filler | `interest_filler.yaml` |
| Resume analysis | `resume_analysis.yaml` |
| Resume score | `resume_score.yaml` |
| Validate JD | `validate_jd.yaml` |
| Interview questions | `interview_question_generator.yaml` |
| Ranker | `ranker.yaml` |
| Job matcher | `job_matcher.yaml` |
| Market/course recommender | `market_and_course_recommender.yaml` |
| Assessment recommender | `assessment_recommender.yaml` |
| Assessment evaluator | `assessment_evaluator.yaml` |
| Report generator | `report_generator.yaml` |
| Career chatbot | `career_chatbot.yaml` |
| Skill & career advisor | `skill_and_career_advisor.yaml` |
| Assessment validator | `assessment_validator.yaml` |
| Assessment builder | `assessment_builder.yaml` |

---

## Tips

1. **Start small** — Run one agent’s eval before changing prompts.
2. **Use `promptfoo view`** — Easier to debug than terminal output.
3. **Keep prompts in sync** — Promptfoo prompts are copies; update production (`prompt_generator.py`, `prompts/`) when you’re happy.
4. **Assert on structure first** — Validate JSON shape and required fields before adding quality checks.
5. **Check API keys** — If you see "API key not set", ensure `.env` or `.env.development` is loaded (providers load it automatically when run via Promptfoo).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GOOGLE_API_KEY not set` | Add to `.env` or `.env.development` in `agents/` |
| `GROQ_API_KEY not set` | Same as above for Groq-based agents |
| Assertion fails on valid-looking output | Inspect the exact output in `promptfoo view`; assertion logic may be too strict |
| Provider import error | Run from `agents/` so Python can resolve imports |
| Timeout | Increase timeout in the provider or simplify the prompt |

---

## Further Reading

- [Promptfoo docs](https://www.promptfoo.dev/docs/)
- [Configuration reference](https://www.promptfoo.dev/docs/configuration/reference)
