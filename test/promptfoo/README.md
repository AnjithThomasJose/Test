# Promptfoo Evals

Eval configs and prompts for all KA agents.

**→ [PROMPTFOO_USE_GUIDE.md](./PROMPTFOO_USE_GUIDE.md)** — Use guide for teammates (how to run evals, edit prompts, add tests).

**→ [PROMPTFOO_USE_GUIDE.tex](./PROMPTFOO_USE_GUIDE.tex)** — LaTeX version of the guide (compile with `pdflatex PROMPTFOO_USE_GUIDE.tex`). Edit the `.txt` prompt files to test changes; run evals to compare outputs.

## Structure

```
promptfoo/
├── prompts/              # Prompt templates (edit these to test)
│   ├── resume_parser.txt
│   ├── enhanced_role_fit.txt
│   ├── assessment_questions.txt
│   ├── jd_parser.txt
│   ├── prescreening_questions.txt
│   ├── skill_proficiency_analyzer.txt
│   ├── interview_transcript_evaluator.txt
│   ├── validate_resume.txt
│   ├── mock_interview_prep.txt
│   ├── goal_plan_generator.txt
│   ├── resume_content_generator.txt
│   ├── interest_filler.txt
│   ├── resume_analysis.txt
│   ├── resume_score.txt
│   ├── validate_jd.txt
│   ├── interview_question_generator.txt
│   ├── ranker.txt
│   ├── job_matcher.txt
│   ├── market_and_course_recommender.txt
│   ├── assessment_recommender.txt
│   ├── assessment_evaluator.txt
│   ├── report_generator.txt
│   ├── career_chatbot.txt
│   ├── skill_and_career_advisor.txt
│   ├── assessment_validator.txt
│   └── assessment_builder.txt
├── providers/
│   ├── resume_provider.py       # Groq
│   ├── jd_parser_provider.py    # Groq
│   ├── enhanced_role_fit_provider.py
│   ├── assessment_question_provider.py
│   └── gemini_provider.py       # Generic Gemini for most agents
└── *.yaml                       # One config per agent
```

## Run Evals

From the `agents/` directory:

```bash
# Groq-based (resume, JD)
npx promptfoo eval -c promptfoo/resume_parser.yaml
npx promptfoo eval -c promptfoo/jd_parser.yaml

# Gemini-based
npx promptfoo eval -c promptfoo/enhanced_role_fit.yaml
npx promptfoo eval -c promptfoo/assessment_questions.yaml
npx promptfoo eval -c promptfoo/prescreening_questions.yaml
npx promptfoo eval -c promptfoo/skill_proficiency_analyzer.yaml
npx promptfoo eval -c promptfoo/interview_transcript_evaluator.yaml
npx promptfoo eval -c promptfoo/validate_resume.yaml
npx promptfoo eval -c promptfoo/mock_interview_prep.yaml
npx promptfoo eval -c promptfoo/goal_plan_generator.yaml
npx promptfoo eval -c promptfoo/resume_content_generator.yaml
npx promptfoo eval -c promptfoo/interest_filler.yaml
npx promptfoo eval -c promptfoo/resume_analysis.yaml
npx promptfoo eval -c promptfoo/resume_score.yaml
npx promptfoo eval -c promptfoo/validate_jd.yaml
npx promptfoo eval -c promptfoo/interview_question_generator.yaml
npx promptfoo eval -c promptfoo/ranker.yaml
npx promptfoo eval -c promptfoo/job_matcher.yaml
npx promptfoo eval -c promptfoo/market_and_course_recommender.yaml
npx promptfoo eval -c promptfoo/assessment_recommender.yaml
npx promptfoo eval -c promptfoo/assessment_evaluator.yaml
npx promptfoo eval -c promptfoo/report_generator.yaml
npx promptfoo eval -c promptfoo/career_chatbot.yaml
npx promptfoo eval -c promptfoo/skill_and_career_advisor.yaml
npx promptfoo eval -c promptfoo/assessment_validator.yaml
npx promptfoo eval -c promptfoo/assessment_builder.yaml
```

## View Results

```bash
npx promptfoo view
```

## Testing Different Prompts

1. Edit the `.txt` file in `prompts/` for the agent
2. Run the corresponding eval
3. Compare outputs in the Promptfoo UI

**Note:** Prompts here are extracted/simplified from production. Production agents may use inline prompts or `prompt_generator.py`. Sync changes when promoting to production.
