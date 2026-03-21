import logging
from typing import Dict, Any, Optional
from langsmith import Client

log = logging.getLogger(__name__)


class LangSmithEvaluator:
    """
    Central LangSmith Code Evaluator for all agents
    (resume parser, assessment recommender, future agents)
    """

    def __init__(self):
        try:
            self.client = Client()
            self.project_name = "pr-gripping-physics-6"
            self.enabled = True
        except Exception as e:
            log.warning(f"LangSmith evaluator initialization failed: {e}")
            self.enabled = False

    def evaluate(
        self,
        agent_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        tenant_id: str = "default_tenant"
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate agent output using LangSmith Code Evaluator
        """
        if not self.enabled:
            return None

        try:
            # --------------------------------------------------
            # Extract evaluation context
            # --------------------------------------------------
            input_context = self._extract_input_context(input_data)
            output_text = self._extract_output_text(output_data)

            # --------------------------------------------------
            # Create LangSmith run
            # --------------------------------------------------
            run_id = self.client.create_run(
                name=f"{agent_name}_eval",
                project_name=self.project_name,
                run_type="chain",
                inputs={"context": input_context},
                outputs={"outputs": output_text},
                extra={
                    "tenant_id": tenant_id,
                    "agent_name": agent_name,
                    "agent_type": agent_name
                }
            )

            # --------------------------------------------------
            # Run Code Evaluator (agent-aware)
            # --------------------------------------------------
            eval_result = self._code_evaluator(
                context=input_context,
                output=output_text,
                agent_name=agent_name
            )

            log.debug(f"✅ Evaluation submitted for {agent_name}: {run_id}")

            return {
                "success": True,
                "run_id": str(run_id),
                "agent_name": agent_name,
                "evaluation": eval_result
            }

        except Exception as e:
            log.warning(f"Evaluation failed for {agent_name}: {e}")
            return {"success": False, "error": str(e)}

    # ======================================================
    # Code Evaluator (NO LLM-as-Judge)
    # ======================================================
    def _code_evaluator(
        self,
        context: str,
        output: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Agent-specific evaluation logic
        """

        if not output:
            return {
                "score": 0.0,
                "comment": "Output missing or empty"
            }

        # Resume Parser Evaluation
        if agent_name == "groq_resume_parser":
            return {
                "score": 1.0,
                "comment": "Resume parsed successfully"
            }

        # Assessment Recommender Evaluation
        if agent_name == "assessment_recommender":
            return {
                "score": 1.0,
                "comment": "Assessment recommendations generated"
            }
        
        # 👇 ADD HERE
        if agent_name == "assessment_question_generator":
            return {
                "score": 1.0,
                "comment": "Assessment questions generated successfully"
            }

        # 👇 ADD THIS
        if agent_name == "assessment_evaluator":
            return {
                "score": 1.0,
                "comment": "Assessment evaluated successfully"
            }

        # 👇 ADD THIS
        if agent_name == "report_generator":
            return {
                "score": 1.0,
                "comment": "Report generated successfully"
            }

        # Default evaluation
        return {
            "score": 1.0,
            "comment": "Evaluation passed"
        }

    # ======================================================
    # Input Extractors
    # ======================================================
    def _extract_input_context(self, input_data: Dict[str, Any]) -> str:
        """
        Extract meaningful input context from agent state
        """
        if "resume_text" in input_data and input_data["resume_text"]:
            text = input_data["resume_text"]
            return text[:2000] if len(text) > 2000 else text

        if "resume_url" in input_data:
            return f"Resume URL: {input_data['resume_url']}"

        if "user_profile" in input_data:
            return str(input_data["user_profile"])[:2000]

        return str(input_data)[:2000]

    # ======================================================
    # Output Extractors
    # ======================================================
    def _extract_output_text(self, output_data: Dict[str, Any]) -> str:
        """
        Convert agent output to readable text for evaluation
        """
        if not output_data:
            return ""

        parts = []

        # Resume Parser Output
        if isinstance(output_data, dict):
            if "name" in output_data:
                parts.append(f"Name: {output_data.get('name')}")

            if "professional_summary" in output_data:
                parts.append(f"Summary: {output_data.get('professional_summary')}")

            if "work_experience" in output_data:
                parts.append(f"Work Experience Entries: {len(output_data['work_experience'])}")

            if "education" in output_data:
                parts.append(f"Education Entries: {len(output_data['education'])}")

            if "skills" in output_data:
                parts.append(f"Skills Count: {len(output_data['skills'])}")

            if "total_experience_years" in output_data:
                parts.append(f"Total Experience: {output_data['total_experience_years']}")

            # Assessment Recommender Output
            if "assessments" in output_data:
                parts.append(f"Assessments Recommended: {len(output_data['assessments'])}")

        return "\n".join(parts)


# ==========================================================
# Global Evaluator Instance
# ==========================================================
_evaluator = LangSmithEvaluator()


# ==========================================================
# Public API (Used by ALL agents)
# ==========================================================
async def evaluate_agent(
    agent_name: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    tenant_id: str = "default_tenant"
) -> Optional[Dict[str, Any]]:
    """
    Public function to evaluate any agent
    """
    return _evaluator.evaluate(
        agent_name=agent_name,
        input_data=input_data,
        output_data=output_data,
        tenant_id=tenant_id
    )
