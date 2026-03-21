"""
Promptfoo provider for assessment question generation. Uses promptfoo/prompts/assessment_questions.txt.
Calls Gemini (gemini-2.5-flash).
"""
import os
import json
import asyncio
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_env() -> None:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
        for name in (".env", ".env.development", ".env.dev"):
            for base in (_SCRIPT_DIR.parent.parent, _SCRIPT_DIR):
                path = base / name
                if path.exists():
                    load_dotenv(path)
                    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
                        return
    except ImportError:
        pass


def _extract_json_from_response(response_text: str) -> str | None:
    if not response_text:
        return None
    text = response_text.strip()
    fence_pos = text.find("```")
    if fence_pos != -1:
        text = text[fence_pos + 3:]
        if text.startswith("json"):
            text = text[4:].lstrip()
        second = text.find("```")
        if second != -1:
            text = text[:second].strip()
    start = text.find("{")
    if start == -1:
        return None
    brace_count = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i + 1].strip()
    last = text.rfind("}")
    if last > start:
        return text[start:last + 1].strip()
    return None


def call_api(prompt, options, context):
    """prompt = full prompt with {{topic}}, {{difficulty}}, etc. substituted."""
    _load_env()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"output": json.dumps({"error": "GOOGLE_API_KEY not set"}), "error": "Missing API key"}
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return {"output": json.dumps({"error": "langchain_google_genai not installed"}), "error": "Import error"}
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=8000,
        google_api_key=api_key,
    )
    try:
        response = asyncio.run(asyncio.wait_for(model.ainvoke(prompt), timeout=90))
        raw = getattr(response, "content", str(response)) or ""
        extracted = _extract_json_from_response(raw)
        return {"output": extracted if extracted else raw}
    except Exception as e:
        return {"output": json.dumps({"error": str(e)}), "error": str(e)}
