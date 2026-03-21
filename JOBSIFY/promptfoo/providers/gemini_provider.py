"""
Generic Promptfoo provider for Gemini-based agents.
Uses the full prompt (after var substitution) and calls gemini-2.5-flash.
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
    for start_tag in ("```json", "```"):
        if start_tag in text:
            idx = text.find(start_tag)
            text = text[idx + len(start_tag):].lstrip()
            if text.startswith("json"):
                text = text[4:].lstrip()
            end = text.find("```")
            if end != -1:
                text = text[:end].strip()
            break
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1:
        return None
    open_ch, close_ch = ("{", "}") if text[start] == "{" else ("[", "]")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()
    return text[start:].strip()


def call_api(prompt, options, context):
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
        max_output_tokens=8192,
        google_api_key=api_key,
    )
    try:
        response = asyncio.run(asyncio.wait_for(model.ainvoke(prompt), timeout=90))
        raw = getattr(response, "content", str(response)) or ""
        extracted = _extract_json_from_response(raw)
        return {"output": extracted if extracted else raw}
    except Exception as e:
        return {"output": json.dumps({"error": str(e)}), "error": str(e)}
