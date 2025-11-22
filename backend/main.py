import json
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required. Set it in your environment or .env file.")

client = OpenAI(api_key=API_KEY)

app = FastAPI(title="ResearchR MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    assumptions: List[Dict[str, Any]] = Field(default_factory=list)
    last_equation_hits: List[Dict[str, Any]] = Field(default_factory=list)
    turn_index: int = 0


sessions: Dict[str, SessionState] = {}


SYSTEM_PROMPT = """
You are ResearchR, a scientific assistant. Goals:
- Answer user technical questions clearly.
- Detect equations (LaTeX or plain text). When equations appear, use the `web_search` tool to find papers or technical sources that contain the same or equivalent equations. Do not invent links; only use results from `web_search`.
- Extract new assumptions and check consistency with prior assumptions.

When you need literature or examples for an equation, invoke the `web_search` tool with both the literal equation text and a short semantic description. Only surface URLs returned by `web_search`.

Respond with a JSON object and nothing else, with keys:
{
  "chat_answer": "<main answer for the user in Markdown>",
  "equation_hits": [
    {"title": "", "authors": "", "year": "", "url": "", "context_snippet": "", "relation_to_query_equation": "", "source_tool_id": "<id of the web_search call you used>"}
  ],
  "assumptions_delta": [
    {"text": "", "type": "", "variables": [], "source": "model"}
  ],
  "consistency_warnings": [
    {"assumption_text": "", "problem_type": "", "explanation": ""}
  ]
}
Keep fields empty if not applicable. Do not fabricate references. If web search finds nothing convincing, return an empty equation_hits list and note that in chat_answer.
"""


def get_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]


def build_model_input(state: SessionState, user_message: str) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Current assumptions (for consistency checking): {json.dumps(state.assumptions)}",
        },
    ]
    for msg in state.messages:
        history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": user_message})
    return history


def parse_response_text(text: str) -> Dict[str, Any]:
    """Extract JSON block from model text output."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    return {
        "chat_answer": text,
        "equation_hits": [],
        "assumptions_delta": [],
        "consistency_warnings": [],
    }


def call_model(state: SessionState, user_message: str) -> Dict[str, Any]:
    inputs = build_model_input(state, user_message)
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=inputs,
        tools=[{"type": "web_search"}],
        max_output_tokens=8192,
    )

    text: Optional[str] = getattr(response, "output_text", None)
    if not text:
        # Fallback to concatenating text segments if output_text is unavailable.
        parts: List[str] = []
        if hasattr(response, "output"):
            for item in response.output or []:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", "") == "output_text":
                        parts.append(getattr(content, "text", ""))
        text = "\n".join(parts)

    if not text:
        text = "Model returned no text."

    return parse_response_text(text)


def merge_assumptions(state: SessionState, deltas: List[Dict[str, Any]], turn_index: int) -> None:
    for delta in deltas or []:
        text = (delta.get("text") or "").strip()
        if not text:
            continue
        if any(a["text"].lower() == text.lower() for a in state.assumptions):
            continue
        state.assumptions.append(
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "type": delta.get("type") or "",
                "variables": delta.get("variables") or [],
                "source": delta.get("source") or "model",
                "created_at_turn": turn_index,
            }
        )


@app.post("/api/chat")
def chat(request: ChatRequest = Body(...)) -> JSONResponse:
    state = get_session(request.session_id)
    state.turn_index += 1

    model_payload = call_model(state, request.message)

    assistant_answer = model_payload.get("chat_answer", "")
    equation_hits = model_payload.get("equation_hits") or []
    assumptions_delta = model_payload.get("assumptions_delta") or []
    consistency_warnings = model_payload.get("consistency_warnings") or []

    merge_assumptions(state, assumptions_delta, state.turn_index)
    state.last_equation_hits = equation_hits

    state.messages.append({"role": "user", "content": request.message, "turn": state.turn_index})
    state.messages.append({"role": "assistant", "content": assistant_answer, "turn": state.turn_index})

    response_payload = {
        "assistant_message": assistant_answer,
        "equation_hits": equation_hits,
        "assumptions": state.assumptions,
        "consistency_warnings": consistency_warnings,
    }
    return JSONResponse(content=response_payload)


@app.get("/")
def serve_index() -> FileResponse:
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(os.path.abspath(frontend_path))
