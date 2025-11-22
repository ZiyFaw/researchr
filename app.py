import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI


def get_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")


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


def parse_response_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    # Strip code fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Escape backslashes that are not valid JSON escapes (e.g., LaTeX like \frac, \alpha, \)).
    cleaned_safe = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", cleaned)
    for candidate in (cleaned_safe, cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(candidate, strict=False)
            except Exception:
                continue

    # Fallback: find first JSON object in the text.
    for blob in (cleaned_safe, cleaned):
        match = re.search(r"\{.*\}", blob, flags=re.S)
        if match:
            for candidate in (match.group(0),):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        return json.loads(candidate, strict=False)
                    except Exception:
                        continue

    return {
        "chat_answer": text,
        "equation_hits": [],
        "assumptions_delta": [],
        "consistency_warnings": [],
    }


def extract_structured_from_response(response: Any) -> Optional[Dict[str, Any]]:
    """Prefer structured output_json content if present."""
    output = getattr(response, "output", None) or []
    for item in output:
        contents = getattr(item, "content", None) or []
        for content in contents:
            if getattr(content, "type", "") == "output_json":
                parsed = getattr(content, "parsed", None)
                if isinstance(parsed, dict):
                    return parsed
    return None


def call_model(client: OpenAI, messages: List[Dict[str, Any]], assumptions: List[Dict[str, Any]]) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Current assumptions (for consistency checking): {json.dumps(assumptions)}",
        },
    ]
    history.extend(messages)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=history,
        tools=[{"type": "web_search"}],
        max_output_tokens=8192,
        response_format={"type": "json_object"},
    )

    structured = extract_structured_from_response(response)
    if structured:
        return structured

    text: Optional[str] = getattr(response, "output_text", None)
    if not text:
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


def merge_assumptions(assumptions: List[Dict[str, Any]], deltas: List[Dict[str, Any]], turn_index: int) -> List[Dict[str, Any]]:
    updated = list(assumptions)
    for delta in deltas or []:
        text = (delta.get("text") or "").strip()
        if not text:
            continue
        if any(a["text"].lower() == text.lower() for a in updated):
            continue
        updated.append(
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "type": delta.get("type") or "",
                "variables": delta.get("variables") or [],
                "source": delta.get("source") or "model",
                "created_at_turn": turn_index,
            }
        )
    return updated


def merge_hits(existing: List[Dict[str, Any]], new_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for hit in (existing or []) + (new_hits or []):
        key = ((hit.get("url") or "").strip().lower(), (hit.get("title") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def merge_warnings(existing: List[Dict[str, Any]], new_warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for warn in (existing or []) + (new_warnings or []):
        key = (
            (warn.get("problem_type") or "").strip().lower(),
            (warn.get("assumption_text") or "").strip().lower(),
            (warn.get("explanation") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(warn)
    return merged


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("assumptions", [])
    st.session_state.setdefault("equation_hits", [])
    st.session_state.setdefault("equation_hits_history", [])
    st.session_state.setdefault("turn_index", 0)
    st.session_state.setdefault("latest_warnings", [])
    st.session_state.setdefault("warnings_history", [])


def main() -> None:
    st.set_page_config(page_title="ResearchR", layout="wide")
    init_state()

    st.markdown(
        """
        <style>
        .chat-scroll {
            max-height: 70vh;
            overflow-y: auto;
            padding-right: 8px;
        }
        .sidebar-scroll {
            max-height: 70vh;
            overflow-y: auto;
            padding-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    api_key = get_api_key()
    if not api_key:
        st.error("Set OPENAI_API_KEY via environment or Streamlit secrets to run this app.")
        return

    client = OpenAI(api_key=api_key)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.title("ResearchR")
        st.caption("Equation-level retrieval + assumption tracking (powered by OpenAI Responses API)")

        chat_box = st.container()
        chat_box.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with chat_box.chat_message(msg["role"]):
                chat_box.markdown(msg["content"])
        chat_box.markdown("</div>", unsafe_allow_html=True)

        user_input = st.chat_input("Paste an equation or ask a technical question...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.turn_index += 1

            model_payload = call_model(client, st.session_state.messages, st.session_state.assumptions)
            assistant_answer = model_payload.get("chat_answer", "")
            equation_hits = model_payload.get("equation_hits") or []
            assumptions_delta = model_payload.get("assumptions_delta") or []
            consistency_warnings = model_payload.get("consistency_warnings") or []

            st.session_state.assumptions = merge_assumptions(
                st.session_state.assumptions, assumptions_delta, st.session_state.turn_index
            )
            st.session_state.equation_hits = equation_hits
            st.session_state.equation_hits_history = merge_hits(
                st.session_state.equation_hits_history, equation_hits
            )
            st.session_state.latest_warnings = consistency_warnings
            st.session_state.warnings_history = merge_warnings(
                st.session_state.warnings_history, consistency_warnings
            )
            st.session_state.messages.append({"role": "assistant", "content": assistant_answer})
            # Rerun so the just-appended messages render immediately.
            st.rerun()

    with col_side:
        side_box = st.container()
        side_box.markdown('<div class="sidebar-scroll">', unsafe_allow_html=True)

        side_box.subheader("Equation matches")
        hits_to_show = st.session_state.equation_hits_history or st.session_state.equation_hits
        if hits_to_show:
            for hit in hits_to_show:
                url = hit.get("url") or "#"
                label = hit.get("title") or url or "Source"
                side_box.markdown(f"**[{label}]({url})**")
                if hit.get("url"):
                    side_box.caption(hit["url"])
                meta_bits = " • ".join(filter(None, [hit.get("authors", ""), hit.get("year", "")]))
                if meta_bits:
                    side_box.caption(meta_bits)
                if hit.get("context_snippet"):
                    side_box.write(hit["context_snippet"])
                if hit.get("relation_to_query_equation"):
                    side_box.info(hit["relation_to_query_equation"])
                side_box.divider()
        else:
            side_box.caption("No matches yet.")

        side_box.subheader("Current assumptions")
        if st.session_state.assumptions:
            for a in st.session_state.assumptions:
                side_box.markdown(f"- {a['text']} *(turn {a.get('created_at_turn', '?')})*")
        else:
            side_box.caption("None yet.")

        side_box.subheader("Consistency warnings")
        all_warnings = st.session_state.get("warnings_history", []) or st.session_state.get("latest_warnings", [])
        if all_warnings:
            for w in all_warnings:
                side_box.error(f"{w.get('problem_type', 'Issue')}: {w.get('explanation', '')}")
        else:
            side_box.caption("None.")

        side_box.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
