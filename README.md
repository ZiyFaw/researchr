# ResearchR MVP

Minimal web app for equation-level retrieval and assumption tracking using the OpenAI Responses API + `web_search`. The primary entrypoint is the Streamlit app (`app.py`), suitable for local use and Streamlit Cloud. A FastAPI prototype remains in `backend/` for reference but is not required for deployment.

## Features
- Paste equations or technical questions; get a normal answer.
- When equations are detected, the model calls the `web_search` tool to surface papers/sources that contain the same or equivalent equations (no fabricated links).
- Extracts assumptions each turn and flags consistency warnings against prior assumptions.
- Streamlit UI with chat + sidebar for equation matches, assumptions, and warnings.

## Prerequisites
- Python 3.9+
- An OpenAI API key with access to a model that supports `web_search` (e.g., `gpt-4.1` or newer).

## Local setup (Streamlit)
1. Clone this repo.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # or source .venv/bin/activate on macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Provide your API key (keep it private):
   - Either set an environment variable `OPENAI_API_KEY`
   - Or create `.env` from `.env.example` and fill in the key (do **not** commit).
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Add a secret named `OPENAI_API_KEY` in the Streamlit app settings (`App settings` → `Secrets`):
   ```toml
   OPENAI_API_KEY = "sk-***"
   OPENAI_MODEL = "gpt-4.1"  # optional override
   ```
   Secrets are kept server-side and are not exposed in the repo.

## Configuration
- `OPENAI_API_KEY`: required.
- `OPENAI_MODEL`: override model name (default `gpt-4.1`).

## Notes
- Streamlit session state keeps messages, assumptions, equation hits, and latest warnings per user session.
- The model is instructed to return structured JSON with `chat_answer`, `equation_hits`, `assumptions_delta`, and `consistency_warnings`.
- The legacy FastAPI prototype remains in `backend/main.py` if you prefer an API server; otherwise, Streamlit (`app.py`) is the recommended path.
