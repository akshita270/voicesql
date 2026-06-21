---
title: VoiceSQL
emoji: ▪️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# VoiceSQL — Talk to Your Data

VoiceSQL lets you upload any CSV file and ask questions about it in plain English — by voice or text. It transcribes your question, generates SQL, runs it against your data, verifies the answer for accuracy, and renders a chart — all in one seamless flow.

---

## Architecture

```
[Browser audio / typed text]
            │
            ▼
   [ASR — faster-whisper]
            │
            ▼
[Intent Rewriter — GPT-4o]
            │
            ▼
[Text-to-SQL — GPT-4o + Schema + Memory]
            │
            ▼
[SQL Validator — sqlglot + column check]
            │
    ┌───────┴────────┐
    │                │
[DuckDB Execute]  [Self-correct loop]
    │
    ▼
[Hallucination Guard]
    │
    ▼
[Plotly Chart + Written Answer]
    │
    ▼
[Streamlit UI]
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd voicesql

# 2. Copy and fill in API keys
cp .env.example .env
# Edit .env — add OPENAI_API_KEY

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run Streamlit directly
streamlit run frontend/streamlit_app.py

# 4b. Or use Docker
docker build -t voicesql .
docker run -p 7860:7860 --env-file .env voicesql
```

---

## How to Use

1. **Upload** — Upload a CSV file (any size, any columns)
2. **Check schema** — Expand "Schema & data preview" to see detected columns and auto-inferred rules
3. **Ask by voice** — Click the mic widget and ask your question naturally
4. **Or type** — Use the text input if you prefer typing
5. **View results** — See the generated SQL, a chart, and a verified written answer
6. **Follow up** — Ask follow-up questions; VoiceSQL remembers the last 6 turns
7. **Reset** — Click "Reset" to start a new session

---

## Example Questions

Ask these against a sales CSV with columns like `region`, `sale_amount`, `sale_date`, `product`:

1. "What are total sales by region?"
2. "Show me the top 5 products by revenue"
3. "Compare sales in Q1 versus Q2 this year"
4. "Which region had the highest average sale amount?"
5. "How many orders were placed in December?"
6. "Show monthly sales trend for last year"
7. "What percentage of sales came from the North region?"
8. "Which product has the most returns?" *(if returns column exists)*

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Voice input | Browser mic via `st.audio_input` | Records question in-browser |
| ASR | faster-whisper (base/large-v3) | Local speech-to-text |
| LLM | GPT-4o | Intent rewriting + Text-to-SQL + narration |
| Conversation memory | Custom sliding window | Last 6 turns of context |
| SQL validation | sqlglot | Parse and validate SQL before execution |
| Database | DuckDB | In-memory CSV querying |
| Schema inference | pandas | Auto-detected column types, date formats, and business rules |
| Hallucination guard | Custom Python module | Verify numbers in narration match DB result |
| Charts | Plotly | Auto-generated visualisations |
| Frontend | Streamlit | Full UI |
| Deployment | Docker | Containerised app, deployable on Hugging Face Spaces |

---

## Known Limitations

- **faster-whisper large-v3** model is ~3 GB — use `WHISPER_MODEL_SIZE=base` for free-tier hosting
- **DuckDB is in-memory** — data is not persisted between server restarts
- **Session state is per-browser-tab** — not suitable for multi-user production scale (use Redis for that)
- **No spoken output** — narration is text-only; the answer is verified for accuracy but not read aloud

---

## Future Improvements

- PostgreSQL / BigQuery connector (not just CSV)
- Spoken answer playback (TTS)
- Export conversation as PDF report
- Persistent sessions via Redis + PostgreSQL
- Multi-file joins (upload and query across multiple CSVs)
- User authentication and per-user session isolation
