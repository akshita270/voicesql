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

**Live demo:** [akshita27-voicesql.hf.space](https://akshita27-voicesql.hf.space)

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
[Intent Rewriter — gpt-4o-mini]
            │
            ▼
   [Semantic Cache lookup]
      (embedding similarity
       against past questions
       in this session)
            │
      ┌─────┴──────┐
      │            │
  [Cache hit]  [Cache miss]
      │            │
      │     [Text-to-SQL — GPT-4o
      │      streamed + Schema + Memory]
      │            │
      │            ▼
      │     [SQL Validator — sqlglot + column check]
      │            │
      │     ┌──────┴───────┐
      │     │              │
      │ [DuckDB Execute] [Self-correct loop]
      │     │
      └─────┤
            ▼
   [Hallucination Guard — gpt-4o-mini]
   (skipped on cache hit if result
    rows match the cached entry)
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

## Cost & Latency Optimizations

This project was deliberately optimized beyond a basic working version, since every question triggers multiple LLM calls:

- **Model tiering** — `gpt-4o-mini` handles intent rewriting and answer narration (simple rewriting/summarisation tasks); `gpt-4o` is reserved only for SQL generation, where real schema reasoning is needed.
- **Semantic caching** — each resolved question is embedded (`text-embedding-3-small`) and checked against past questions in the session via cosine similarity (threshold 0.93). A close match skips SQL generation entirely and reuses the cached SQL — re-executed fresh against DuckDB so results stay correct. If the re-executed result also matches the cached rows, the narration call is skipped too. A cache hit can save up to 2 of the 3 LLM calls per query.
- **Prompt caching** — the SQL-generation system prompt is structured so the static portion (rules + full schema) always comes first and the dynamic portion (conversation history, current date) comes last, letting OpenAI's automatic prompt caching discount/speed up the repeated static prefix across every turn in a session.
- **Streaming** — SQL generation streams tokens live into the UI instead of a blank wait, improving perceived latency without changing total wall-clock time.
- **No redundant API calls** — the semantic cache reuses an embedding it already computed during lookup instead of re-embedding the same text again when storing a new entry.

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
| LLM (reasoning) | GPT-4o | Text-to-SQL generation and self-correction |
| LLM (lightweight) | GPT-4o-mini | Intent rewriting + answer narration |
| Semantic cache | text-embedding-3-small + cosine similarity | Skip redundant SQL/narration calls for paraphrased repeat questions |
| Conversation memory | Custom sliding window | Last 6 turns of context |
| SQL validation | sqlglot | Parse and validate SQL before execution |
| Database | DuckDB | In-memory CSV querying |
| Schema inference | pandas | Auto-detected column types, date formats, and business rules |
| Hallucination guard | Custom Python module | Verify numbers in narration match DB result |
| Charts | Plotly | Auto-generated visualisations |
| Frontend | Streamlit | Full UI, streamed SQL generation |
| Deployment | Docker | Containerised app, deployable on Hugging Face Spaces |

---

## Known Limitations

- **faster-whisper large-v3** model is ~3 GB — use `WHISPER_MODEL_SIZE=base` for free-tier hosting
- **DuckDB is in-memory** — data is not persisted between server restarts
- **Session state is per-browser-tab** — not suitable for multi-user production scale (use Redis for that); the semantic cache and conversation memory both reset when the session ends
- **No spoken output** — narration is text-only; the answer is verified for accuracy but not read aloud
- **Semantic cache is in-memory per session** — it does not persist across page reloads or get shared between users

---

## Future Improvements

- PostgreSQL / BigQuery connector (not just CSV)
- Spoken answer playback (TTS)
- Export conversation as PDF report
- Persistent sessions and cross-session semantic cache via Redis + PostgreSQL
- Multi-file joins (upload and query across multiple CSVs)
- User authentication and per-user session isolation
- Offline LLM-as-judge evaluation harness to catch SQL-generation regressions when prompts change, run outside the live request path so it doesn't add cost/latency to user queries
- Schema caching by file hash to skip re-running auto-detection when the same CSV is re-uploaded
