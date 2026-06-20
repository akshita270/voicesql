# VoiceSQL — Talk to Your Data

VoiceSQL lets you upload any CSV file and ask questions about it entirely by voice. It listens to your question, converts speech to text, generates SQL, runs it against your data, verifies the answer for accuracy, speaks the result out loud, and renders a chart — all in one seamless flow.

---

## Architecture

```
[Microphone / Browser audio]
        │
        ▼
[VAD — webrtcvad]   →   [ASR — faster-whisper]
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
                  [DuckDB Execute]   [Self-correct loop]
                        │
                        ▼
              [Hallucination Guard]
                        │
                  ┌─────┴──────┐
                  │            │
              [ElevenLabs    [Plotly
               TTS voice]     Chart]
                  │
                  ▼
         [Streamlit UI / FastAPI response]
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd voicesql

# 2. Copy and fill in API keys
cp .env.example .env
# Edit .env — add OPENAI_API_KEY and ELEVENLABS_API_KEY

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run Streamlit (standalone — recommended for development)
streamlit run frontend/streamlit_app.py

# 4b. Run FastAPI backend separately
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4c. Or use Docker
docker build -t voicesql .
docker run -p 8000:8000 -p 8501:8501 --env-file .env voicesql
```

---

## How to Use

1. **Upload** — Use the sidebar to upload a CSV file (any size, any columns)
2. **Check schema** — Expand the "Schema" panel to see detected columns and sample values
3. **Ask by voice** — Click the microphone widget and ask your question naturally
4. **Or type** — Use the text input if you prefer typing
5. **View results** — See the generated SQL, a chart, and hear the spoken answer
6. **Follow up** — Ask follow-up questions; VoiceSQL remembers the last 6 turns
7. **Clear** — Hit "Clear conversation" in the sidebar to start fresh

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
| Voice capture | PyAudio + webrtcvad | Mic audio, auto-stop on silence |
| ASR | faster-whisper (base/large-v3) | Local speech-to-text |
| LLM | GPT-4o | Intent rewriting + Text-to-SQL + narration |
| Conversation memory | Custom sliding window | Last 6 turns of context |
| SQL validation | sqlglot | Parse and validate SQL before execution |
| Database | DuckDB | In-memory CSV querying |
| Schema inference | pandas | Column types + sample values |
| Hallucination guard | Custom Python module | Verify numbers in narration match DB result |
| TTS | ElevenLabs SDK (+ pyttsx3 fallback) | Spoken answers |
| Charts | Plotly | Auto-generated visualisations |
| Backend | FastAPI + uvicorn | Async REST API |
| Frontend | Streamlit | Full UI |
| Deployment | Docker | Containerised app |

---

## Known Limitations

- **Mic capture (VAD)** requires `pyaudio` which needs `portaudio` installed on the host; browser recording via `st.audio_input` works without this
- **faster-whisper large-v3** model is ~3 GB — use `WHISPER_MODEL_SIZE=base` for development
- **DuckDB is in-memory** — data is not persisted between server restarts
- **ElevenLabs free tier** has a monthly character limit; the app falls back to `pyttsx3` automatically
- **Multi-user sessions** use an in-memory dict — not suitable for production scale (use Redis)
- **Date parsing** depends on DuckDB's `read_csv_auto` — exotic date formats may need manual casting

---

## Future Improvements

- PostgreSQL / BigQuery connector (not just CSV)
- Real-time streaming TTS playback
- Voice command to navigate history ("repeat that" / "show the chart again")
- Export conversation as PDF report
- Persistent sessions via Redis + PostgreSQL
- Fine-tuned Text-to-SQL model for domain-specific schemas
- Multi-file joins (upload and query across multiple CSVs)
- User authentication and per-user session isolation
