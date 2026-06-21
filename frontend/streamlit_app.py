from __future__ import annotations
import os
import sys
import tempfile
import uuid

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.core.asr import transcribe_file_bytes
from app.core.chart_generator import detect_chart_type, generate_chart
from app.core.db_engine import DBEngine
from app.core.hallucination_guard import generate_verified_narration
from app.core.memory import ConversationMemory
from app.core.schema_builder import build_schema_string, get_column_names, get_date_column_formats
from app.core.sql_validator import validate_sql
from app.core.text_to_sql import correct_sql, fix_date_casts, generate_sql, rewrite_intent
from app.core.semantic_cache import SemanticCache

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceSQL",
    page_icon="▪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS — light, editorial, single accent ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #1a1a1a;
    --paper: #fafaf8;
    --line: #e0ddd5;
    --muted: #8a8678;
    --accent: #a8581f;
    --accent-soft: #f3e8dd;
    --card: #ffffff;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

.stApp {
    background: var(--paper);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.main .block-container {
    max-width: 680px;
    margin: 0 auto;
    padding: 3rem 1.5rem 5rem;
}

/* ── Masthead ── */
.masthead {
    text-align: center;
    padding-bottom: 2rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--line);
}
.masthead-mark {
    font-family: 'Source Serif 4', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -1px;
    margin: 0;
}
.masthead-mark span { color: var(--accent); }
.masthead-tag {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Setup row (upload, inline, top) ── */
.setup-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 2rem;
}
.setup-meta {
    font-size: 0.82rem;
    color: var(--muted);
}
.setup-meta b { color: var(--ink); font-weight: 600; }

/* ── Welcome ── */
.welcome {
    text-align: center;
    padding: 3.5rem 1rem;
}
.welcome h2 {
    font-family: 'Source Serif 4', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.5rem;
}
.welcome p {
    color: var(--muted);
    font-size: 0.92rem;
    max-width: 380px;
    margin: 0 auto 2rem;
}
.steps {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 1rem;
}
.step {
    text-align: center;
    max-width: 130px;
}
.step-n {
    font-family: 'Source Serif 4', serif;
    font-size: 1.6rem;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 6px;
}
.step-t {
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.4;
}

/* ── Ask block ── */
.ask-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.or-row {
    text-align: center;
    color: var(--ink);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 0.6rem 0;
    position: relative;
}

/* ── Result entries (no card box — editorial separation by rule) ── */
.entry {
    padding: 1.6rem 0;
    border-bottom: 1px solid var(--line);
}
.entry:first-of-type { padding-top: 0; }
.entry-q {
    font-family: 'Source Serif 4', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.3rem;
}
.entry-intent {
    font-size: 0.8rem;
    color: var(--muted);
    font-style: italic;
    margin-bottom: 1rem;
}
.tag-row { margin-bottom: 0.8rem; }
.tag {
    display: inline-block;
    font-size: 0.72rem;
    color: var(--accent);
    background: var(--accent-soft);
    border-radius: 4px;
    padding: 2px 8px;
    margin-right: 6px;
    font-weight: 500;
}
.answer-line {
    border-left: 2px solid var(--accent);
    padding-left: 14px;
    margin-top: 1rem;
    font-size: 0.93rem;
    color: #3a3a3a;
    line-height: 1.6;
}
.err-line {
    border-left: 2px solid #b04545;
    padding-left: 14px;
    font-size: 0.88rem;
    color: #8a3030;
}
.sql-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0.8rem 0 0.4rem;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.3rem !important;
}
.stButton > button:hover { background: var(--accent) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: 5px !important;
    color: var(--ink) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stAudioInput"], .stFileUploader {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: 5px !important;
}

/* ── Expander as plain entries ── */
[data-testid="stExpander"] {
    background: var(--paper) !important;
    border: none !important;
    border-bottom: 1px solid var(--line) !important;
    border-radius: 0 !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: var(--ink) !important;
    background: var(--paper) !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    padding: 1rem 0 !important;
}
[data-testid="stExpanderDetails"] {
    background: var(--paper) !important;
}

.stCodeBlock, pre, pre code, .stCodeBlock code {
    background: #f4f2ec !important;
    border: 1px solid var(--line) !important;
    border-radius: 5px !important;
    color: #2a2a2a !important;
}
.stCodeBlock span, pre span {
    color: #2a2a2a !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 5px;
}

[data-testid="stStatus"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: 5px !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--line); }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "csv_loaded": False,
        "schema_string": "",
        "csv_filename": "",
        "csv_row_count": 0,
        "csv_col_count": 0,
        "duckdb_engine": None,
        "conversation_memory": ConversationMemory(k=int(os.getenv("MAX_CONVERSATION_TURNS", "6"))),
        "semantic_cache": SemanticCache(),
        "query_history": [],
        "session_id": str(uuid.uuid4()),
        "total_queries": 0,
        "total_exec_ms": 0.0,
        "columns": [],
        "df_preview": None,
        "date_col_formats": {},
        "last_audio_hash": None,
        "last_text_query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-mark">Voice<span>SQL</span></div>
    <div class="masthead-tag">Ask your data anything</div>
</div>
""", unsafe_allow_html=True)


# ── Setup row ─────────────────────────────────────────────────────────────────
if not st.session_state.csv_loaded:
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        try:
            raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_bytes = raw_bytes.decode("latin-1").encode("utf-8")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        with st.spinner("Reading dataset…"):
            schema_string, df = build_schema_string(tmp_path)
            engine = DBEngine()
            result = engine.load_csv(tmp_path)

        if result.success:
            st.session_state.csv_loaded = True
            st.session_state.schema_string = schema_string
            st.session_state.csv_filename = uploaded.name
            st.session_state.csv_row_count = result.row_count
            st.session_state.csv_col_count = len(result.columns)
            st.session_state.duckdb_engine = engine
            st.session_state.columns = result.columns
            st.session_state.df_preview = df.head(5)
            st.session_state.date_col_formats = get_date_column_formats(df)
            st.rerun()
        else:
            st.error(f"Failed to load: {result.error}")

    st.markdown("""
    <div class="welcome">
        <h2>Talk to your data</h2>
        <p>Upload a CSV and ask questions in plain English — by voice or text. No SQL knowledge required.</p>
        <div class="steps">
            <div class="step"><div class="step-n">1</div><div class="step-t">Upload a CSV file above</div></div>
            <div class="step"><div class="step-n">2</div><div class="step-t">Record or type a question</div></div>
            <div class="step"><div class="step-n">3</div><div class="step-t">Get SQL, a chart, and an answer</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

else:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        cache_stats = st.session_state.semantic_cache.stats()
        cache_note = f" &nbsp;·&nbsp; cache hits: {cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']}" if (cache_stats['hits'] + cache_stats['misses']) > 0 else ""
        st.markdown(f"""
        <div class="setup-row">
            <div class="setup-meta">▪ <b>{st.session_state.csv_filename}</b> &nbsp;·&nbsp; {st.session_state.csv_row_count:,} rows &nbsp;·&nbsp; {st.session_state.csv_col_count} columns &nbsp;·&nbsp; {st.session_state.total_queries} queries asked{cache_note}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        if st.button("Reset", use_container_width=True):
            for key in ["csv_loaded", "schema_string", "csv_filename", "query_history"]:
                st.session_state[key] = False if key == "csv_loaded" else ("" if "string" in key or "name" in key else [])
            st.session_state.conversation_memory.clear()
            st.session_state.semantic_cache.clear()
            st.session_state.total_queries = 0
            st.rerun()

    with st.expander("Schema & data preview"):
        st.code(st.session_state.schema_string, language="text")
        if st.session_state.df_preview is not None:
            st.dataframe(st.session_state.df_preview, use_container_width=True, hide_index=True)


# ── Ask section ───────────────────────────────────────────────────────────────
st.markdown('<div class="ask-label">Ask a question</div>', unsafe_allow_html=True)

try:
    audio_input = st.audio_input("", label_visibility="collapsed")
except AttributeError:
    st.caption("Upgrade Streamlit for mic recording: `pip install --upgrade streamlit`")
    audio_input = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], label_visibility="collapsed")

st.markdown('<div class="or-row">or type instead</div>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    text_query = st.text_input("", placeholder="What are total sales by region?", label_visibility="collapsed")
with col2:
    ask_text = st.button("Ask", use_container_width=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(user_question: str):
    schema_string = st.session_state.schema_string
    columns = st.session_state.columns
    memory: ConversationMemory = st.session_state.conversation_memory
    cache: SemanticCache = st.session_state.semantic_cache
    db: DBEngine = st.session_state.duckdb_engine
    max_retries = int(os.getenv("MAX_SQL_RETRIES", "2"))
    date_formats = st.session_state.get("date_col_formats", {})

    record = {
        "user_question": user_question,
        "rewritten_intent": "",
        "sql": "",
        "result_rows": [],
        "result_columns": [],
        "row_count": 0,
        "execution_time_ms": 0.0,
        "narration_text": "",
        "chart_type": "none",
        "chart_fig": None,
        "error": None,
        "truncated": False,
    }

    status = st.status("Processing…", expanded=True)
    try:
        with status:
            st.write("Understanding your question…")
            history_str = memory.get_history_string()
            rewritten = rewrite_intent(user_question, conversation_history=history_str)
            record["rewritten_intent"] = rewritten

            st.write("Checking cache…")
            cached = cache.find(rewritten)

            if cached is not None:
                st.write("Cache hit — skipping SQL generation")
                sql = cached.sql
            else:
                st.write("Generating SQL…")
                sql_placeholder = st.empty()
                sql = generate_sql(
                    rewritten, schema_string, history_str,
                    on_token=lambda partial: sql_placeholder.code(partial, language="sql"),
                )
                sql_placeholder.empty()
                sql = fix_date_casts(sql, date_formats)

            if sql == "CANNOT_ANSWER":
                msg = f"I can't answer that from the available data. Columns: {', '.join(columns)}"
                record["narration_text"] = msg
                record["error"] = msg
                status.update(label="Done", state="complete")
                st.session_state.query_history.append(record)
                return

            validation = validate_sql(sql, columns)
            for _ in range(max_retries):
                if validation.is_valid:
                    break
                sql = fix_date_casts(correct_sql(sql, validation.error_reason, schema_string, history_str), date_formats)
                validation = validate_sql(sql, columns)

            if not validation.is_valid:
                record["error"] = validation.error_reason
                status.update(label="Validation failed", state="error")
                st.session_state.query_history.append(record)
                return

            record["sql"] = validation.cleaned_sql

            st.write("Running query…")
            qr = db.execute_query(validation.cleaned_sql)
            if not qr.success:
                record["error"] = qr.error
                status.update(label="Query failed", state="error")
                st.session_state.query_history.append(record)
                return

            record.update({
                "result_rows": qr.rows,
                "result_columns": qr.columns,
                "row_count": qr.row_count,
                "execution_time_ms": qr.execution_time_ms,
                "truncated": qr.truncated,
            })

            if cached is not None and cached.result_rows == qr.rows:
                st.write("Reusing cached answer…")
                narration = cached.narration
            else:
                st.write("Preparing answer…")
                narration = generate_verified_narration(user_question, qr.rows)
            record["narration_text"] = narration

            chart_type = detect_chart_type(qr.columns, qr.rows)
            record["chart_type"] = chart_type
            record["chart_fig"] = generate_chart(qr.columns, qr.rows, chart_type)

            if cached is None:
                cache.add(
                    rewritten_intent=rewritten,
                    sql=validation.cleaned_sql,
                    result_rows=qr.rows,
                    result_columns=qr.columns,
                    narration=narration,
                    chart_type=chart_type,
                )

            memory.add_turn(
                user_question=user_question,
                rewritten_intent=rewritten,
                sql=validation.cleaned_sql,
                result_rows=qr.rows,
                result_columns=qr.columns,
                answer_text=narration,
            )
            st.session_state.total_queries += 1
            st.session_state.total_exec_ms += qr.execution_time_ms

        status.update(label="Done", state="complete")
    except Exception as e:
        record["error"] = str(e)
        status.update(label=f"Error: {e}", state="error")

    st.session_state.query_history.append(record)


# ── Trigger ───────────────────────────────────────────────────────────────────
if audio_input is not None:
    audio_bytes = audio_input.read() if hasattr(audio_input, "read") else bytes(audio_input)
    audio_hash = hash(audio_bytes)
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Transcribing…"):
            asr = transcribe_file_bytes(audio_bytes)
        if asr.error or not asr.text:
            st.warning(asr.error or "Couldn't understand the audio. Please try again.")
        else:
            run_pipeline(asr.text)
            st.rerun()

elif ask_text and text_query.strip():
    if text_query.strip() != st.session_state.last_text_query:
        st.session_state.last_text_query = text_query.strip()
        run_pipeline(text_query.strip())
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
history = st.session_state.query_history

if history:
    import pandas as pd
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    for i, record in enumerate(reversed(history)):
        is_latest = i == 0
        label = record['user_question'][:90] + ('…' if len(record['user_question']) > 90 else '')

        with st.expander(label, expanded=is_latest):

            if record.get("error") and not record.get("result_rows"):
                st.markdown(f'<div class="err-line">{record["error"]}</div>', unsafe_allow_html=True)
                continue

            if record.get("rewritten_intent") and record["rewritten_intent"] != record["user_question"]:
                st.markdown(f'<div class="entry-intent">Interpreted as: {record["rewritten_intent"]}</div>', unsafe_allow_html=True)

            if record.get("sql"):
                st.markdown('<div class="sql-label">SQL</div>', unsafe_allow_html=True)
                st.code(record["sql"], language="sql")
                st.markdown(f"""
                <div class="tag-row">
                    <span class="tag">{record['row_count']:,} rows</span>
                    <span class="tag">{record['execution_time_ms']:.1f} ms</span>
                    {"<span class='tag'>truncated to 500</span>" if record.get('truncated') else ""}
                </div>
                """, unsafe_allow_html=True)

            if record.get("chart_fig") is not None:
                record["chart_fig"].update_layout(
                    template="simple_white",
                    font_family="Inter",
                    margin={"t": 20, "b": 20},
                )
                st.plotly_chart(record["chart_fig"], use_container_width=True, key=f"chart_{len(history)-1-i}")

            if record.get("result_rows"):
                st.dataframe(pd.DataFrame(record["result_rows"]), use_container_width=True, hide_index=True)

            if record.get("narration_text") and not record["narration_text"].startswith("The result shows:"):
                st.markdown(f'<div class="answer-line">{record["narration_text"]}</div>', unsafe_allow_html=True)
