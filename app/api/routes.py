import base64
import json
import logging
import os
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import (
    ConversationTurnResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadCSVResponse,
)
from app.core.asr import transcribe_file_bytes
from app.core.chart_generator import detect_chart_type, generate_chart
from app.core.db_engine import DBEngine
from app.core.hallucination_guard import generate_verified_narration
from app.core.memory import ConversationMemory
from app.core.schema_builder import build_schema_string, get_column_names
from app.core.sql_validator import validate_sql
from app.core.text_to_sql import correct_sql, generate_sql, rewrite_intent
from app.core.tts import synthesise_speech

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session stores (use Redis/DB for production)
_sessions: dict[str, dict[str, Any]] = {}


def _get_or_create_session(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "db": DBEngine(),
            "memory": ConversationMemory(k=int(os.getenv("MAX_CONVERSATION_TURNS", "6"))),
            "schema_string": "",
            "columns": [],
        }
    return _sessions[session_id]


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.post("/upload-csv", response_model=UploadCSVResponse)
async def upload_csv(session_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    session = _get_or_create_session(session_id)
    schema_string, _ = build_schema_string(tmp_path)
    result = session["db"].load_csv(tmp_path)

    if not result.success:
        raise HTTPException(status_code=500, detail=f"Failed to load CSV: {result.error}")

    session["schema_string"] = schema_string
    session["columns"] = result.columns

    return UploadCSVResponse(
        table_name="user_data",
        columns=result.columns,
        row_count=result.row_count,
        schema_string=schema_string,
    )


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    session = _get_or_create_session(req.session_id)
    schema_string = session["schema_string"]
    columns = session["columns"]
    memory: ConversationMemory = session["memory"]
    db: DBEngine = session["db"]

    if not schema_string:
        raise HTTPException(status_code=400, detail="No CSV loaded for this session.")

    # 1. Resolve input text
    if req.text_query:
        user_question = req.text_query
    elif req.audio_base64:
        audio_bytes = base64.b64decode(req.audio_base64)
        asr_result = transcribe_file_bytes(audio_bytes)
        if asr_result.error or not asr_result.text:
            raise HTTPException(status_code=422, detail=asr_result.error or "Transcription failed.")
        user_question = asr_result.text
    else:
        raise HTTPException(status_code=400, detail="Provide audio_base64 or text_query.")

    # 2. Rewrite intent
    rewritten = rewrite_intent(user_question)

    # 3. Generate SQL (with retry loop)
    history_str = memory.get_history_string()
    sql = generate_sql(rewritten, schema_string, history_str)

    if sql == "CANNOT_ANSWER":
        col_list = ", ".join(columns)
        narration = f"I can't answer that from the available data. The columns I can see are: {col_list}"
        return QueryResponse(
            transcription=user_question, rewritten_intent=rewritten, sql="",
            result_rows=[], result_columns=[], row_count=0, execution_time_ms=0,
            narration_text=narration, audio_base64="", chart_type="none", chart_json="{}",
        )

    # 4. Validate SQL with retry
    max_retries = int(os.getenv("MAX_SQL_RETRIES", "2"))
    validation = validate_sql(sql, columns)
    for attempt in range(max_retries):
        if validation.is_valid:
            break
        sql = correct_sql(sql, validation.error_reason, schema_string, history_str)
        validation = validate_sql(sql, columns)

    if not validation.is_valid:
        raise HTTPException(status_code=422, detail=validation.error_reason)

    # 5. Execute
    query_result = db.execute_query(validation.cleaned_sql)
    if not query_result.success:
        raise HTTPException(status_code=500, detail=query_result.error)

    # 6. Narration + guard
    narration = generate_verified_narration(user_question, query_result.rows)

    # 7. TTS
    tts_result = synthesise_speech(narration)
    audio_b64 = base64.b64encode(tts_result.audio_bytes).decode() if tts_result.audio_bytes else ""

    # 8. Chart
    chart_type = detect_chart_type(query_result.columns, query_result.rows)
    fig = generate_chart(query_result.columns, query_result.rows, chart_type)
    chart_json = fig.to_json() if fig else "{}"

    # 9. Update memory
    memory.add_turn(
        user_question=user_question,
        rewritten_intent=rewritten,
        sql=validation.cleaned_sql,
        result_rows=query_result.rows,
        result_columns=query_result.columns,
        answer_text=narration,
    )

    return QueryResponse(
        transcription=user_question,
        rewritten_intent=rewritten,
        sql=validation.cleaned_sql,
        result_rows=query_result.rows,
        result_columns=query_result.columns,
        row_count=query_result.row_count,
        execution_time_ms=query_result.execution_time_ms,
        narration_text=narration,
        audio_base64=audio_b64,
        chart_type=chart_type,
        chart_json=chart_json,
        truncated=query_result.truncated,
    )


@router.get("/conversation-history/{session_id}", response_model=list[ConversationTurnResponse])
async def conversation_history(session_id: str):
    session = _get_or_create_session(session_id)
    memory: ConversationMemory = session["memory"]
    return [
        ConversationTurnResponse(
            user_question=t.user_question,
            rewritten_intent=t.rewritten_intent,
            sql=t.sql,
            result_summary=t.result_summary,
            answer_text=t.answer_text,
        )
        for t in memory.all_turns()
    ]


@router.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    session = _get_or_create_session(session_id)
    session["memory"].clear()
    return {"status": "cleared", "session_id": session_id}
