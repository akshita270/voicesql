from typing import Any, Optional
from pydantic import BaseModel


class UploadCSVResponse(BaseModel):
    table_name: str
    columns: list[str]
    row_count: int
    schema_string: str


class QueryRequest(BaseModel):
    session_id: str
    audio_base64: Optional[str] = None
    text_query: Optional[str] = None


class QueryResponse(BaseModel):
    transcription: str
    rewritten_intent: str
    sql: str
    result_rows: list[dict[str, Any]]
    result_columns: list[str]
    row_count: int
    execution_time_ms: float
    narration_text: str
    audio_base64: str
    chart_type: str
    chart_json: str
    truncated: bool = False
    error: Optional[str] = None


class ConversationTurnResponse(BaseModel):
    user_question: str
    rewritten_intent: str
    sql: str
    result_summary: str
    answer_text: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
