import os
import re
from datetime import datetime
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


INTENT_SYSTEM_PROMPT = """You are a query normaliser. Convert the user's spoken question into a clean, precise data question.
Rules:
- Resolve relative time references: "last month" → specific month name if current date is known
- Remove filler words: "uh", "um", "like", "you know"
- Standardise comparison phrases: "vs", "versus", "compared to" → "compare X and Y"
- Resolve pronouns and references using conversation history: "that country", "it", "those products", "the same period" → replace with the actual value from prior context
- Keep the question concise and unambiguous
Return ONLY the rewritten question. No explanation."""

SQL_SYSTEM_PROMPT_TEMPLATE = """You are an expert SQL analyst. Generate a single valid SQL query to answer the user's question.

DATABASE SCHEMA:
{schema_string}

CONVERSATION HISTORY:
{conversation_history}

CURRENT DATE: {current_date}

RULES:
1. Use ONLY column names that exist in the schema above. Never invent column names.
2. The table name is always: user_data
3. Return ONLY the SQL query. No explanation, no markdown, no backticks.
4. For date filtering, use DuckDB date syntax.
5. Always include appropriate GROUP BY when using aggregate functions.
6. Limit results to 100 rows maximum unless user asks for all.
7. Use conversation history to resolve references like "that country", "it", "those products" — replace them with actual values from prior turns.
8. Only return CANNOT_ANSWER if the question is completely unrelated to the dataset (e.g. asking about weather, news). Never return CANNOT_ANSWER just because a column has a different name than expected — look for equivalent columns.
9. Follow all DATA-SPECIFIC RULES listed in the schema above — they are auto-detected from the actual dataset and override general assumptions.
10. CRITICAL: Never use CAST(column AS TIMESTAMP) for date strings. Always use strptime(column, 'format') with the format from DATA-SPECIFIC RULES. Example: strftime(strptime("InvoiceDate", '%m/%d/%Y %H:%M'), '%Y-%m') AS Month"""


def rewrite_intent(
    user_question: str,
    current_date: Optional[str] = None,
    conversation_history: str = "",
) -> str:
    """Rewrite a raw spoken question into a clean, precise data question."""
    date_ctx = f"\nCurrent date: {current_date}" if current_date else ""
    history_ctx = f"\n\nCONVERSATION HISTORY (use this to resolve pronouns and references):\n{conversation_history}" if conversation_history else ""
    system = INTENT_SYSTEM_PROMPT + date_ctx + history_ctx
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_question},
        ],
        max_tokens=200,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_sql(
    rewritten_question: str,
    schema_string: str,
    conversation_history: str = "",
) -> str:
    """Generate SQL from a rewritten question and schema. Returns SQL or 'CANNOT_ANSWER'."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    system = SQL_SYSTEM_PROMPT_TEMPLATE.format(
        schema_string=schema_string,
        conversation_history=conversation_history or "No prior conversation.",
        current_date=current_date,
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": rewritten_question},
        ],
        max_tokens=500,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    return _strip_markdown(raw)


def _strip_markdown(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def fix_date_casts(sql: str, date_col_formats: dict[str, str]) -> str:
    """Replace CAST(col AS TIMESTAMP) with strptime(col, fmt) for known date columns."""
    for col, fmt in date_col_formats.items():
        pattern = re.compile(
            r'CAST\s*\(\s*["\']?' + re.escape(col) + r'["\']?\s+AS\s+TIMESTAMP\s*\)',
            re.IGNORECASE,
        )
        replacement = f"strptime(\"{col}\", '{fmt}')"
        sql = pattern.sub(replacement, sql)
    return sql


def correct_sql(
    original_sql: str,
    error_reason: str,
    schema_string: str,
    conversation_history: str = "",
) -> str:
    """Ask GPT-4o to fix an invalid SQL given the error reason."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    system = SQL_SYSTEM_PROMPT_TEMPLATE.format(
        schema_string=schema_string,
        conversation_history=conversation_history or "No prior conversation.",
        current_date=current_date,
    )
    correction_prompt = (
        f"The following SQL query has an error:\n\n{original_sql}\n\n"
        f"Error: {error_reason}\n\n"
        "Please fix the SQL query. Return ONLY the corrected SQL. No explanation, no markdown."
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": correction_prompt},
        ],
        max_tokens=500,
        temperature=0,
    )
    return _strip_markdown(response.choices[0].message.content.strip())
