import re
from dataclasses import dataclass
from typing import Optional

import sqlglot
import sqlglot.expressions as exp


FORBIDDEN_KEYWORDS = {"DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE", "EXEC"}
ALLOWED_TABLE = "user_data"


@dataclass
class SQLValidationResult:
    is_valid: bool
    error_reason: Optional[str] = None
    cleaned_sql: Optional[str] = None


def _clean_sql(sql: str) -> str:
    """Strip markdown fences and surrounding whitespace."""
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def validate_sql(sql: str, known_columns: list[str]) -> SQLValidationResult:
    """Validate SQL for safety, correctness, and schema adherence."""
    cleaned = _clean_sql(sql)

    # Reject forbidden DML/DDL keywords
    upper = cleaned.upper()
    for kw in FORBIDDEN_KEYWORDS:
        pattern = r'\b' + kw + r'\b'
        if re.search(pattern, upper):
            return SQLValidationResult(
                is_valid=False,
                error_reason=f"Forbidden keyword detected: {kw}. Only SELECT queries are allowed.",
            )

    # Parse with sqlglot
    try:
        statements = sqlglot.parse(cleaned, dialect="duckdb")
    except Exception as e:
        return SQLValidationResult(is_valid=False, error_reason=f"SQL parse error: {e}")

    if not statements or statements[0] is None:
        return SQLValidationResult(is_valid=False, error_reason="Could not parse SQL statement.")

    statement = statements[0]

    # Ensure it's a SELECT
    if not isinstance(statement, exp.Select):
        return SQLValidationResult(
            is_valid=False,
            error_reason="Only SELECT statements are permitted.",
        )

    # Validate table names
    tables_used = {t.name.lower() for t in statement.find_all(exp.Table)}
    for t in tables_used:
        if t and t != ALLOWED_TABLE:
            return SQLValidationResult(
                is_valid=False,
                error_reason=f"Query references unknown table '{t}'. Only '{ALLOWED_TABLE}' is available.",
            )

    # Column name validation is handled by DuckDB at execution time.
    # Static analysis here causes false positives on aliases and subquery references.

    return SQLValidationResult(is_valid=True, cleaned_sql=cleaned)
