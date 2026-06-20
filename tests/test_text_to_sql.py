"""
Tests for text_to_sql module.
These tests mock the OpenAI client to avoid real API calls.
"""
import pytest
from unittest.mock import MagicMock, patch


SAMPLE_SCHEMA = """Table: user_data
Columns:
  - order_id (int64): sample values [1001, 1002, 1003]
  - region (object): sample values ['North', 'South', 'East']
  - sale_amount (float64): sample values [4200.0, 8750.0, 3100.0]
  - sale_date (object): sample values ['2024-01-15', '2024-01-16', '2024-01-17']"""


def _make_chat_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("app.core.text_to_sql._get_client")
def test_intent_rewrite_removes_filler(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_chat_response(
        "Compare total sales for April 2024 vs March 2024"
    )
    mock_get_client.return_value = mock_client

    from app.core.text_to_sql import rewrite_intent
    result = rewrite_intent("uh, how did we do last month versus the month before that?")
    assert "uh" not in result
    assert len(result) > 0


@patch("app.core.text_to_sql._get_client")
def test_generate_sql_returns_clean_sql(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_chat_response(
        "SELECT region, SUM(sale_amount) FROM user_data GROUP BY region"
    )
    mock_get_client.return_value = mock_client

    from app.core.text_to_sql import generate_sql
    sql = generate_sql("Total sales by region", SAMPLE_SCHEMA)
    assert "```" not in sql
    assert "SELECT" in sql.upper()


@patch("app.core.text_to_sql._get_client")
def test_generate_sql_strips_markdown(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_chat_response(
        "```sql\nSELECT * FROM user_data\n```"
    )
    mock_get_client.return_value = mock_client

    from app.core.text_to_sql import generate_sql
    sql = generate_sql("Show all data", SAMPLE_SCHEMA)
    assert "```" not in sql
    assert sql.strip().upper().startswith("SELECT")


@patch("app.core.text_to_sql._get_client")
def test_cannot_answer_returned_for_irrelevant(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_chat_response("CANNOT_ANSWER")
    mock_get_client.return_value = mock_client

    from app.core.text_to_sql import generate_sql
    result = generate_sql("What is the capital of France?", SAMPLE_SCHEMA)
    assert result == "CANNOT_ANSWER"
