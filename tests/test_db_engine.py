import csv
import os
import tempfile

import pytest

from app.core.db_engine import DBEngine


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "test_data.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "region", "sale_amount"])
        writer.writeheader()
        writer.writerows([
            {"order_id": 1, "region": "North", "sale_amount": 5000},
            {"order_id": 2, "region": "South", "sale_amount": 3000},
            {"order_id": 3, "region": "North", "sale_amount": 7000},
        ])
    return str(path)


@pytest.fixture
def engine(sample_csv):
    db = DBEngine()
    db.load_csv(sample_csv)
    return db


def test_csv_loads_correctly(sample_csv):
    db = DBEngine()
    result = db.load_csv(sample_csv)
    assert result.success
    assert result.row_count == 3
    assert "order_id" in result.columns
    assert "sale_amount" in result.columns


def test_query_returns_correct_structure(engine):
    result = engine.execute_query("SELECT region, SUM(sale_amount) AS total FROM user_data GROUP BY region ORDER BY region")
    assert result.success
    assert result.row_count == 2
    assert "region" in result.columns
    assert "total" in result.columns
    assert isinstance(result.rows, list)
    assert isinstance(result.rows[0], dict)


def test_execution_time_captured(engine):
    result = engine.execute_query("SELECT COUNT(*) FROM user_data")
    assert result.success
    assert result.execution_time_ms >= 0.0


def test_invalid_query_returns_error(engine):
    result = engine.execute_query("SELECT * FROM nonexistent_table_xyz")
    assert not result.success
    assert result.error is not None


def test_aggregate_query_correct_values(engine):
    result = engine.execute_query("SELECT SUM(sale_amount) AS total FROM user_data")
    assert result.success
    assert result.rows[0]["total"] == 15000


def test_row_count_vs_returned_rows(engine):
    result = engine.execute_query("SELECT * FROM user_data")
    assert result.success
    assert result.row_count == len(result.rows)
