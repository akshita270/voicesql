import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import duckdb


@dataclass
class QueryResult:
    success: bool
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    truncated: bool = False


MAX_ROWS = 500


class DBEngine:
    def __init__(self):
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = threading.Lock()

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(database=":memory:")
        return self._conn

    def load_csv(self, csv_path: str, table_name: str = "user_data") -> QueryResult:
        """Load a CSV file into DuckDB as a table."""
        start = time.perf_counter()
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS "
                    f"SELECT * FROM read_csv_auto('{csv_path}', header=True, ignore_errors=True)"
                )
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                cols = [desc[0] for desc in conn.execute(f"DESCRIBE {table_name}").fetchall()]
                elapsed = (time.perf_counter() - start) * 1000
                return QueryResult(
                    success=True,
                    columns=cols,
                    row_count=count,
                    execution_time_ms=elapsed,
                )
            except Exception as e:
                return QueryResult(success=False, error=str(e),
                                   execution_time_ms=(time.perf_counter() - start) * 1000)

    def execute_query(self, sql: str) -> QueryResult:
        """Execute a SQL query and return structured results."""
        start = time.perf_counter()
        with self._lock:
            try:
                conn = self._get_conn()
                result = conn.execute(sql)
                columns = [desc[0] for desc in result.description]
                all_rows = result.fetchall()
                truncated = len(all_rows) > MAX_ROWS
                rows_to_return = all_rows[:MAX_ROWS]
                rows = [dict(zip(columns, row)) for row in rows_to_return]
                elapsed = (time.perf_counter() - start) * 1000
                return QueryResult(
                    success=True,
                    rows=rows,
                    columns=columns,
                    row_count=len(all_rows),
                    execution_time_ms=elapsed,
                    truncated=truncated,
                )
            except Exception as e:
                return QueryResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
