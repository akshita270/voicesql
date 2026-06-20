from __future__ import annotations
import re
import pandas as pd
from pathlib import Path


def _try_utf8(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.read().decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _detect_encoding(path: str) -> str:
    return "utf-8" if _try_utf8(path) else "latin-1"


def _detect_date_format(series: pd.Series) -> str | None:
    """Try common date formats on sample values, return format string if found."""
    formats = [
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
    ]
    sample = series.dropna().head(5).astype(str).tolist()
    for fmt in formats:
        try:
            pd.to_datetime(sample, format=fmt)
            return fmt
        except Exception:
            continue
    # Try pandas inference
    try:
        pd.to_datetime(sample, infer_datetime_format=True)
        return "auto"
    except Exception:
        return None


def _is_date_column(col: str, series: pd.Series) -> bool:
    """Heuristic: column name suggests date, or values parse as dates."""
    date_keywords = {"date", "time", "datetime", "timestamp", "created", "updated", "at", "on"}
    if any(kw in col.lower() for kw in date_keywords):
        return True
    if series.dtype == object:
        return _detect_date_format(series) is not None
    return False


def _build_dynamic_rules(df: pd.DataFrame) -> list[str]:
    """
    Infer dataset-specific SQL rules from the actual data.
    These rules are generic and work for ANY CSV, not just e-commerce.
    """
    rules = []

    for col in df.columns:
        series = df[col]

        # Detect date/time columns stored as strings
        if series.dtype == object and _is_date_column(col, series):
            fmt = _detect_date_format(series)
            if fmt and fmt != "auto":
                rules.append(
                    f"- Column '{col}' contains date/time values stored as text with format '{fmt}'. "
                    f"Always parse using strptime: strptime(\"{col}\", '{fmt}'). "
                    f"Example: strftime(strptime(\"{col}\", '{fmt}'), '%Y-%m') AS Month"
                )
            elif fmt == "auto":
                rules.append(
                    f"- Column '{col}' contains date/time values stored as text. "
                    f"Always parse using: TRY_CAST(\"{col}\" AS TIMESTAMP). "
                    f"Example: strftime(TRY_CAST(\"{col}\" AS TIMESTAMP), '%Y-%m') AS Month"
                )

        # Detect potential ID/code columns (should not be summed)
        if series.dtype in ["int64", "float64"]:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["id", "code", "no", "num", "number", "ref"]):
                rules.append(
                    f"- Column '{col}' appears to be an identifier/code. "
                    f"Use COUNT(DISTINCT \"{col}\") to count unique values, not SUM."
                )

    # Detect if revenue/sales needs to be computed from two columns
    qty_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["qty", "quantity", "units", "count"])]
    price_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["price", "rate", "cost", "amount", "value"])]
    revenue_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["revenue", "sales", "total", "income"])]

    if qty_cols and price_cols and not revenue_cols:
        rules.append(
            f"- There is no pre-computed revenue/sales column. "
            f"Calculate it as: SUM(\"{qty_cols[0]}\" * \"{price_cols[0]}\")"
        )

    return rules


def build_schema_string(csv_path: str, table_name: str = "user_data") -> tuple[str, pd.DataFrame]:
    """Read CSV and build a schema string for LLM injection."""
    encoding = _detect_encoding(csv_path)
    df = pd.read_csv(csv_path, encoding=encoding)

    lines = [f"Table: {table_name}", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        samples = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): sample values {samples}")

    # Add auto-detected rules
    rules = _build_dynamic_rules(df)
    if rules:
        lines.append("\nDATA-SPECIFIC RULES (auto-detected from this dataset):")
        lines.extend(rules)

    schema_string = "\n".join(lines)
    return schema_string, df


def infer_schema_from_dataframe(df: pd.DataFrame, table_name: str = "user_data") -> str:
    """Build schema string from an existing DataFrame."""
    lines = [f"Table: {table_name}", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        samples = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): sample values {samples}")
    rules = _build_dynamic_rules(df)
    if rules:
        lines.append("\nDATA-SPECIFIC RULES (auto-detected from this dataset):")
        lines.extend(rules)
    return "\n".join(lines)


def get_date_column_formats(df: pd.DataFrame) -> dict[str, str]:
    """Return a mapping of date column name → detected strptime format."""
    result = {}
    for col in df.columns:
        series = df[col]
        if series.dtype == object and _is_date_column(col, series):
            fmt = _detect_date_format(series)
            if fmt and fmt != "auto":
                result[col] = fmt
    return result


def get_column_names(schema_string: str) -> list[str]:
    """Extract column names from a schema string."""
    columns = []
    for line in schema_string.splitlines():
        line = line.strip()
        if line.startswith("- ") and "(" in line and "sample values" in line:
            col_part = line[2:]
            col_name = col_part.split("(")[0].strip()
            columns.append(col_name)
    return columns
