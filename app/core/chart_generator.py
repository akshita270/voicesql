from typing import Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


DATE_KEYWORDS = {"date", "month", "year", "week", "day", "quarter", "period", "time"}


def _is_date_like(col: str, df: pd.DataFrame) -> bool:
    if any(kw in col.lower() for kw in DATE_KEYWORDS):
        return True
    try:
        pd.to_datetime(df[col].dropna().head(5))
        return True
    except Exception:
        return False


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def detect_chart_type(columns: list[str], rows: list[dict]) -> str:
    if not rows or not columns:
        return "none"

    df = pd.DataFrame(rows, columns=columns)
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    n_rows = len(df)

    if n_rows == 1 and len(num_cols) == 1:
        return "stat_card"

    date_cols = [c for c in cat_cols if _is_date_like(c, df)]
    non_date_cats = [c for c in cat_cols if c not in date_cols]

    if date_cols and len(num_cols) == 1:
        return "line_chart"
    if len(num_cols) == 2 and not cat_cols:
        return "scatter_chart"
    if len(non_date_cats) == 1 and len(num_cols) == 2:
        return "grouped_bar_chart"
    if len(non_date_cats) == 1 and len(num_cols) == 1:
        if n_rows > 10:
            return "horizontal_bar_chart"
        return "bar_chart"

    return "bar_chart"


def generate_chart(
    columns: list[str],
    rows: list[dict],
    chart_type: Optional[str] = None,
) -> Optional[go.Figure]:
    """Generate a Plotly figure from query results."""
    if not rows or not columns:
        return None

    df = pd.DataFrame(rows, columns=columns)
    if chart_type is None:
        chart_type = detect_chart_type(columns, rows)

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    date_cols = [c for c in cat_cols if _is_date_like(c, df)]
    non_date_cats = [c for c in cat_cols if c not in date_cols]

    fig = None

    if chart_type == "stat_card":
        col = num_cols[0]
        val = df[col].iloc[0]
        fig = go.Figure(go.Indicator(
            mode="number",
            value=float(val),
            title={"text": col.replace("_", " ").title()},
        ))

    elif chart_type == "line_chart":
        x_col = date_cols[0] if date_cols else columns[0]
        y_col = num_cols[0]
        fig = px.line(
            df, x=x_col, y=y_col,
            labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
            markers=True,
        )

    elif chart_type == "bar_chart":
        x_col = non_date_cats[0] if non_date_cats else (date_cols[0] if date_cols else columns[0])
        y_col = num_cols[0] if num_cols else columns[1]
        fig = px.bar(
            df, x=x_col, y=y_col,
            labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
            text_auto=True,
        )

    elif chart_type == "horizontal_bar_chart":
        x_col = non_date_cats[0] if non_date_cats else columns[0]
        y_col = num_cols[0] if num_cols else columns[1]
        fig = px.bar(
            df, y=x_col, x=y_col, orientation="h",
            labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
            text_auto=True,
        )

    elif chart_type == "scatter_chart":
        x_col, y_col = num_cols[0], num_cols[1]
        fig = px.scatter(
            df, x=x_col, y=y_col,
            labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        )

    elif chart_type == "grouped_bar_chart":
        x_col = non_date_cats[0] if non_date_cats else columns[0]
        fig = px.bar(
            df, x=x_col, y=num_cols, barmode="group",
            text_auto=True,
        )

    else:
        # Fallback: try a simple bar
        if num_cols and cat_cols:
            fig = px.bar(df, x=cat_cols[0], y=num_cols[0], text_auto=True)

    if fig is not None:
        fig.update_layout(template="plotly_white", margin={"t": 40, "b": 40})

    return fig
