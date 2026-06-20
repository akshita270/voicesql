from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class ConversationTurn:
    user_question: str
    rewritten_intent: str
    sql: str
    result_summary: str
    answer_text: str


class ConversationMemory:
    """Sliding-window conversation memory (last k turns)."""

    def __init__(self, k: int = 6):
        self.k = k
        self._turns: deque[ConversationTurn] = deque(maxlen=k)

    def add_turn(
        self,
        user_question: str,
        rewritten_intent: str,
        sql: str,
        result_rows: list[dict],
        result_columns: list[str],
        answer_text: str,
    ) -> None:
        summary = _summarise_result(result_rows, result_columns)
        turn = ConversationTurn(
            user_question=user_question,
            rewritten_intent=rewritten_intent,
            sql=sql,
            result_summary=summary,
            answer_text=answer_text,
        )
        self._turns.append(turn)

    def get_history_string(self) -> str:
        if not self._turns:
            return ""
        parts = []
        for i, turn in enumerate(self._turns, 1):
            parts.append(
                f"Turn {i}:\n"
                f"  User asked: \"{turn.user_question}\"\n"
                f"  SQL run: {turn.sql}\n"
                f"  Result summary: {turn.result_summary}\n"
                f"  Answer given: \"{turn.answer_text}\""
            )
        return "\n\n".join(parts)

    def clear(self) -> None:
        self._turns.clear()

    def all_turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)


def _summarise_result(rows: list[dict], columns: list[str]) -> str:
    """Produce a compact summary of query results to inject into prompts."""
    if not rows:
        return "No results returned."
    if len(rows) == 1:
        return ", ".join(f"{k}={v}" for k, v in rows[0].items())
    if len(rows) <= 5:
        lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in rows]
        return "; ".join(lines)
    # Summarise aggregate-style for large results
    return (
        f"{len(rows)} rows returned. "
        f"First row: {', '.join(f'{k}={v}' for k, v in rows[0].items())}. "
        f"Last row: {', '.join(f'{k}={v}' for k, v in rows[-1].items())}."
    )
