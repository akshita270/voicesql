import os
import re
import json
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


NARRATION_SYSTEM_PROMPT = """You are a data analyst giving a verbal briefing. Summarise this query result in 2-3 natural spoken sentences.

STRICT RULES:
- Use ONLY numbers that appear in the query result above
- Do not round numbers unless they are already rounded in the result
- Do not add context or reasons that aren't in the data
- Keep it concise — this will be spoken out loud
- Use Indian number formatting where appropriate (lakhs, crores)"""


def generate_narration(user_question: str, query_result: list[dict]) -> str:
    """Generate a spoken narration of the query result using GPT-4o."""
    result_json = json.dumps(query_result, default=str)
    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query result: {result_json}\nUser's question: {user_question}",
            },
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text, handling Indian formatting."""
    # Handle Indian number words first
    text_lower = text.lower()
    converted = _convert_indian_words(text_lower)

    # Remove commas used as thousands separators
    converted = re.sub(r'(\d),(\d{3})', r'\1\2', converted)

    # Extract all numbers (including decimals)
    matches = re.findall(r'\b\d+(?:\.\d+)?\b', converted)
    return [float(m) for m in matches]


def _convert_indian_words(text: str) -> str:
    """Replace lakh/crore textual mentions with numeric equivalents."""
    # Pattern: "42 lakhs" or "42 lakh" → "4200000"
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs)',
        lambda m: str(float(m.group(1)) * 100_000),
        text,
    )
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:crore|crores)',
        lambda m: str(float(m.group(1)) * 10_000_000),
        text,
    )
    return text


def _extract_all_numbers_from_result(query_result: list[dict]) -> list[float]:
    """Flatten all numeric values from query result rows."""
    numbers = []
    for row in query_result:
        for v in row.values():
            try:
                numbers.append(float(v))
            except (TypeError, ValueError):
                pass
    return numbers


def verify_narration(
    narration: str,
    query_result: list[dict],
    tolerance: float = 0.02,
) -> tuple[bool, Optional[str]]:
    """
    Verify that all numbers in narration exist in the query result (within tolerance).
    Returns (is_valid, mismatch_description).
    """
    narration_nums = extract_numbers(narration)
    result_nums = _extract_all_numbers_from_result(query_result)

    if not narration_nums:
        return True, None

    for num in narration_nums:
        denom = max(abs(num), 1.0)
        if not any(abs(num - r) / denom <= tolerance for r in result_nums):
            return False, f"Narration mentions {num} but data contains: {result_nums[:10]}"

    return True, None


def generate_verified_narration(
    user_question: str,
    query_result: list[dict],
    tolerance: float = None,
    max_retries: int = None,
) -> str:
    """Generate narration and verify it against query results, with self-correction."""
    if tolerance is None:
        tolerance = float(os.getenv("HALLUCINATION_TOLERANCE", "0.02"))
    if max_retries is None:
        max_retries = int(os.getenv("MAX_SQL_RETRIES", "2"))

    if not query_result:
        return "The query returned no results."

    narration = generate_narration(user_question, query_result)
    is_valid, mismatch = verify_narration(narration, query_result, tolerance)

    if is_valid:
        return narration

    for attempt in range(max_retries):
        correction_prompt = (
            f"Your previous answer mentioned a number that doesn't match the data.\n"
            f"Issue: {mismatch}\n\n"
            f"Query result: {json.dumps(query_result, default=str)}\n"
            f"User's question: {user_question}\n\n"
            "Rewrite your answer using ONLY exact figures from the result above."
        )
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
                {"role": "user", "content": correction_prompt},
            ],
            max_tokens=300,
            temperature=0,
        )
        narration = response.choices[0].message.content.strip()
        is_valid, mismatch = verify_narration(narration, query_result, tolerance)
        if is_valid:
            return narration

    # Fallback to raw template
    lines = []
    for row in query_result:
        parts = [f"{k}: {v}" for k, v in row.items()]
        lines.append(", ".join(parts))
    return "The result shows: " + "; ".join(lines)
