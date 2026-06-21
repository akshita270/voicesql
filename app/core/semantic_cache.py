import os
import math
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: Optional[OpenAI] = None
EMBED_MODEL = "text-embedding-3-small"
DEFAULT_THRESHOLD = 0.93


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def get_embedding(text: str) -> list[float]:
    response = _get_client().embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class CacheEntry:
    rewritten_intent: str
    embedding: list[float]
    sql: str
    result_rows: list[dict]
    result_columns: list[str]
    narration: str
    chart_type: str


@dataclass
class SemanticCache:
    """
    Caches (rewritten_intent -> SQL/result/narration) within a session.
    Lookup is by embedding similarity, not exact string match, so paraphrased
    questions ("total sales by country" vs "what are the total sales per country")
    still hit the cache.
    """
    threshold: float = DEFAULT_THRESHOLD
    entries: list[CacheEntry] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    _pending_embedding: Optional[tuple[str, list[float]]] = field(default=None, repr=False)

    def find(self, rewritten_intent: str) -> Optional[CacheEntry]:
        if not self.entries:
            self.misses += 1
            return None

        query_embedding = get_embedding(rewritten_intent)
        # Stash it so add() can reuse it instead of re-embedding the same text.
        self._pending_embedding = (rewritten_intent, query_embedding)

        best_entry = None
        best_score = 0.0

        for entry in self.entries:
            score = cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.threshold:
            self.hits += 1
            return best_entry

        self.misses += 1
        return None

    def add(
        self,
        rewritten_intent: str,
        sql: str,
        result_rows: list[dict],
        result_columns: list[str],
        narration: str,
        chart_type: str,
    ) -> None:
        if self._pending_embedding is not None and self._pending_embedding[0] == rewritten_intent:
            embedding = self._pending_embedding[1]
        else:
            embedding = get_embedding(rewritten_intent)
        self._pending_embedding = None

        self.entries.append(CacheEntry(
            rewritten_intent=rewritten_intent,
            embedding=embedding,
            sql=sql,
            result_rows=result_rows,
            result_columns=result_columns,
            narration=narration,
            chart_type=chart_type,
        ))

    def clear(self) -> None:
        self.entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate_pct": hit_rate}
