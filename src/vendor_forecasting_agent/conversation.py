"""Pure, dependency-light conversation helpers (conversation.py).

Two tiny, side-effect-free helpers that support the AI Q&A conversation surface
without pulling in Streamlit or any provider/AWS dependency:

* :func:`resolve_explicit_vendor` — data-driven context resolution: does the
  user's question explicitly name one of the known vendors? (case-insensitive).
* :func:`trim_history` — bound the short-term conversation memory to the most
  recent N messages.

Boundaries / guarantees
-----------------------
* **Streamlit-free and boto3-free.** So this module is safe to import from both
  the Streamlit UI (``ui/app.py``) and the test-suite, and it keeps the
  import-discipline guard happy.
* **Pure / read-only.** Neither helper mutates its inputs; both return new
  values. No I/O, no network, no LLM. Determinism: identical inputs always
  produce identical outputs.
* **No authoritative computation.** These helpers only resolve *what the user is
  referring to* and *how much history to keep*; they never touch risk scores,
  forecasts, inventory, recommendations, or what-if numbers.
"""

from typing import Optional

__all__ = [
    "resolve_explicit_vendor",
    "trim_history",
    "first_sentence",
    "dedupe_ai_text",
]


def resolve_explicit_vendor(
    question: str,
    vendor_ids: list[str],
) -> Optional[str]:
    """Return the vendor id explicitly named in ``question``, else ``None``.

    Purely data-driven (there are NO hardcoded vendor names): each candidate is
    taken from ``vendor_ids``. A vendor is considered explicitly named when its
    id, or the first whitespace-delimited token of its id (e.g. "Gamma" from
    "Gamma Micro"), appears as a case-insensitive substring of the question.

    When more than one vendor matches, the one whose matched text is longest is
    preferred (a full id like "Gamma Micro" wins over a bare token "Gamma"),
    ties broken by the vendor id ascending for determinism. Follow-up questions
    that carry no explicit vendor (e.g. "Why?" / "it") return ``None`` so the
    caller can fall back to the currently selected vendor and to conversation
    history for reference resolution.
    """
    if not question or not vendor_ids:
        return None

    lowered_q = question.lower()
    matches: list[tuple[int, str]] = []  # (matched-text length, vendor_id)

    for vendor_id in vendor_ids:
        if not vendor_id:
            continue
        candidates = {vendor_id}
        first_token = vendor_id.split()[0] if vendor_id.split() else ""
        if first_token:
            candidates.add(first_token)

        best_len = 0
        for candidate in candidates:
            cand = candidate.lower().strip()
            if cand and cand in lowered_q:
                best_len = max(best_len, len(cand))

        if best_len > 0:
            matches.append((best_len, vendor_id))

    if not matches:
        return None

    # Longest matched text wins; ties -> smallest vendor_id (deterministic).
    matches.sort(key=lambda pair: (-pair[0], pair[1]))
    return matches[0][1]


def trim_history(
    history: list[dict],
    max_messages: int = 10,
) -> list[dict]:
    """Return the most recent ``max_messages`` messages of ``history``.

    ``history`` is a small list of turn dicts (e.g. ``{"role": ..., "text":
    ...}``); this bounds the short-term conversation memory so the embedded
    context stays small. The input list is never mutated — a new list slice is
    returned. ``max_messages <= 0`` yields an empty list; ``None`` history yields
    an empty list.
    """
    if not history:
        return []
    if max_messages <= 0:
        return []
    return list(history[-max_messages:])


def first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (used only for duplicate detection)."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    return flat.split(". ", 1)[0].strip().rstrip(".").strip()


def dedupe_ai_text(text: str) -> str:
    """Collapse a model-emitted duplicated opening for display (presentation only).

    Some AI responses begin with a short "Key takeaway: X" (or "Key takeaway"
    label) whose sentence X is then immediately repeated as the opening of the
    next paragraph, so the same sentence renders twice. This removes ONLY that
    redundant repetition: it keeps the "Key takeaway" line and the rest of the
    body, but drops (or trims) a following paragraph that merely restates the
    takeaway sentence. It never removes distinct content (Key Changes / Business
    Interpretation / Note / later paragraphs are untouched) and never fabricates
    text. Pure and non-mutating.
    """
    if not text:
        return text
    paragraphs = list(str(text).split("\n\n"))
    if len(paragraphs) < 2:
        return text

    def _norm(s: str) -> str:
        return " ".join(str(s or "").split()).strip().lower().rstrip(".")

    first = paragraphs[0].strip()
    low_first = first.lower()
    if low_first.startswith("key takeaway"):
        after_label = first.split(":", 1)[1] if ":" in first else first
        takeaway_sentence = _norm(first_sentence(after_label))
    else:
        takeaway_sentence = _norm(first_sentence(first))

    if takeaway_sentence:
        second = paragraphs[1].strip()
        second_first = _norm(first_sentence(second))
        if second_first and second_first == takeaway_sentence:
            flat_second = " ".join(second.split())
            parts = flat_second.split(". ", 1)
            remainder = parts[1].strip() if len(parts) > 1 else ""
            if remainder:
                paragraphs[1] = remainder
            else:
                del paragraphs[1]
            return "\n\n".join(paragraphs).strip()

    return text
