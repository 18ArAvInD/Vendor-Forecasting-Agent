"""Tests for the pure conversation helpers (conversation.py).

Standard pytest only. No Streamlit, no boto3, no network. These cover the two
tiny building blocks used by the AI Q&A surface: explicit-vendor resolution and
short-term conversation-memory trimming.
"""

from vendor_forecasting_agent.conversation import (
    dedupe_ai_text,
    first_sentence,
    resolve_explicit_vendor,
    trim_history,
)

_IDS = ["Alpha Semiconductors", "Beta Components", "Gamma Micro"]


def test_resolve_explicit_vendor_matches_full_name():
    assert (
        resolve_explicit_vendor("Why is Gamma Micro risky?", _IDS) == "Gamma Micro"
    )


def test_resolve_explicit_vendor_matches_first_token():
    # "Gamma" alone (first token) still resolves to the full vendor id.
    assert resolve_explicit_vendor("Why is Gamma risky?", _IDS) == "Gamma Micro"


def test_resolve_explicit_vendor_is_case_insensitive():
    assert resolve_explicit_vendor("tell me about gamma micro", _IDS) == "Gamma Micro"
    assert resolve_explicit_vendor("WHY IS BETA SLOW?", _IDS) == "Beta Components"


def test_resolve_explicit_vendor_no_match_returns_none():
    # Follow-ups carrying no explicit vendor return None (caller falls back).
    assert resolve_explicit_vendor("Why?", _IDS) is None
    assert resolve_explicit_vendor("What should I do next?", _IDS) is None


def test_resolve_explicit_vendor_empty_inputs():
    assert resolve_explicit_vendor("", _IDS) is None
    assert resolve_explicit_vendor("Gamma", []) is None


def test_resolve_explicit_vendor_prefers_longest_match():
    # A question naming the full id prefers it over a bare token collision.
    assert (
        resolve_explicit_vendor("Compare Gamma Micro and Beta", _IDS) == "Gamma Micro"
    )


def test_trim_history_keeps_last_n():
    history = [{"role": "user", "text": f"m{i}"} for i in range(14)]
    trimmed = trim_history(history, 10)
    assert len(trimmed) == 10
    assert trimmed[0]["text"] == "m4"
    assert trimmed[-1]["text"] == "m13"


def test_trim_history_shorter_than_limit_unchanged():
    history = [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}]
    assert trim_history(history, 10) == history


def test_trim_history_does_not_mutate_input():
    history = [{"role": "user", "text": f"m{i}"} for i in range(12)]
    original = list(history)
    trim_history(history, 10)
    assert history == original


def test_trim_history_empty_and_nonpositive():
    assert trim_history([], 10) == []
    assert trim_history([{"role": "user", "text": "x"}], 0) == []



# ---- dedupe_ai_text: duplicate-opening removal (presentation only) ----------


def test_dedupe_removes_repeated_opening_after_key_takeaway_label():
    text = (
        "Key takeaway: Gamma is the highest-risk vendor.\n\n"
        "Gamma is the highest-risk vendor. It has six risk drivers.\n\n"
        "Business Interpretation: monitor closely."
    )
    out = dedupe_ai_text(text)
    # The takeaway label line is kept.
    assert out.startswith("Key takeaway: Gamma is the highest-risk vendor.")
    # The duplicated opening sentence is removed from the 2nd paragraph, but the
    # remainder of that paragraph is preserved.
    assert "It has six risk drivers." in out
    # The duplicated sentence should not appear twice.
    assert out.count("Gamma is the highest-risk vendor") == 1
    # Distinct later content is untouched.
    assert "Business Interpretation: monitor closely." in out


def test_dedupe_drops_second_paragraph_when_only_the_duplicate():
    text = (
        "Key takeaway: Beta warrants review.\n\n"
        "Beta warrants review.\n\n"
        "Note: lead-time deterioration."
    )
    out = dedupe_ai_text(text)
    assert out.count("Beta warrants review") == 1
    assert "Note: lead-time deterioration." in out


def test_dedupe_no_change_when_no_duplication():
    text = (
        "Alpha is the strongest supplier.\n\n"
        "Beta is usable but warrants monitoring."
    )
    assert dedupe_ai_text(text) == text


def test_dedupe_handles_single_paragraph_and_empty():
    assert dedupe_ai_text("Just one paragraph.") == "Just one paragraph."
    assert dedupe_ai_text("") == ""


def test_first_sentence_basic():
    assert first_sentence("Gamma is critical. It has drivers.") == "Gamma is critical"
    assert first_sentence("") == ""
