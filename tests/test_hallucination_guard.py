import pytest
from app.core.hallucination_guard import (
    extract_numbers,
    verify_narration,
)


def test_matching_numbers_pass():
    narration = "Total sales were 4200000."
    result = [{"total_sales": 4200000}]
    valid, _ = verify_narration(narration, result)
    assert valid


def test_mismatched_numbers_fail():
    narration = "Total sales were 9999999."
    result = [{"total_sales": 4200000}]
    valid, mismatch = verify_narration(narration, result)
    assert not valid
    assert mismatch is not None


def test_indian_lakh_formatting():
    # "42 lakhs" = 4,200,000
    narration = "December sales were 42 lakhs."
    result = [{"total_sales": 4200000.0}]
    valid, _ = verify_narration(narration, result)
    assert valid


def test_indian_crore_formatting():
    # "1.5 crores" = 15,000,000
    narration = "Revenue was 1.5 crores."
    result = [{"revenue": 15000000.0}]
    valid, _ = verify_narration(narration, result)
    assert valid


def test_tolerance_small_rounding():
    # 1% rounding difference should pass with default 2% tolerance
    narration = "Total is 100000."
    result = [{"total": 101000.0}]  # 1% higher
    valid, _ = verify_narration(narration, result, tolerance=0.02)
    assert valid


def test_tolerance_exceeded_fails():
    narration = "Total is 100000."
    result = [{"total": 110000.0}]  # 10% higher — exceeds 2% tolerance
    valid, _ = verify_narration(narration, result, tolerance=0.02)
    assert not valid


def test_extract_numbers_basic():
    nums = extract_numbers("Sales were 1,200 and profit was 350.5")
    assert 1200.0 in nums
    assert 350.5 in nums


def test_extract_numbers_empty():
    assert extract_numbers("No numbers here at all.") == []


def test_no_numbers_in_narration_always_passes():
    narration = "The data shows no significant trend."
    result = [{"count": 42}]
    valid, _ = verify_narration(narration, result)
    assert valid
