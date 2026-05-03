"""Fixtures for validation tests."""

import pytest
from datasets import Dataset


@pytest.fixture(autouse=True)
def ensure_plugins_loaded():
    """Ensure validation plugins are loaded before each test."""
    # Import plugins to register them
    from src.community.catalog import catalog

    catalog.reload()

    yield  # Run the test

    # No cleanup needed - plugins stay registered


@pytest.fixture
def small_dataset():
    """Small valid dataset (10 samples)."""
    return Dataset.from_dict(
        {
            "text": [
                "This is a sample text for testing validation.",
                "Another example with sufficient length.",
                "Dataset validation is working correctly.",
                "Machine learning requires quality data.",
                "Testing different text lengths here.",
                "Ensuring diversity in vocabulary usage.",
                "Quality checks prevent training issues.",
                "Good data leads to better models.",
                "Validation catches problems early.",
                "Testing edge cases is important.",
            ]
        }
    )


@pytest.fixture
def large_dataset():
    """Large dataset (1000 samples)."""
    texts = []
    for i in range(1000):
        texts.append(f"Sample text number {i} with some variety in content and length.")
    return Dataset.from_dict({"text": texts})


@pytest.fixture
def empty_samples_dataset():
    """Dataset with many empty samples."""
    return Dataset.from_dict(
        {
            "text": [
                "Valid text",
                "",
                "   ",
                "Another valid",
                "",
                "Good",
                "",
                "",
                "Text",
                "Last",
            ]
        }
    )


@pytest.fixture
def short_samples_dataset():
    """Dataset with very short samples."""
    return Dataset.from_dict({"text": ["Hi", "Ok", "Yes", "No", "Go", "Stop", "Run", "Walk", "Talk", "Read"]})


@pytest.fixture
def duplicate_dataset():
    """Dataset with many duplicates."""
    return Dataset.from_dict(
        {
            "text": [
                "Duplicate text",
                "Duplicate text",
                "Unique text one",
                "Duplicate text",
                "Unique text two",
                "Duplicate text",
                "Duplicate text",
                "Unique text three",
                "Duplicate text",
                "Duplicate text",
            ]
        }
    )


@pytest.fixture
def dpo_valid_dataset():
    """Valid DPO dataset with chosen/rejected pairs."""
    return Dataset.from_dict(
        {
            "prompt": ["Question 1", "Question 2", "Question 3"],
            "chosen": ["Good answer", "Better response", "Excellent reply"],
            "rejected": ["Bad answer", "Worse response", "Poor reply"],
        }
    )


@pytest.fixture
def dpo_invalid_format_dataset():
    """DPO dataset with missing fields."""
    return Dataset.from_dict(
        {
            "prompt": ["Question 1", "Question 2", "Question 3"],
            "chosen": ["Good answer", "Better response", "Excellent reply"],
            # Missing "rejected" field!
        }
    )


@pytest.fixture
def dpo_identical_pairs_dataset():
    """DPO dataset with identical chosen/rejected."""
    return Dataset.from_dict(
        {
            "prompt": ["Question 1", "Question 2", "Question 3"],
            "chosen": ["Same answer", "Different answer", "Same answer"],
            "rejected": ["Same answer", "Another answer", "Same answer"],
        }
    )


@pytest.fixture
def messages_format_dataset():
    """Dataset with messages format (conversational)."""
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
            ]
        }
    )
