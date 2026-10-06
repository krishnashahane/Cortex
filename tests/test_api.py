import pytest
from pydantic import ValidationError

from cortex.app import StartRequest


def test_api_limits_iterations():
    assert StartRequest(max_iterations=30).max_iterations == 30
    with pytest.raises(ValidationError):
        StartRequest(max_iterations=31)


def test_api_limits_goal_length():
    assert len(StartRequest(goal="x" * 1000).goal) == 1000
    with pytest.raises(ValidationError):
        StartRequest(goal="x" * 1001)
