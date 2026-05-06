"""TemporalDescriptor pressure tests.

Stress points:
- UTC timezone enforcement (naive and non-UTC rejected)
- Both-or-neither start/end validation (half-open ranges rejected)
- Ordering invariant (start <= end)
- Positive resolution and observation_count >= 1
- Frozen, hashable, equatable semantics
- Three distinct semantic states: None, all-None TemporalDescriptor, populated
"""

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from quarry_core.artifact import TemporalDescriptor


def test_construct_all_none_is_unknown_state():
    """TemporalDescriptor() constructs with no error; all four fields are None."""
    descriptor = TemporalDescriptor()
    assert descriptor.start is None
    assert descriptor.end is None
    assert descriptor.resolution is None
    assert descriptor.observation_count is None


def test_construct_with_utc_instant():
    """TemporalDescriptor(start=t, end=t) constructs with no error; start == end."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    descriptor = TemporalDescriptor(start=t, end=t)
    assert descriptor.start == t
    assert descriptor.end == t
    assert descriptor.start == descriptor.end


def test_construct_with_utc_range():
    """TemporalDescriptor(start=t0, end=t1, resolution=timedelta(minutes=15), observation_count=140256) constructs with no error."""
    t0 = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc)
    descriptor = TemporalDescriptor(
        start=t0,
        end=t1,
        resolution=timedelta(minutes=15),
        observation_count=140256,
    )
    assert descriptor.start == t0
    assert descriptor.end == t1
    assert descriptor.resolution == timedelta(minutes=15)
    assert descriptor.observation_count == 140256


def test_naive_start_rejected():
    """TemporalDescriptor(start=datetime(2024, 6, 15), end=datetime(2024, 6, 15)) raises ValueError with "timezone-aware" in the message."""
    naive = datetime(2024, 6, 15)
    with pytest.raises(ValueError, match="timezone-aware"):
        TemporalDescriptor(start=naive, end=naive)


def test_naive_end_rejected():
    """TemporalDescriptor(start=datetime(2024, 6, 15, tzinfo=timezone.utc), end=datetime(2024, 6, 16)) raises ValueError with "timezone-aware" in the message."""
    utc = datetime(2024, 6, 15, tzinfo=timezone.utc)
    naive = datetime(2024, 6, 16)
    with pytest.raises(ValueError, match="timezone-aware"):
        TemporalDescriptor(start=utc, end=naive)


def test_non_utc_start_rejected():
    """Pass a datetime with a non-UTC timezone(timedelta(hours=-5)); raises ValueError with "UTC" in the message."""
    est = timezone(timedelta(hours=-5))
    t_est = datetime(2024, 6, 15, 10, 30, tzinfo=est)
    t_utc = datetime(2024, 6, 15, 15, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="UTC"):
        TemporalDescriptor(start=t_est, end=t_utc)


def test_half_open_range_rejected_start_only():
    """TemporalDescriptor(start=t, end=None) raises ValueError mentioning "both be set or both be None"."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="both be set or both be None"):
        TemporalDescriptor(start=t, end=None)


def test_half_open_range_rejected_end_only():
    """TemporalDescriptor(start=None, end=t) raises ValueError."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="both be set or both be None"):
        TemporalDescriptor(start=None, end=t)


def test_inverted_range_rejected():
    """TemporalDescriptor(start=t1, end=t0) where t1 > t0 raises ValueError with "start ... must be <=" in the message."""
    t0 = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="start .* must be <="):
        TemporalDescriptor(start=t1, end=t0)


def test_zero_observation_count_rejected():
    """observation_count=0 raises ValueError mentioning ">= 1"."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=">= 1"):
        TemporalDescriptor(start=t, end=t, observation_count=0)


def test_negative_observation_count_rejected():
    """observation_count=-1 raises ValueError."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=">= 1"):
        TemporalDescriptor(start=t, end=t, observation_count=-1)


def test_zero_resolution_rejected():
    """resolution=timedelta(0) raises ValueError mentioning "positive"."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="positive"):
        TemporalDescriptor(start=t, end=t, resolution=timedelta(0))


def test_negative_resolution_rejected():
    """resolution=timedelta(seconds=-1) raises ValueError."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="positive"):
        TemporalDescriptor(start=t, end=t, resolution=timedelta(seconds=-1))


def test_descriptor_is_frozen():
    """Attempting descriptor.start = some_other_dt raises dataclasses.FrozenInstanceError."""
    t = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    descriptor = TemporalDescriptor(start=t, end=t)
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.start = datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc)


def test_descriptor_is_hashable():
    """Two equal descriptors produce the same hash(); descriptor can be used as a dict key."""
    t0 = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc)
    d1 = TemporalDescriptor(
        start=t0,
        end=t1,
        resolution=timedelta(minutes=15),
        observation_count=100,
    )
    d2 = TemporalDescriptor(
        start=t0,
        end=t1,
        resolution=timedelta(minutes=15),
        observation_count=100,
    )
    assert hash(d1) == hash(d2)
    # Can be used as dict key
    mapping = {d1: "value"}
    assert mapping[d2] == "value"


def test_descriptor_equality():
    """Two identical TemporalDescriptor instances compare ==; differing instances compare !=."""
    t0 = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc)
    d1 = TemporalDescriptor(start=t0, end=t1, resolution=timedelta(minutes=15))
    d2 = TemporalDescriptor(start=t0, end=t1, resolution=timedelta(minutes=15))
    d3 = TemporalDescriptor(start=t0, end=t1, resolution=timedelta(minutes=30))
    assert d1 == d2
    assert d1 != d3
    assert d1 != "not a descriptor"
