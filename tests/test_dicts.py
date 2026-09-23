import pytest

from parsonaut.dicts import flatten_dict, unflatten_dict


def test_flatten_dict():
    assert flatten_dict({"1": "2", "3": {"4": "5"}}) == {"1": "2", "3.4": "5"}


def test_unflatten_dict():
    assert unflatten_dict({"1": "2", "3.4": "5"}) == {"1": "2", "3": {"4": "5"}}


def test_unflatten_dict_rejects_a_key_used_both_ways():
    with pytest.raises(ValueError, match="both as a value and as a group"):
        unflatten_dict({"a.b": 1, "a": 2})
    with pytest.raises(ValueError, match="both as a value and as a group"):
        unflatten_dict({"a": 2, "a.b": 1})


def test_flatten_dict_rejects_a_key_specified_twice():
    with pytest.raises(ValueError, match="specified more than once"):
        flatten_dict({"b.b": "set", "b": {"b": "other"}})
    with pytest.raises(ValueError, match="specified more than once"):
        flatten_dict({"b": {"b": "other"}, "b.b": "set"})


def test_flatten_dict_keeps_empty_groups():
    # Dropping them turned `copy({"a": {}})` into a silent no-op.
    assert flatten_dict({"a": {}, "b": 1}) == {"a": {}, "b": 1}
