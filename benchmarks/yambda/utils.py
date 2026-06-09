import functools
from typing import Any


def sum_dicts(d1: dict[Any, Any], d2: dict[Any, Any]) -> dict[Any, Any]:
    if len(d1) == 0:
        return d2
    if len(d2) == 0:
        return d1

    if d1.keys() != d2.keys():
        raise ValueError("Keys do not match.")

    result = {}
    for key in d1:
        assert isinstance(d1, type(d2))

        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            result[key] = sum_dicts(d1[key], d2[key])
        else:
            result[key] = d1[key] + d2[key]

    return result


def divide_dict(d: dict[Any, Any], denom: float) -> dict[Any, Any]:
    result = {}
    for key in d:
        if isinstance(d[key], dict):
            result[key] = divide_dict(d[key], denom)
        else:
            result[key] = d[key] / denom

    return result


def mean_dicts(arr: list[dict[Any, Any]]) -> dict[Any, Any]:
    return divide_dict(functools.reduce(sum_dicts, arr), len(arr))


def argmax(a: list[Any], key=lambda x: x) -> Any:
    return max(range(len(a)), key=lambda x: key(a[x]))
