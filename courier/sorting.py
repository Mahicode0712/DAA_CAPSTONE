"""Sorting algorithms used to prioritize courier parcels."""

from collections.abc import Callable, Sequence
from typing import TypeVar

from courier.models import Parcel

T = TypeVar("T")
ParcelKey = Callable[[Parcel], tuple[float, int]]


def parcel_priority(parcel: Parcel) -> tuple[float, int]:
    """Return deadline ascending and value descending; time and space are O(1)."""
    return (parcel.deadline, -parcel.value)


def merge_sort(items: Sequence[T], key: Callable[[T], object]) -> list[T]:
    """Return a stable sorted copy using merge sort; time is O(n log n), space is O(n)."""
    result = list(items)
    auxiliary = list(result)
    width = 1
    while width < len(result):
        for start in range(0, len(result), width * 2):
            middle = min(start + width, len(result))
            end = min(start + width * 2, len(result))
            left = start
            right = middle
            for destination in range(start, end):
                if left < middle and (
                    right >= end or key(result[left]) <= key(result[right])
                ):
                    auxiliary[destination] = result[left]
                    left += 1
                else:
                    auxiliary[destination] = result[right]
                    right += 1
        result, auxiliary = auxiliary, result
        width *= 2
    return result


def insertion_sort(items: Sequence[T], key: Callable[[T], object]) -> list[T]:
    """Return a stable insertion-sorted copy used as a baseline; time is O(n^2), space is O(n)."""
    result = list(items)
    for index in range(1, len(result)):
        current = result[index]
        current_key = key(current)
        position = index - 1
        while position >= 0 and key(result[position]) > current_key:
            result[position + 1] = result[position]
            position -= 1
        result[position + 1] = current
    return result
