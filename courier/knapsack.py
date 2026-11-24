"""0/1 knapsack dynamic programming and a small brute-force oracle."""

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Sequence

from courier.models import Parcel


@dataclass
class KnapsackResult:
    """Optimal packing result and its full dynamic-programming table; storage includes O(nC) table cells."""

    best_value: int
    total_weight: int
    chosen: list[Parcel]
    table: list[list[int]]


def knapsack_01(parcels: Sequence[Parcel], capacity: int) -> KnapsackResult:
    """Solve 0/1 knapsack and traceback selected parcels; time and space are O(nC)."""
    if capacity < 0:
        raise ValueError("Capacity cannot be negative.")
    if any(parcel.weight <= 0 for parcel in parcels):
        raise ValueError("Parcel weights must be positive.")
    count = len(parcels)
    table = [[0] * (capacity + 1) for _ in range(count + 1)]
    for item_index, parcel in enumerate(parcels, start=1):
        for current_capacity in range(capacity + 1):
            table[item_index][current_capacity] = table[item_index - 1][current_capacity]
            if parcel.weight <= current_capacity:
                included_value = (
                    table[item_index - 1][current_capacity - parcel.weight] + parcel.value
                )
                table[item_index][current_capacity] = max(
                    table[item_index][current_capacity], included_value
                )

    chosen_reversed: list[Parcel] = []
    remaining_capacity = capacity
    for item_index in range(count, 0, -1):
        if table[item_index][remaining_capacity] != table[item_index - 1][remaining_capacity]:
            parcel = parcels[item_index - 1]
            chosen_reversed.append(parcel)
            remaining_capacity -= parcel.weight
    chosen = list(reversed(chosen_reversed))
    total_weight = sum(parcel.weight for parcel in chosen)
    return KnapsackResult(table[count][capacity], total_weight, chosen, table)


def brute_force_knapsack(parcels: Sequence[Parcel], capacity: int) -> int:
    """Return the maximum value by enumerating every subset; time is O(2^n n), space is O(n)."""
    if capacity < 0:
        raise ValueError("Capacity cannot be negative.")
    best_value = 0
    for subset_size in range(len(parcels) + 1):
        for subset in combinations(parcels, subset_size):
            if sum(parcel.weight for parcel in subset) <= capacity:
                best_value = max(best_value, sum(parcel.value for parcel in subset))
    return best_value
