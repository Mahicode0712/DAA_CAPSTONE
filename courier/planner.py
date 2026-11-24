"""Integration pipeline for parcel selection and delivery routing."""

from dataclasses import dataclass
from time import perf_counter
from collections.abc import Sequence

from courier.knapsack import KnapsackResult, knapsack_01
from courier.models import Parcel, RoadGraph
from courier.routing import expand_tour_to_road_path, nearest_neighbor_tour
from courier.sorting import merge_sort, parcel_priority


@dataclass
class TripPlan:
    """All decisions and elapsed stages for one van trip; storage is O(nC + V + E)."""

    sorted_parcels: list[Parcel]
    packing: KnapsackResult
    stop_order: list[int]
    road_path: list[int]
    total_km: float
    timings: dict[str, float]


def plan_trip(
    parcels: Sequence[Parcel], graph: RoadGraph, capacity: int, depot: int = 0
) -> TripPlan:
    """Run stable prioritization, exact packing, and nearest-neighbour routing; time is O(n log n + nC + kV log V + k^2)."""
    started = perf_counter()
    sorted_parcels = merge_sort(parcels, parcel_priority)
    sorting_time = perf_counter() - started

    started = perf_counter()
    packing = knapsack_01(sorted_parcels, capacity)
    knapsack_time = perf_counter() - started

    started = perf_counter()
    stops = sorted({parcel.dest for parcel in packing.chosen if parcel.dest != depot})
    stop_order, _ = nearest_neighbor_tour(graph, stops, depot)
    road_path, total_km = expand_tour_to_road_path(graph, stop_order)
    routing_time = perf_counter() - started
    return TripPlan(
        sorted_parcels,
        packing,
        stop_order,
        road_path,
        total_km,
        {"sorting": sorting_time, "knapsack": knapsack_time, "routing": routing_time},
    )


def plan_all_trips(
    parcels: Sequence[Parcel], graph: RoadGraph, capacity: int, depot: int = 0
) -> list[TripPlan]:
    """Repeat one-trip planning until every parcel is selected exactly once; worst-case time is O(n^2 C + repeated routing)."""
    if capacity <= 0 and parcels:
        raise ValueError("Capacity must be positive when parcels remain.")
    remaining = list(parcels)
    plans: list[TripPlan] = []
    while remaining:
        plan = plan_trip(remaining, graph, capacity, depot)
        if not plan.packing.chosen:
            raise ValueError("No remaining parcel fits in the van capacity.")
        selected_ids = {parcel.id for parcel in plan.packing.chosen}
        plans.append(plan)
        remaining = [parcel for parcel in remaining if parcel.id not in selected_ids]
    return plans
