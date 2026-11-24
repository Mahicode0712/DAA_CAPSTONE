"""Focused correctness tests for sorting, packing, routing, and integration."""

import random
import unittest

from courier.knapsack import brute_force_knapsack, knapsack_01
from courier.models import Parcel, RoadGraph
from courier.planner import plan_all_trips, plan_trip
from courier.routing import (
    brute_force_tour,
    dijkstra,
    nearest_neighbor_tour,
    reconstruct_path,
    route_length,
    tour_approximation_ratio,
    two_opt,
    expand_tour_to_road_path,
)
from courier.sorting import merge_sort, parcel_priority


def sample_graph() -> RoadGraph:
    """Build the hand-solvable five-node test graph; time and space are O(V + E)."""
    names = {node: f"Node {node}" for node in range(5)}
    adjacency = {node: [] for node in names}
    edges = [(0, 1, 2), (1, 2, 2), (0, 2, 5), (2, 3, 1), (3, 4, 3), (1, 4, 10)]
    for first, second, weight in edges:
        adjacency[first].append((second, float(weight)))
        adjacency[second].append((first, float(weight)))
    return RoadGraph(names, adjacency)


def test_parcel(index: int, weight: int, value: int, deadline: float, dest: int = 1) -> Parcel:
    """Construct a test parcel with predictable fields; time and space are O(1)."""
    return Parcel(f"P{index}", dest, weight, value, deadline)


class SortingTests(unittest.TestCase):
    """Check stable merge-sort behavior and edge cases."""

    def test_merge_sort_matches_python_sorted_on_random_parcels(self) -> None:
        """Compare against Python's stable sorted result over 200 parcels; time is O(n log n)."""
        randomizer = random.Random(22)
        parcels = [
            test_parcel(index, randomizer.randint(1, 20), randomizer.randint(10, 100), randomizer.randint(1, 10))
            for index in range(200)
        ]
        self.assertEqual(merge_sort(parcels, parcel_priority), sorted(parcels, key=parcel_priority))

    def test_merge_sort_empty_and_single_item(self) -> None:
        """Check empty and singleton inputs; time and space are O(1)."""
        parcel = test_parcel(1, 1, 10, 1)
        self.assertEqual(merge_sort([], parcel_priority), [])
        self.assertEqual(merge_sort([parcel], parcel_priority), [parcel])

    def test_merge_sort_is_stable_for_equal_keys(self) -> None:
        """Ensure equal-priority parcels retain their input order; time is O(n log n)."""
        parcels = [test_parcel(index, 1, 50, 2) for index in range(5)]
        self.assertEqual(merge_sort(parcels, parcel_priority), parcels)


class KnapsackTests(unittest.TestCase):
    """Check dynamic programming results and traceback."""

    def test_hand_checked_knapsack_example(self) -> None:
        """Verify the requested capacity-seven example has value nine; time is O(nC)."""
        parcels = [
            test_parcel(1, 1, 1, 1),
            test_parcel(2, 3, 4, 2),
            test_parcel(3, 4, 5, 3),
            test_parcel(4, 5, 7, 4),
        ]
        result = knapsack_01(parcels, 7)
        self.assertEqual(result.best_value, 9)
        self.assertEqual(result.total_weight, 7)

    def test_dynamic_programming_matches_brute_force(self) -> None:
        """Compare 30 deterministic random instances against exhaustive subsets; time is O(30 * 2^n n)."""
        randomizer = random.Random(302)
        for instance in range(30):
            parcels = [
                test_parcel(
                    index,
                    randomizer.randint(1, 12),
                    randomizer.randint(10, 100),
                    randomizer.randint(1, 10),
                )
                for index in range(randomizer.randint(1, 9))
            ]
            capacity = randomizer.randint(0, 25)
            result = knapsack_01(parcels, capacity)
            self.assertEqual(result.best_value, brute_force_knapsack(parcels, capacity))
            self.assertLessEqual(result.total_weight, capacity)

    def test_zero_capacity(self) -> None:
        """Ensure no positive-weight parcel is selected at zero capacity; time is O(n)."""
        result = knapsack_01([test_parcel(1, 1, 10, 1)], 0)
        self.assertEqual(result.best_value, 0)
        self.assertEqual(result.chosen, [])
        self.assertEqual(result.total_weight, 0)


class RoutingTests(unittest.TestCase):
    """Check shortest paths, tour validity, and route improvements."""

    def test_dijkstra_distances_and_path(self) -> None:
        """Verify solved distances and the shortest path on the five-node graph; time is O((V + E) log V)."""
        graph = sample_graph()
        distances, predecessors = dijkstra(graph, 0)
        self.assertEqual(distances[4], 8.0)
        self.assertEqual(reconstruct_path(0, 4, predecessors), [0, 1, 2, 3, 4])

    def test_unreachable_node_raises_error(self) -> None:
        """Check that an unreachable target is explicit; time is O(V)."""
        graph = RoadGraph({0: "Depot", 1: "A", 2: "B"}, {0: [(1, 1.0)], 1: [(0, 1.0)], 2: []})
        distances, predecessors = dijkstra(graph, 0)
        self.assertEqual(float("inf"), distances[2])
        with self.assertRaises(ValueError):
            reconstruct_path(0, 2, predecessors)
        with self.assertRaises(ValueError):
            nearest_neighbor_tour(graph, [2])

    def test_negative_edge_is_rejected(self) -> None:
        """Ensure invalid negative roads are rejected at graph construction; time and space are O(1)."""
        with self.assertRaises(ValueError):
            RoadGraph({0: "Depot", 1: "A"}, {0: [(1, -1.0)], 1: [(0, -1.0)]})

    def test_nearest_neighbor_and_two_opt_against_optimum(self) -> None:
        """Check tour membership, closed-route length, optimum bound, and 2-opt monotonicity; time includes O(k! k) oracle."""
        graph = sample_graph()
        stops = [1, 2, 3, 4]
        route, length = nearest_neighbor_tour(graph, stops)
        optimum_route, optimum = brute_force_tour(graph, stops)
        improved_route, improved_length = two_opt(graph, route)
        road_path, road_distance = expand_tour_to_road_path(graph, route)
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 0)
        self.assertCountEqual(route[1:-1], stops)
        self.assertEqual(len(route[1:-1]), len(set(route[1:-1])))
        self.assertAlmostEqual(length, route_length(graph, route))
        self.assertEqual(road_path[0], 0)
        self.assertEqual(road_path[-1], 0)
        self.assertAlmostEqual(length, road_distance)
        edge_sum = 0.0
        for first, second in zip(road_path, road_path[1:]):
            edge_sum += next(
                weight
                for neighbor, weight in graph.adjacency[first]
                if neighbor == second
            )
        self.assertAlmostEqual(length, edge_sum)
        self.assertGreaterEqual(length + 1e-9, optimum)
        self.assertLessEqual(improved_length, length + 1e-9)
        self.assertCountEqual(improved_route[1:-1], stops)
        self.assertIsNotNone(optimum_route)
        self.assertGreaterEqual(tour_approximation_ratio(graph, route, stops), 1.0)

    def test_two_opt_matches_or_improves_brute_force(self) -> None:
        """Ensure 2-opt does not exceed the small-instance optimum; time includes O(k! k) oracle."""
        graph = sample_graph()
        route, _ = nearest_neighbor_tour(graph, [1, 2, 3, 4])
        improved, length = two_opt(graph, route)
        _, optimum = brute_force_tour(graph, [1, 2, 3, 4])
        self.assertEqual(improved[0], improved[-1])
        self.assertLessEqual(length, route_length(graph, route) + 1e-9)
        self.assertGreaterEqual(length + 1e-9, optimum)


class PlannerTests(unittest.TestCase):
    """Check capacity constraints and repeated-trip coverage."""

    def test_plan_trip_respects_capacity(self) -> None:
        """Ensure the selected payload fits the configured van; time is O(n log n + nC + route)."""
        parcels = [test_parcel(1, 4, 40, 1), test_parcel(2, 5, 50, 2)]
        plan = plan_trip(parcels, sample_graph(), 5)
        self.assertLessEqual(plan.packing.total_weight, 5)

    def test_plan_all_trips_covers_every_parcel(self) -> None:
        """Ensure repeated plans deliver each input parcel once within capacity; time depends on repeated planning."""
        parcels = [
            test_parcel(1, 4, 40, 1, 1),
            test_parcel(2, 5, 50, 2, 2),
            test_parcel(3, 3, 30, 3, 3),
            test_parcel(4, 2, 20, 4, 4),
        ]
        plans = plan_all_trips(parcels, sample_graph(), 5)
        delivered = [parcel.id for plan in plans for parcel in plan.packing.chosen]
        self.assertCountEqual(delivered, [parcel.id for parcel in parcels])
        for plan in plans:
            self.assertLessEqual(plan.packing.total_weight, 5)


if __name__ == "__main__":
    unittest.main()
