"""Measure algorithm runtimes and save CSV and chart outputs."""

import csv
import random
from pathlib import Path
from statistics import median
from time import perf_counter

from courier.knapsack import knapsack_01
from courier.models import Parcel, RoadGraph
from courier.routing import nearest_neighbor_tour, route_length, two_opt
from courier.sorting import insertion_sort, merge_sort, parcel_priority


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
SIZES = [50, 100, 200, 400, 800, 1600]


def timed_median(operation, repetitions: int = 3) -> tuple[float, object]:
    """Run an operation repeatedly and return median elapsed seconds and its final value; time is repetitions times the operation."""
    measurements = []
    result = None
    for _ in range(repetitions):
        started = perf_counter()
        result = operation()
        measurements.append(perf_counter() - started)
    return median(measurements), result


def benchmark_graph(node_count: int, seed: int) -> RoadGraph:
    """Create a deterministic connected sparse graph for route benchmarks; time and space are O(V + E)."""
    randomizer = random.Random(seed)
    names = {node: f"Point {node}" for node in range(node_count)}
    adjacency: dict[int, list[tuple[int, float]]] = {node: [] for node in names}
    edges: set[tuple[int, int]] = set()
    for node in range(1, node_count):
        parent = randomizer.randrange(node)
        edges.add((parent, node))
    desired = max(node_count - 1, node_count * 2)
    while len(edges) < desired:
        first, second = randomizer.randrange(node_count), randomizer.randrange(node_count)
        if first != second:
            edges.add((min(first, second), max(first, second)))
    for first, second in edges:
        weight = round(randomizer.uniform(1.0, 20.0), 2)
        adjacency[first].append((second, weight))
        adjacency[second].append((first, weight))
    return RoadGraph(names, adjacency)


def create_benchmark_parcels(count: int, node_count: int, seed: int) -> list[Parcel]:
    """Make deterministic positive-weight parcels for algorithm timings; time and space are O(n)."""
    randomizer = random.Random(seed)
    return [
        Parcel(
            f"B{index:05d}",
            randomizer.randrange(1, node_count),
            randomizer.randint(1, 25),
            randomizer.randint(10, 100),
            randomizer.uniform(1.0, 48.0),
        )
        for index in range(count)
    ]


def run_benchmarks() -> list[dict[str, float | int]]:
    """Measure sorting, packing, and routing for each prescribed size; time follows the measured algorithms."""
    results: list[dict[str, float | int]] = []
    for count in SIZES:
        node_count = max(20, count // 4)
        parcels = create_benchmark_parcels(count, node_count, 1000 + count)
        graph = benchmark_graph(node_count, 2000 + count)
        stops = list(range(1, node_count))
        merge_time, _ = timed_median(lambda: merge_sort(parcels, parcel_priority))
        insertion_time, _ = timed_median(lambda: insertion_sort(parcels, parcel_priority))
        knapsack_time, _ = timed_median(lambda: knapsack_01(parcels, 200))

        def route_operation():
            """Run Dijkstra-backed nearest-neighbour planning once for this graph; time is O(kV log V + k^2)."""
            return nearest_neighbor_tour(graph, stops)

        route_time, nearest_result = timed_median(route_operation)
        nearest_route, nearest_length = nearest_result
        two_opt_length: float | None = None
        two_opt_time = 0.0
        if len(stops) <= 60:
            two_opt_time, improved_result = timed_median(
                lambda: two_opt(graph, nearest_route), repetitions=1
            )
            two_opt_length = improved_result[1]
        results.append(
            {
                "parcels": count,
                "graph_nodes": node_count,
                "delivery_stops": len(stops),
                "merge_sort_seconds": merge_time,
                "insertion_sort_seconds": insertion_time,
                "knapsack_seconds": knapsack_time,
                "dijkstra_nearest_seconds": route_time,
                "nearest_neighbor_km": nearest_length,
                "two_opt_seconds": two_opt_time,
                "two_opt_km": two_opt_length if two_opt_length is not None else "",
            }
        )
    return results


def save_results(results: list[dict[str, float | int]]) -> None:
    """Write benchmark rows and three PNG charts; time and space are O(number of measured sizes)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [int(row["parcels"]) for row in results]
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, [float(row["merge_sort_seconds"]) for row in results], marker="o", label="Merge sort")
    plt.plot(sizes, [float(row["insertion_sort_seconds"]) for row in results], marker="s", label="Insertion sort")
    plt.xlabel("Number of parcels")
    plt.ylabel("Median runtime (seconds)")
    plt.title("Sorting runtime")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sorting_runtime.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for field, label in [
        ("merge_sort_seconds", "Merge sort"),
        ("insertion_sort_seconds", "Insertion sort"),
        ("knapsack_seconds", "Knapsack DP"),
        ("dijkstra_nearest_seconds", "Dijkstra + nearest neighbour"),
    ]:
        plt.plot(sizes, [float(row[field]) for row in results], marker="o", label=label)
    plt.xlabel("Number of parcels")
    plt.ylabel("Median runtime (seconds, log scale)")
    plt.yscale("log")
    plt.title("Runtime per algorithm stage")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_runtime_log.png", dpi=150)
    plt.close()

    route_rows = [row for row in results if row["two_opt_km"] != ""]
    plt.figure(figsize=(8, 5))
    route_sizes = [int(row["delivery_stops"]) for row in route_rows]
    plt.plot(route_sizes, [float(row["nearest_neighbor_km"]) for row in route_rows], marker="o", label="Nearest neighbour")
    plt.plot(route_sizes, [float(row["two_opt_km"]) for row in route_rows], marker="s", label="After 2-opt")
    plt.xlabel("Number of delivery stops")
    plt.ylabel("Route length (km)")
    plt.title("Route length: nearest neighbour vs 2-opt (up to 60 stops)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "route_length_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    """Run benchmarks, save artifacts, and print the measurements; complexity follows all benchmark cases."""
    results = run_benchmarks()
    save_results(results)
    print("Benchmark results (median of three runs; seconds):")
    print(
        f"{'n':>5} {'nodes':>6} {'merge':>12} {'insert':>12} "
        f"{'knapsack':>12} {'route':>12} {'NN km':>12} {'2-opt km':>12}"
    )
    for row in results:
        two_opt_text = f"{float(row['two_opt_km']):.2f}" if row["two_opt_km"] != "" else "not run"
        print(
            f"{row['parcels']:>5} {row['graph_nodes']:>6} "
            f"{float(row['merge_sort_seconds']):>12.6f} "
            f"{float(row['insertion_sort_seconds']):>12.6f} "
            f"{float(row['knapsack_seconds']):>12.6f} "
            f"{float(row['dijkstra_nearest_seconds']):>12.6f} "
            f"{float(row['nearest_neighbor_km']):>12.2f} {two_opt_text:>12}"
        )
    print(f"Saved CSV and charts in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
