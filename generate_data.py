"""Generate the reproducible sample datasets used by the capstone."""

import argparse
import csv
import json
import random
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def generate_dataset(name: str, node_count: int, parcel_count: int, seed: int) -> None:
    """Create a connected sparse graph and reproducible parcel CSV; time and space are O(V + E + n)."""
    randomizer = random.Random(seed)
    nodes = [{"id": 0, "name": "Depot"}]
    nodes.extend(
        {"id": node, "name": f"Delivery Point {node:02d}"}
        for node in range(1, node_count)
    )
    edges: list[list[int | float]] = []
    existing: set[tuple[int, int]] = set()

    def add_edge(source: int, target: int, distance: float) -> None:
        """Add one undirected edge if it is new; time and space are O(1) average."""
        key = (min(source, target), max(source, target))
        if key not in existing:
            existing.add(key)
            edges.append([source, target, distance])

    for node in range(1, node_count):
        parent = randomizer.randrange(node)
        add_edge(node, parent, round(randomizer.uniform(1.0, 20.0), 2))
    target_edges = max(node_count - 1, int(node_count * 1.5))
    while len(edges) < target_edges:
        first = randomizer.randrange(node_count)
        second = randomizer.randrange(node_count)
        if first != second:
            add_edge(first, second, round(randomizer.uniform(1.0, 20.0), 2))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = DATA_DIR / f"{name}_graph.json"
    with graph_path.open("w", encoding="utf-8") as file:
        json.dump({"nodes": nodes, "edges": edges}, file, indent=2)
        file.write("\n")

    parcel_path = DATA_DIR / f"{name}_parcels.csv"
    with parcel_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "dest", "weight", "value", "deadline"])
        for index in range(1, parcel_count + 1):
            writer.writerow(
                [
                    f"P{index:04d}",
                    randomizer.randrange(1, node_count),
                    randomizer.randint(1, 25),
                    randomizer.randint(10, 100),
                    round(randomizer.uniform(1.0, 48.0), 1),
                ]
            )
    print(f"Generated {name}: {node_count} nodes, {len(edges)} edges, {parcel_count} parcels.")


def main() -> None:
    """Generate the required small and large datasets; time and space are O(V + E + n)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=302)
    args = parser.parse_args()
    generate_dataset("small", 12, 15, args.seed)
    generate_dataset("large", 60, 150, args.seed + 1)


if __name__ == "__main__":
    main()
