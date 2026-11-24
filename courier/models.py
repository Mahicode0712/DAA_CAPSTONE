"""Shared parcel and road-network data structures."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Parcel:
    """One parcel and its delivery requirements; storage is O(1)."""

    id: str
    dest: int
    weight: int
    value: int
    deadline: float


class RoadGraph:
    """An undirected weighted graph represented by an adjacency list; storage is O(V + E)."""

    def __init__(self, names: dict[int, str], adjacency: dict[int, list[tuple[int, float]]]) -> None:
        """Validate and retain an adjacency list; time and space are O(V + E)."""
        if not names:
            raise ValueError("The road graph must contain at least one node.")
        if 0 not in names:
            raise ValueError("Node 0 must be the depot.")
        if set(names) != set(adjacency):
            raise ValueError("Graph names and adjacency-list node ids must match.")
        self.names = dict(names)
        self.adjacency = {node: list(neighbors) for node, neighbors in adjacency.items()}
        for node, neighbors in self.adjacency.items():
            for neighbor, weight in neighbors:
                if neighbor not in self.names:
                    raise ValueError(f"Edge from {node} refers to unknown node {neighbor}.")
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("Road weights must be finite and non-negative.")

    @classmethod
    def from_json(cls, path: str | Path) -> RoadGraph:
        """Read a graph JSON file and build both directions of every edge; time and space are O(V + E)."""
        with Path(path).open("r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)
        names: dict[int, str] = {}
        for item in data["nodes"]:
            if isinstance(item, dict):
                node_id = int(item["id"])
                name = str(item["name"])
            else:
                node_id = int(item)
                name = f"Point {node_id}"
            if node_id in names:
                raise ValueError(f"Duplicate graph node id: {node_id}.")
            names[node_id] = name
        adjacency = {node: [] for node in names}
        seen_edges: set[tuple[int, int]] = set()
        for edge in data["edges"]:
            if len(edge) != 3:
                raise ValueError("Every edge must have the form [u, v, weight].")
            source, target, raw_weight = edge
            source, target, weight = int(source), int(target), float(raw_weight)
            if source not in adjacency or target not in adjacency:
                raise ValueError(f"Edge refers to unknown node: {source}, {target}.")
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Road weights must be finite and non-negative.")
            if source == target:
                raise ValueError("Self-loop edges are not supported.")
            edge_key = (min(source, target), max(source, target))
            if edge_key in seen_edges:
                raise ValueError(f"Duplicate undirected edge: {edge_key}.")
            seen_edges.add(edge_key)
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))
        return cls(names, adjacency)


def load_parcels(path: str | Path) -> list[Parcel]:
    """Load and validate parcels from CSV; time and space are O(n)."""
    parcels: list[Parcel] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "dest", "weight", "value", "deadline"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Parcel CSV must have id, dest, weight, value, and deadline columns.")
        for row in reader:
            parcel_id = row["id"].strip()
            destination = int(row["dest"])
            weight = int(row["weight"])
            value = int(row["value"])
            deadline = float(row["deadline"])
            if not parcel_id or parcel_id in seen_ids:
                raise ValueError(f"Parcel ids must be non-empty and unique: {parcel_id!r}.")
            if weight <= 0:
                raise ValueError(f"Parcel {parcel_id} must have a positive integer weight.")
            if not 10 <= value <= 100:
                raise ValueError(f"Parcel {parcel_id} value must be between 10 and 100.")
            if not math.isfinite(deadline) or deadline < 0:
                raise ValueError(f"Parcel {parcel_id} deadline must be finite and non-negative.")
            seen_ids.add(parcel_id)
            parcels.append(Parcel(parcel_id, destination, weight, value, deadline))
    return parcels
