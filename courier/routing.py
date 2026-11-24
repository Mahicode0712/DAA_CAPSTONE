"""Shortest paths and travelling-salesperson route heuristics."""

from itertools import permutations
import heapq
import math
from collections.abc import Sequence

from courier.models import RoadGraph


def dijkstra(
    graph: RoadGraph, source: int
) -> tuple[dict[int, float], dict[int, int | None]]:
    """Compute distances and predecessors with a binary min-heap; time is O((V + E) log V), space is O(V)."""
    if source not in graph.adjacency:
        raise ValueError(f"Unknown Dijkstra source node: {source}.")
    distances = {node: math.inf for node in graph.adjacency}
    predecessors: dict[int, int | None] = {node: None for node in graph.adjacency}
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in graph.adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, predecessors


def reconstruct_path(
    source: int, target: int, predecessors: dict[int, int | None]
) -> list[int]:
    """Rebuild a shortest path or reject an unreachable target; time and space are O(V)."""
    if source not in predecessors or target not in predecessors:
        raise ValueError("Path endpoint is not present in the predecessor map.")
    path = [target]
    current = target
    while current != source:
        previous = predecessors[current]
        if previous is None:
            raise ValueError(f"Node {target} is unreachable from node {source}.")
        current = previous
        path.append(current)
    path.reverse()
    return path


def _distance(
    graph: RoadGraph,
    source: int,
    target: int,
    cache: dict[int, tuple[dict[int, float], dict[int, int | None]]],
) -> float:
    """Return a cached shortest-path distance, computing a source tree once; time is O((V + E) log V) on a cache miss."""
    if source not in cache:
        cache[source] = dijkstra(graph, source)
    distance = cache[source][0][target]
    if math.isinf(distance):
        raise ValueError(f"Node {target} is unreachable from node {source}.")
    return distance


def route_length(
    graph: RoadGraph,
    route: Sequence[int],
    cache: dict[int, tuple[dict[int, float], dict[int, int | None]]] | None = None,
) -> float:
    """Sum shortest-path distances between consecutive tour stops; time is O(k) after caching."""
    distance_cache = {} if cache is None else cache
    return sum(
        _distance(graph, route[index], route[index + 1], distance_cache)
        for index in range(len(route) - 1)
    )


def nearest_neighbor_tour(
    graph: RoadGraph, stops: Sequence[int], depot: int = 0
) -> tuple[list[int], float]:
    """Build a depot tour by repeatedly taking the closest unvisited stop; time is O(kV log V + k^2), space is O(kV)."""
    if depot not in graph.adjacency:
        raise ValueError(f"Unknown depot node: {depot}.")
    if any(stop not in graph.adjacency for stop in stops):
        raise ValueError("Every delivery stop must be a graph node.")
    if depot in stops:
        raise ValueError("Delivery stops must not include the depot.")
    if len(set(stops)) != len(stops):
        raise ValueError("Delivery stops must be unique.")
    cache: dict[int, tuple[dict[int, float], dict[int, int | None]]] = {}
    unvisited = set(stops)
    route = [depot]
    current = depot
    while unvisited:
        if current not in cache:
            cache[current] = dijkstra(graph, current)
        distances = cache[current][0]
        reachable = [(distances[stop], stop) for stop in unvisited]
        distance, next_stop = min(reachable)
        if math.isinf(distance):
            raise ValueError(f"No reachable unvisited stop from node {current}.")
        route.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop
    if stops:
        _distance(graph, current, depot, cache)
        route.append(depot)
    return route, route_length(graph, route, cache)


def two_opt(
    graph: RoadGraph, route: Sequence[int]
) -> tuple[list[int], float]:
    """Improve a closed tour using first-improvement 2-opt reversals; each pass is O(k^2) and space is O(kV)."""
    if len(route) < 4:
        result = list(route)
        return result, route_length(graph, result)
    if route[0] != route[-1]:
        raise ValueError("2-opt requires a closed tour that starts and ends at the depot.")
    distance_cache: dict[int, tuple[dict[int, float], dict[int, int | None]]] = {}
    improved = list(route)
    current_length = route_length(graph, improved, distance_cache)
    changed = True
    while changed:
        changed = False
        for first in range(1, len(improved) - 2):
            for last in range(first + 1, len(improved) - 1):
                candidate = improved[:first] + list(reversed(improved[first : last + 1])) + improved[last + 1 :]
                candidate_length = route_length(graph, candidate, distance_cache)
                if candidate_length < current_length - 1e-12:
                    improved = candidate
                    current_length = candidate_length
                    changed = True
                    break
            if changed:
                break
    return improved, current_length


def brute_force_tour(
    graph: RoadGraph, stops: Sequence[int], depot: int = 0
) -> tuple[list[int], float]:
    """Find an optimal tour by checking all stop permutations for at most eight stops; time is O(k! k), space is O(kV)."""
    if len(stops) > 8:
        raise ValueError("Brute-force TSP is limited to eight stops.")
    if len(set(stops)) != len(stops) or depot in stops:
        raise ValueError("Stops must be unique and must not include the depot.")
    if not stops:
        return [depot], 0.0
    cache: dict[int, tuple[dict[int, float], dict[int, int | None]]] = {}
    best_route: list[int] | None = None
    best_length = math.inf
    for ordering in permutations(stops):
        candidate = [depot, *ordering, depot]
        length = route_length(graph, candidate, cache)
        if length < best_length:
            best_route = candidate
            best_length = length
    if best_route is None:
        raise ValueError("No complete route exists for the requested stops.")
    return best_route, best_length


def expand_tour_to_road_path(
    graph: RoadGraph, route: Sequence[int]
) -> tuple[list[int], float]:
    """Expand each tour leg to its actual shortest road path and total km; time is O(k(V + E) log V), space is O(V)."""
    road_path: list[int] = []
    total_distance = 0.0
    cache: dict[int, tuple[dict[int, float], dict[int, int | None]]] = {}
    for index in range(len(route) - 1):
        source, target = route[index], route[index + 1]
        if source not in cache:
            cache[source] = dijkstra(graph, source)
        distances, predecessors = cache[source]
        if math.isinf(distances[target]):
            raise ValueError(f"Node {target} is unreachable from node {source}.")
        segment = reconstruct_path(source, target, predecessors)
        if road_path:
            road_path.extend(segment[1:])
        else:
            road_path.extend(segment)
        total_distance += distances[target]
    if not route:
        return [], 0.0
    return road_path or [route[0]], total_distance
