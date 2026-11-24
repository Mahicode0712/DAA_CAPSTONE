"""Command-line entry point for the Mini-Courier Planner."""

import argparse
from pathlib import Path

from courier.models import RoadGraph, load_parcels
from courier.planner import TripPlan, plan_all_trips, plan_trip
from courier.routing import tour_approximation_ratio


ROOT = Path(__file__).parent


def format_trip(plan: TripPlan, graph: RoadGraph, trip_number: int = 1) -> str:
    """Format one trip's results and exact small-tour comparison; time can include O(k! k), space is O(n + path length)."""
    lines = [f"TRIP {trip_number}", "=" * 72, "Sorted parcels (deadline asc, value desc):"]
    lines.append(f"{'ID':<9} {'Destination':<24} {'kg':>4} {'Value':>6} {'Deadline h':>12}")
    for parcel in plan.sorted_parcels:
        lines.append(
            f"{parcel.id:<9} {graph.names.get(parcel.dest, str(parcel.dest)):<24} "
            f"{parcel.weight:>4} {parcel.value:>6} {parcel.deadline:>12.1f}"
        )
    selected_ids = ", ".join(parcel.id for parcel in plan.packing.chosen) or "(none)"
    lines.extend(
        [
            "",
            "Packing plan (0/1 dynamic programming):",
            f"  Best value: {plan.packing.best_value}",
            f"  Load: {plan.packing.total_weight} kg",
            f"  Chosen parcel ids: {selected_ids}",
            "",
            "Delivery stop order:",
            "  " + " -> ".join(graph.names[node] for node in plan.stop_order),
            "Full road path:",
            "  " + " -> ".join(graph.names[node] for node in plan.road_path),
            f"Total distance: {plan.total_km:.2f} km",
        ]
    )
    stops = plan.stop_order[1:-1]
    if len(stops) <= 8:
        ratio = tour_approximation_ratio(graph, plan.stop_order, stops)
        lines.append(f"Nearest-neighbour approximation ratio: {ratio:.3f}")
    lines.append("Stage timings:")
    for stage, seconds in plan.timings.items():
        lines.append(f"  {stage.title():<10} {seconds:.6f} seconds")
    return "\n".join(lines)


def main() -> None:
    """Parse CLI options and run one or all trips; complexity follows the selected planner operation."""
    parser = argparse.ArgumentParser(description="Plan courier parcels and road routes.")
    parser.add_argument("--dataset", choices=("small", "large"), default="small")
    parser.add_argument("--capacity", type=int, default=60, help="Van capacity in kg (default: 60)")
    parser.add_argument("--all-trips", action="store_true", help="Repeat trips until all parcels are delivered.")
    args = parser.parse_args()
    if args.capacity < 0:
        parser.error("--capacity must be zero or greater.")

    data_dir = ROOT / "data"
    parcels = load_parcels(data_dir / f"{args.dataset}_parcels.csv")
    graph = RoadGraph.from_json(data_dir / f"{args.dataset}_graph.json")
    if any(parcel.dest not in graph.names for parcel in parcels):
        raise ValueError("A parcel destination is missing from the selected road graph.")
    if args.all_trips:
        plans = plan_all_trips(parcels, graph, args.capacity)
    else:
        plans = [plan_trip(parcels, graph, args.capacity)]
    print(f"Mini-Courier Planner | dataset={args.dataset} | capacity={args.capacity} kg")
    print(f"Parcels: {len(parcels)} | graph nodes: {len(graph.names)}")
    for number, plan in enumerate(plans, start=1):
        print()
        print(format_trip(plan, graph, number))
    if args.all_trips:
        delivered = sum(len(plan.packing.chosen) for plan in plans)
        print(f"\nAll trips complete: {len(plans)} trip(s), {delivered} parcels delivered.")


if __name__ == "__main__":
    main()
