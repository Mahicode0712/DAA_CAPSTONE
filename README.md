# Mini-Courier Planner

Mini-Courier Planner is a beginner-readable Python capstone for Design & Analysis of Algorithms (ETCCDA302). It prioritizes parcels, selects a maximum-value set that fits one van, and plans a short delivery tour over a sparse road network. A separate benchmark compares algorithm runtimes as the input grows.

## Requirements

- Python 3.10 or later
- `matplotlib` for benchmark charts
- All other code uses the Python standard library

Install matplotlib if needed:

```bash
python3 -m pip install matplotlib
```

## Quick start

From the project folder:

```bash
python3 generate_data.py
python3 -m unittest -v
python3 main.py --dataset small --capacity 60
python3 main.py --dataset large --capacity 120
python3 main.py --dataset small --capacity 60 --all-trips
python3 benchmark.py
```

The CLI displays the sorted parcel table, optimal packing value and load, selected ids, named stop order, expanded road path, route distance, and stage timings. `--all-trips` repeats packing and routing on parcels not yet delivered. If a parcel cannot fit the van capacity, the planner reports an error instead of looping forever.

`generate_data.py` creates the deterministic small (12 nodes, 15 parcels) and large (60 nodes, 150 parcels) datasets using seed 302 (large uses seed 303). You can choose another seed with `python3 generate_data.py --seed 123`.

The benchmark evaluates 50, 100, 200, 400, 800, and 1600 parcels, using a graph with `max(20, n // 4)` nodes and a knapsack capacity of 200 kg. It writes:

- `outputs/benchmark_results.csv`
- `outputs/sorting_runtime.png`
- `outputs/stage_runtime_log.png`
- `outputs/route_length_comparison.png`

The route comparison includes 2-opt only when there are at most 60 stops. Example command outputs are saved in `outputs/` alongside the charts.

## Algorithms

- Stable merge sort orders by earliest deadline and then highest value. Insertion sort is included as a quadratic runtime baseline; merge sort was selected for predictable O(n log n) time on large parcel lists.
- 0/1 knapsack uses the full two-dimensional dynamic-programming table and traceback. A brute-force oracle is included for tests only.
- Binary-heap Dijkstra computes shortest paths on the undirected adjacency-list graph.
- Nearest-neighbour TSP repeatedly selects the closest unvisited destination using cached Dijkstra trees and returns to the depot. 2-opt can improve that tour; exhaustive TSP is restricted to eight stops for comparison.
- The CLI reports the nearest-neighbour approximation ratio against the exact tour when a trip has at most eight stops.

## Data formats

`data/*_parcels.csv` has columns `id,dest,weight,value,deadline`. Parcel weights are positive integer kilograms; scores are in [10, 100].

`data/*_graph.json` uses node objects with integer `id` and human-readable `name`, and edges `[u, v, weight]` in kilometres. The loader builds both directions into an adjacency list, rejects negative weights, and verifies all referenced nodes.

## Folder layout

```text
courier/                 Models, sorting, knapsack, routing, planner
data/                    Generated CSV parcel lists and JSON graphs
docs/REPORT.md           Five-phase capstone report and measured results
outputs/                 Saved CLI/benchmark output, CSV, and charts
tests/test_courier.py    unittest correctness suite
benchmark.py             Runtime and route-quality comparisons
generate_data.py         Reproducible sample-data generator
main.py                  Command-line planner
```

See [docs/REPORT.md](docs/REPORT.md) for algorithm analysis, measured benchmark results, and limitations.
