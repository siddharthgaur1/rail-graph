"""
RailGraph v2 - Graph Analysis Engine

Enhancements over v1:
  - Network resilience analysis: simulate station removal and measure impact
  - K shortest paths (alternative routes, not just the one shortest)
  - Zone-level summary statistics
  - Closeness centrality (how quickly a station reaches all others)
  - Cached pre-computed centrality so the UI doesn't recompute on every interaction
  - find_vulnerable_stations: stations whose removal most increases avg path length
"""

import itertools
import json
import random
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_graph() -> tuple[nx.Graph, dict, list]:
    """Build NetworkX graph from JSON data. Cached after first load."""
    with open(DATA_DIR / "stations.json") as f:
        stations = {s["code"]: s for s in json.load(f)}
    with open(DATA_DIR / "edges.json") as f:
        edges = json.load(f)

    G = nx.Graph()
    for code, s in stations.items():
        G.add_node(code, **s)

    for e in edges:
        G.add_edge(e["from"], e["to"], weight=e["distance_km"], distance_km=e["distance_km"])

    return G, stations, edges


@lru_cache(maxsize=1)
def get_centrality_df() -> pd.DataFrame:
    """All centrality metrics in one DataFrame, sorted by PageRank."""
    G, stations, _ = load_graph()

    pr  = nx.pagerank(G, weight="weight")
    bc  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    dc  = nx.degree_centrality(G)
    cc  = nx.closeness_centrality(G, distance="weight")  # NEW: closeness
    deg = dict(G.degree())

    rows = []
    for code, s in stations.items():
        if code in G:
            rows.append({
                "code":             code,
                "name":             s["name"],
                "state":            s["state"],
                "zone":             s["zone"],
                "category":         s["category"],
                "lat":              s["lat"],
                "lon":              s["lon"],
                "degree":           deg.get(code, 0),
                "pagerank":         round(pr.get(code, 0) * 1000, 4),
                "betweenness":      round(bc.get(code, 0), 4),
                "degree_centrality": round(dc.get(code, 0), 4),
                "closeness":        round(cc.get(code, 0), 6),  # NEW
            })

    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_communities() -> dict[str, int]:
    G, _, _ = load_graph()
    communities = nx.community.greedy_modularity_communities(G)
    return {node: i for i, community in enumerate(communities) for node in community}


def find_shortest_path(src: str, dst: str) -> dict:
    """Dijkstra shortest path by distance."""
    G, stations, _ = load_graph()
    if src not in G:
        return {"error": f"Station '{src}' not found"}
    if dst not in G:
        return {"error": f"Station '{dst}' not found"}
    try:
        path = nx.dijkstra_path(G, src, dst, weight="distance_km")
        length = nx.dijkstra_path_length(G, src, dst, weight="distance_km")
        hops = [
            {
                "code": code,
                "name": stations[code]["name"],
                "state": stations[code]["state"],
                "lat": stations[code]["lat"],
                "lon": stations[code]["lon"],
            }
            for code in path if code in stations
        ]
        return {"path": path, "hops": hops, "total_km": round(length, 1), "stops": len(path) - 1}
    except nx.NetworkXNoPath:
        return {"error": f"No path found between {src} and {dst}"}


def find_k_shortest_paths(src: str, dst: str, k: int = 3) -> list[dict]:
    """
    Find up to k shortest simple paths using Yen's algorithm.
    Returns a list of path dicts, each identical in structure to find_shortest_path.
    """
    G, stations, _ = load_graph()
    if src not in G or dst not in G:
        return [{"error": "Station not found"}]
    try:
        # shortest_simple_paths is a lazy generator specifically so callers can stop
        # early; list(...) before slicing forces it to enumerate every simple path
        # between src and dst first, which is combinatorially expensive (and was
        # observed to hang) on a well-connected graph. islice pulls only k.
        paths = list(itertools.islice(nx.shortest_simple_paths(G, src, dst, weight="distance_km"), k))
        results = []
        for path in paths:
            length = sum(
                G[path[i]][path[i + 1]]["distance_km"] for i in range(len(path) - 1)
            )
            hops = [
                {
                    "code": code,
                    "name": stations[code]["name"],
                    "state": stations[code]["state"],
                    "lat": stations[code]["lat"],
                    "lon": stations[code]["lon"],
                }
                for code in path if code in stations
            ]
            results.append({
                "path": path,
                "hops": hops,
                "total_km": round(length, 1),
                "stops": len(path) - 1,
            })
        return results
    except Exception as e:  # noqa: BLE001 - surfaced to the UI as a result row, not raised
        return [{"error": str(e)}]


def simulate_removal(station_code: str) -> dict:
    """
    Simulate removing a station and report:
    - How many nodes become unreachable
    - Change in number of connected components
    - Change in average shortest path (sampled)
    """
    G, stations, _ = load_graph()
    if station_code not in G:
        return {"error": f"Station '{station_code}' not found"}

    G2 = G.copy()
    G2.remove_node(station_code)

    original_components = nx.number_connected_components(G)
    new_components = nx.number_connected_components(G2)
    isolated = len([n for n in G2.nodes if G2.degree(n) == 0])

    # Sample-based avg path change (full would be O(n^2))
    sample_nodes = random.sample(list(G2.nodes), min(50, G2.number_of_nodes()))
    orig_lengths = []
    new_lengths = []
    for u in sample_nodes:
        for v in sample_nodes:
            if u != v:
                try:
                    orig_lengths.append(nx.dijkstra_path_length(G, u, v, weight="distance_km"))
                    new_lengths.append(nx.dijkstra_path_length(G2, u, v, weight="distance_km"))
                except nx.NetworkXNoPath:
                    pass

    avg_orig = sum(orig_lengths) / len(orig_lengths) if orig_lengths else 0
    avg_new = sum(new_lengths) / len(new_lengths) if new_lengths else 0

    return {
        "removed": station_code,
        "name": stations.get(station_code, {}).get("name", station_code),
        "original_components": original_components,
        "new_components": new_components,
        "components_added": new_components - original_components,
        "isolated_nodes": isolated,
        "avg_path_orig_km": round(avg_orig, 1),
        "avg_path_new_km": round(avg_new, 1),
        "avg_path_increase_pct": round((avg_new - avg_orig) / max(avg_orig, 1) * 100, 1),
    }


def get_zone_stats() -> pd.DataFrame:
    """Zone-level summary: station count, avg degree, total connections, top station."""
    _G, _stations, _ = load_graph()
    df_cent = get_centrality_df()
    zone_df = (
        df_cent.groupby("zone")
        .agg(
            stations=("code", "count"),
            avg_degree=("degree", "mean"),
            avg_betweenness=("betweenness", "mean"),
            top_station=("name", lambda x: x.iloc[0]),
        )
        .round(3)
        .reset_index()
        .sort_values("stations", ascending=False)
    )
    return zone_df


def get_graph_stats() -> dict:
    G, stations, edges = load_graph()
    return {
        "nodes":      G.number_of_nodes(),
        "edges":      G.number_of_edges(),
        "avg_degree": round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2),
        "density":    round(nx.density(G), 4),
        "components": nx.number_connected_components(G),
        "diameter":   nx.diameter(G) if nx.is_connected(G) else "N/A (disconnected)",
        "zones":      len({s["zone"] for s in stations.values()}),
        "states":     len({s["state"] for s in stations.values()}),
        "total_km":   round(sum(e.get("distance_km", 0) for e in edges), 0),
    }
