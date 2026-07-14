"""Generate a synthetic Indian Railway network dataset (stations.json, edges.json).

The repo shipped with no data files at all — data/ contained only a .gitkeep,
so the app crashed on load_graph() despite the README claiming data was
bundled. This generates a plausible, connected network at the scale the
README describes (~600 stations, ~800 edges, 15 zones, ~29 states):
real names/codes/approximate coordinates for ~40 well-known major junctions,
procedurally generated stations and connectivity for the rest. This is NOT
scraped official Indian Railways topology — see README's "Data" section.

Usage:
    python scripts/generate_data.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
rng = random.Random(42)

# Real major junctions: (code, name, state, zone, approx lat, approx lon).
# Public, well-known station identities — used to seed recognizable hubs;
# the connectivity between them is procedurally generated, not real topology.
MAJOR_JUNCTIONS = [
    ("NDLS", "New Delhi", "Delhi", "NR", 28.64, 77.22),
    ("CSMT", "Mumbai CSMT", "Maharashtra", "CR", 18.94, 72.84),
    ("MAS", "Chennai Central", "Tamil Nadu", "SR", 13.08, 80.28),
    ("HWH", "Howrah", "West Bengal", "ER", 22.58, 88.34),
    ("SBC", "Bengaluru City", "Karnataka", "SWR", 12.98, 77.57),
    ("SC", "Secunderabad", "Telangana", "SCR", 17.43, 78.50),
    ("NGP", "Nagpur", "Maharashtra", "CR", 21.15, 79.09),
    ("BZA", "Vijayawada", "Andhra Pradesh", "SCR", 16.52, 80.62),
    ("GHY", "Guwahati", "Assam", "NFR", 26.19, 91.74),
    ("JP", "Jaipur", "Rajasthan", "NWR", 26.92, 75.79),
    ("ADI", "Ahmedabad", "Gujarat", "WR", 23.03, 72.60),
    ("PUNE", "Pune Junction", "Maharashtra", "CR", 18.53, 73.87),
    ("BPL", "Bhopal Junction", "Madhya Pradesh", "WCR", 23.27, 77.41),
    ("PNBE", "Patna Junction", "Bihar", "ECR", 25.61, 85.14),
    ("LKO", "Lucknow", "Uttar Pradesh", "NER", 26.84, 80.94),
    ("CNB", "Kanpur Central", "Uttar Pradesh", "NCR", 26.45, 80.35),
    ("PRYJ", "Prayagraj Junction", "Uttar Pradesh", "NCR", 25.45, 81.85),
    ("KGP", "Kharagpur", "West Bengal", "SER", 22.34, 87.32),
    ("BBS", "Bhubaneswar", "Odisha", "ECoR", 20.27, 85.84),
    ("CBE", "Coimbatore Junction", "Tamil Nadu", "SR", 11.00, 76.96),
    ("TVC", "Thiruvananthapuram Central", "Kerala", "SR", 8.49, 76.95),
    ("ERS", "Ernakulam Junction", "Kerala", "SR", 9.97, 76.29),
    ("MDU", "Madurai Junction", "Tamil Nadu", "SR", 9.92, 78.12),
    ("VSKP", "Visakhapatnam", "Andhra Pradesh", "ECoR", 17.72, 83.30),
    ("RNC", "Ranchi", "Jharkhand", "SER", 23.37, 85.32),
    ("DBG", "Darbhanga Junction", "Bihar", "ECR", 26.16, 85.90),
    ("GKP", "Gorakhpur Junction", "Uttar Pradesh", "NER", 26.76, 83.37),
    ("ASR", "Amritsar Junction", "Punjab", "NR", 31.63, 74.87),
    ("UMB", "Ambala Cantt", "Haryana", "NR", 30.35, 76.82),
    ("JU", "Jodhpur Junction", "Rajasthan", "NWR", 26.29, 73.02),
    ("BCT", "Mumbai Central", "Maharashtra", "WR", 18.97, 72.82),
    ("ST", "Surat", "Gujarat", "WR", 21.20, 72.83),
    ("RTM", "Ratlam Junction", "Madhya Pradesh", "WCR", 23.33, 75.04),
    ("JBP", "Jabalpur", "Madhya Pradesh", "WCR", 23.16, 79.95),
    ("R", "Raipur Junction", "Chhattisgarh", "SER", 21.25, 81.63),
    ("BSP", "Bilaspur Junction", "Chhattisgarh", "SER", 22.09, 82.15),
    ("MYS", "Mysuru Junction", "Karnataka", "SWR", 12.31, 76.65),
    ("UBL", "Hubballi Junction", "Karnataka", "SWR", 15.36, 75.13),
    ("DBRG", "Dibrugarh", "Assam", "NFR", 27.48, 94.91),
    ("NJP", "New Jalpaiguri", "West Bengal", "NFR", 26.70, 88.44),
]

ZONE_STATES = {
    "NR": ["Delhi", "Haryana", "Punjab", "Himachal Pradesh", "Jammu and Kashmir"],
    "NWR": ["Rajasthan"],
    "NCR": ["Uttar Pradesh"],
    "NER": ["Uttar Pradesh", "Bihar"],
    "NFR": ["Assam", "West Bengal", "Meghalaya", "Nagaland", "Tripura"],
    "ER": ["West Bengal", "Bihar", "Jharkhand"],
    "ECR": ["Bihar", "Jharkhand"],
    "ECoR": ["Odisha", "Andhra Pradesh"],
    "SER": ["West Bengal", "Odisha", "Jharkhand", "Chhattisgarh"],
    "SR": ["Tamil Nadu", "Kerala", "Puducherry"],
    "SCR": ["Telangana", "Andhra Pradesh"],
    "SWR": ["Karnataka"],
    "WR": ["Gujarat", "Maharashtra"],
    "WCR": ["Madhya Pradesh"],
    "CR": ["Maharashtra"],
}
ZONE_CENTROIDS = {  # approximate (lat, lon) to jitter synthetic stations around
    "NR": (29.5, 76.5), "NWR": (26.5, 73.5), "NCR": (26.0, 80.5), "NER": (27.0, 82.0),
    "NFR": (26.5, 92.5), "ER": (23.0, 87.5), "ECR": (25.5, 85.5), "ECoR": (19.5, 84.5),
    "SER": (22.0, 84.0), "SR": (10.5, 78.0), "SCR": (16.5, 79.0), "SWR": (14.0, 76.0),
    "WR": (21.5, 72.5), "WCR": (23.0, 78.5), "CR": (19.5, 75.5),
}
NAME_PREFIXES = ["Uttar", "Purvi", "Naya", "Shri", "Sant", "Rani", "Raja", "New", "Old", "Madhya"]
NAME_ROOTS = ["pur", "nagar", "ganj", "abad", "puram", "kot", "garh", "wara", "shahar", "kheda"]
CATEGORIES = ["A1", "A", "A", "B", "B", "B", "C", "C", "C", "C"]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def gen_synthetic_station(idx: int, zone: str) -> dict:
    lat0, lon0 = ZONE_CENTROIDS[zone]
    name = f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_ROOTS).capitalize()}{rng.choice(NAME_ROOTS)}"
    code = f"{zone[:2].upper()}{idx:03d}"
    return {
        "code": code,
        "name": name,
        "state": rng.choice(ZONE_STATES[zone]),
        "zone": zone,
        "category": rng.choice(CATEGORIES),
        "lat": round(lat0 + rng.uniform(-3.5, 3.5), 4),
        "lon": round(lon0 + rng.uniform(-3.5, 3.5), 4),
    }


def build_stations(total: int = 600) -> list[dict]:
    stations = [
        {"code": c, "name": n, "state": st, "zone": z, "category": "A1", "lat": lat, "lon": lon}
        for c, n, st, z, lat, lon in MAJOR_JUNCTIONS
    ]
    seen_codes = {s["code"] for s in stations}
    zones = list(ZONE_CENTROIDS.keys())
    idx = 1
    while len(stations) < total:
        zone = zones[len(stations) % len(zones)]
        s = gen_synthetic_station(idx, zone)
        idx += 1
        if s["code"] in seen_codes:
            continue
        seen_codes.add(s["code"])
        stations.append(s)
    return stations


def build_edges(stations: list[dict]) -> list[dict]:
    """Connect each zone internally (nearest-neighbor spanning tree + extra
    edges for realism), then bridge zones together via their major-junction
    hub stations so the whole network is a single connected component."""
    by_zone: dict[str, list[dict]] = {}
    for s in stations:
        by_zone.setdefault(s["zone"], []).append(s)

    edges: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    def add_edge(a: dict, b: dict):
        key = tuple(sorted((a["code"], b["code"])))
        if key in seen_pairs or a["code"] == b["code"]:
            return
        seen_pairs.add(key)
        edges.append({
            "from": a["code"], "to": b["code"],
            "distance_km": round(max(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), 5.0), 1),
        })

    # Nearest-neighbor spanning tree within each zone: guarantees connectivity.
    for zone, zone_stations in by_zone.items():
        connected = [zone_stations[0]]
        remaining = zone_stations[1:]
        while remaining:
            best_pair, best_dist = None, float("inf")
            for c in connected:
                for r in remaining:
                    d = haversine_km(c["lat"], c["lon"], r["lat"], r["lon"])
                    if d < best_dist:
                        best_dist, best_pair = d, (c, r)
            add_edge(*best_pair)
            connected.append(best_pair[1])
            remaining.remove(best_pair[1])

        # A few extra intra-zone edges (nearest-neighbor, not just the tree) for realism.
        for s in zone_stations:
            neighbors = sorted(
                (o for o in zone_stations if o["code"] != s["code"]),
                key=lambda o: haversine_km(s["lat"], s["lon"], o["lat"], o["lon"]),
            )[:2]
            for n in neighbors:
                if rng.random() < 0.5:
                    add_edge(s, n)

    # Bridge zones together via their major-junction hubs (nearest hub-to-hub pairs).
    hubs = [s for s in stations if s["code"] in {m[0] for m in MAJOR_JUNCTIONS}]
    for hub in hubs:
        others = sorted(
            (h for h in hubs if h["zone"] != hub["zone"]),
            key=lambda o: haversine_km(hub["lat"], hub["lon"], o["lat"], o["lon"]),
        )[:2]
        for o in others:
            add_edge(hub, o)

    return edges


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stations = build_stations(600)
    edges = build_edges(stations)

    (DATA_DIR / "stations.json").write_text(json.dumps(stations, indent=2), encoding="utf-8")
    (DATA_DIR / "edges.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")

    print(f"{len(stations)} stations, {len(edges)} edges across {len(ZONE_CENTROIDS)} zones")
    print(f"Written to {DATA_DIR}")


if __name__ == "__main__":
    main()
