"""
RailGraph v2 — Indian Railway Network Explorer

Enhancements over v1:
  - Resilience Simulator: remove a station, see network impact
  - Alternative Routes: k-shortest paths between two stations
  - Zone Comparison table (new tab)
  - Closeness Centrality view (new analysis tab)
  - Total rail km stat in header
  - Route map shows all k-paths with distinct colours
  - Improved sidebar with network density and component count

Run:  streamlit run src/app.py
"""

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html as st_html

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_analysis import (
    load_graph, get_centrality_df, get_communities,
    find_k_shortest_paths, simulate_removal,
    get_graph_stats, get_zone_stats, DATA_DIR,
)

st.set_page_config(
    page_title="RailGraph v2 — Indian Railways",
    page_icon="🚂",
    layout="wide",
)

# ── Load ──────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    G, stations, edges = load_graph()
    df_cent = get_centrality_df()
    communities = get_communities()
    stats = get_graph_stats()
    zone_stats = get_zone_stats()
    return G, stations, edges, df_cent, communities, stats, zone_stats

G, stations, edges, df_cent, communities, stats, zone_stats = load_all()

ZONE_COLORS = {
    "NR": "#EF4444", "NWR": "#F97316", "NCR": "#F59E0B", "NER": "#EAB308",
    "ER": "#84CC16", "ECR": "#22C55E", "ECoR": "#10B981", "SER": "#14B8A6",
    "SR": "#06B6D4", "SCR": "#3B82F6", "SWR": "#6366F1", "WR": "#8B5CF6",
    "WCR": "#A855F7", "CR": "#EC4899", "NFR": "#F43F5E",
}
ROUTE_COLORS = ["#1f2937", "#EF4444", "#3B82F6", "#22C55E", "#F97316"]

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚂 RailGraph v2")
    st.caption("Indian Railway Network Explorer")
    st.divider()
    view = st.radio("View", [
        "📍 Network Map", "📊 Analysis", "🔍 Route Finder",
        "💥 Resilience Simulator", "🗂️ Zone Comparison"
    ])
    st.divider()
    st.markdown(f"""
    **Network stats**
    - {stats['nodes']:,} stations
    - {stats['edges']:,} connections
    - {stats['total_km']:,.0f} km of track (est.)
    - {stats['zones']} railway zones
    - {stats['states']} states covered
    - Avg degree: {stats['avg_degree']}
    - Network density: {stats['density']}
    - Connected components: {stats['components']}
    - Diameter: {stats['diameter']} hops
    """)

# ── KPI row ───────────────────────────────────────────────────────
st.title("🚂 Indian Railway Network Explorer v2")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Stations", f"{stats['nodes']:,}")
k2.metric("Rail Connections", f"{stats['edges']:,}")
k3.metric("Track (est.)", f"{stats['total_km']:,.0f} km")
k4.metric("Most Connected", df_cent.iloc[0]["name"])
k5.metric("Critical Junction",
          df_cent.sort_values("betweenness", ascending=False).iloc[0]["name"])
st.divider()


# ═══════════════════════════════════════════════════════════════════
# VIEW 1: Network Map (unchanged from v1 + closeness option)
# ═══════════════════════════════════════════════════════════════════
if view == "📍 Network Map":
    st.subheader("Interactive Network Map")
    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        color_by = st.selectbox("Colour stations by",
                                ["Railway Zone", "Community", "Degree", "Closeness"])
        show_labels = st.checkbox("Show station names", value=True)
        min_degree = st.slider("Min connections to show", 1, 8, 1)

    m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="CartoDB positron")

    with open(DATA_DIR / "edges.json") as f:
        raw_edges = json.load(f)
    for e in raw_edges:
        s1 = stations.get(e["from"])
        s2 = stations.get(e["to"])
        if s1 and s2:
            folium.PolyLine(
                [(s1["lat"], s1["lon"]), (s2["lat"], s2["lon"])],
                color="#9CA3AF", weight=1.5, opacity=0.6,
                tooltip=f"{s1['name']} → {s2['name']} ({e['distance_km']} km)",
            ).add_to(m)

    closeness_vals = df_cent["closeness"].values
    cl_min, cl_max = closeness_vals.min(), closeness_vals.max()

    for _, row in df_cent.iterrows():
        if row["degree"] < min_degree:
            continue
        code = row["code"]
        s = stations.get(code)
        if not s:
            continue

        if color_by == "Railway Zone":
            color = ZONE_COLORS.get(s["zone"], "#6B7280")
        elif color_by == "Community":
            comm_colors = ["#EF4444","#3B82F6","#22C55E","#F97316","#8B5CF6","#14B8A6","#EC4899"]
            color = comm_colors[communities.get(code, 0) % len(comm_colors)]
        elif color_by == "Closeness":
            # gradient: low = gray, high = dark
            t = (row["closeness"] - cl_min) / max(cl_max - cl_min, 1e-9)
            g = int(31 + t * (255 - 31))
            color = f"#{31:02x}{g:02x}{g:02x}"
        else:
            deg = row["degree"]
            color = "#1f2937" if deg >= 7 else "#374151" if deg >= 5 else "#6B7280" if deg >= 3 else "#9CA3AF"

        radius = 4 + row["degree"] * 1.5
        popup_html = (
            f"<b>{s['name']}</b> ({code})<br>"
            f"Zone: {s['zone']} | State: {s['state']}<br>"
            f"Connections: {row['degree']}<br>"
            f"PageRank: {row['pagerank']:.2f} | Betweenness: {row['betweenness']:.3f}<br>"
            f"Closeness: {row['closeness']:.4f}"
        )
        folium.CircleMarker(
            location=[s["lat"], s["lon"]], radius=radius,
            color="white", weight=1, fill=True, fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=s["name"] if show_labels else "",
        ).add_to(m)

    st_html(m._repr_html_(), height=600)


# ═══════════════════════════════════════════════════════════════════
# VIEW 2: Analysis (+ Closeness tab)
# ═══════════════════════════════════════════════════════════════════
elif view == "📊 Analysis":
    st.subheader("Graph Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 PageRank", "🔗 Betweenness", "📏 Closeness", "🗺️ Communities", "📈 Degree Dist."
    ])

    with tab1:
        top_n = st.slider("Top N", 5, len(df_cent), 20, key="pr_n")
        top = df_cent.head(top_n)
        fig = px.bar(top, x="pagerank", y="name", orientation="h", color="zone",
                     color_discrete_map=ZONE_COLORS,
                     labels={"pagerank": "PageRank Score (×1000)", "name": ""},
                     hover_data=["degree", "state", "zone"])
        fig.update_layout(height=60 + top_n * 22, plot_bgcolor="white", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top[["code","name","state","zone","degree","pagerank","betweenness","closeness"]],
                     use_container_width=True, hide_index=True)

    with tab2:
        top_bc = df_cent.sort_values("betweenness", ascending=False).head(20)
        fig2 = px.bar(top_bc, x="betweenness", y="name", orientation="h", color="zone",
                      color_discrete_map=ZONE_COLORS,
                      labels={"betweenness": "Betweenness Centrality", "name": ""})
        fig2.update_layout(height=500, plot_bgcolor="white", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("High betweenness = removing this station most disrupts the network.")

    with tab3:
        st.markdown("**Closeness Centrality** — how quickly a station can reach all others. "
                    "High closeness = geographically central.")
        top_cc = df_cent.sort_values("closeness", ascending=False).head(20)
        fig_cc = px.bar(top_cc, x="closeness", y="name", orientation="h", color="zone",
                        color_discrete_map=ZONE_COLORS,
                        labels={"closeness": "Closeness Centrality", "name": ""})
        fig_cc.update_layout(height=500, plot_bgcolor="white", margin=dict(t=10, b=10))
        st.plotly_chart(fig_cc, use_container_width=True)

    with tab4:
        comm_df = df_cent.copy()
        comm_df["community"] = comm_df["code"].map(communities)
        comm_df["community_label"] = "Zone " + comm_df["community"].astype(str)
        fig3 = px.scatter_geo(
            comm_df.merge(pd.DataFrame([
                {"code": c, "lat": s["lat"], "lon": s["lon"]} for c, s in stations.items()
            ]), on="code"),
            lat="lat", lon="lon", color="community_label",
            hover_name="name", hover_data={"state": True, "zone": True, "degree": True,
                                            "lat": False, "lon": False},
            size="degree", size_max=20, scope="asia",
            center={"lat": 22, "lon": 80}, projection="natural earth",
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        deg_counts = df_cent["degree"].value_counts().sort_index().reset_index()
        deg_counts.columns = ["Connections", "Stations"]
        fig4 = px.bar(deg_counts, x="Connections", y="Stations",
                      color_discrete_sequence=["#1f2937"])
        fig4.update_layout(plot_bgcolor="white", height=300)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: Route Finder (+ k-shortest paths)
# ═══════════════════════════════════════════════════════════════════
elif view == "🔍 Route Finder":
    st.subheader("Route Finder")
    st.caption("Dijkstra shortest path · alternative routes · zone crossings")

    station_options = {f"{s['name']} ({code})": code for code, s in stations.items()}
    sorted_opts = sorted(station_options.keys())

    col1, col2 = st.columns(2)
    with col1:
        src_label = st.selectbox("From", sorted_opts,
                                 index=sorted_opts.index("New Delhi (NDLS)")
                                 if "New Delhi (NDLS)" in sorted_opts else 0)
    with col2:
        dst_label = st.selectbox("To", sorted_opts,
                                 index=sorted_opts.index("Chennai Central (MAS)")
                                 if "Chennai Central (MAS)" in sorted_opts else 1)

    k_paths = st.slider("Number of alternative routes", 1, 5, 3)

    if st.button("Find Routes", type="primary"):
        src = station_options[src_label]
        dst = station_options[dst_label]
        paths = find_k_shortest_paths(src, dst, k=k_paths)

        if paths and "error" in paths[0]:
            st.error(paths[0]["error"])
        else:
            # Summary table
            summary = [{"Route": f"Route {i+1}", "Stops": p["stops"],
                        "Distance": f"{p['total_km']:,} km"} for i, p in enumerate(paths)]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

            # Route comparison chart
            fig_comp = px.bar(
                pd.DataFrame(summary),
                x="Route", y=[p["total_km"] for p in paths],
                labels={"y": "Distance (km)"},
                color_discrete_sequence=ROUTE_COLORS,
            )
            fig_comp.update_layout(plot_bgcolor="white", height=220, margin=dict(t=10, b=10),
                                   showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Map with all routes
            m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="CartoDB positron")
            for e in edges:
                s1 = stations.get(e["from"])
                s2 = stations.get(e["to"])
                if s1 and s2:
                    folium.PolyLine(
                        [(s1["lat"], s1["lon"]), (s2["lat"], s2["lon"])],
                        color="#E5E7EB", weight=1, opacity=0.4,
                    ).add_to(m)

            for i, p in enumerate(paths):
                color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
                coords = [(stations[c]["lat"], stations[c]["lon"])
                          for c in p["path"] if c in stations]
                folium.PolyLine(coords, color=color, weight=4 - i * 0.5, opacity=0.9,
                                tooltip=f"Route {i+1}: {p['total_km']} km").add_to(m)

            # Mark endpoints
            for code, label_suffix in [(src, "🟢"), (dst, "🔴")]:
                s = stations.get(code)
                if s:
                    folium.Marker(
                        [s["lat"], s["lon"]],
                        popup=s["name"],
                        tooltip=f"{label_suffix} {s['name']}",
                    ).add_to(m)

            st_html(m._repr_html_(), height=500)

            # Detailed route tables
            for i, p in enumerate(paths):
                with st.expander(f"Route {i+1} details — {p['total_km']:,} km, {p['stops']} stops"):
                    hdf = pd.DataFrame(p["hops"])
                    hdf["zone"] = hdf["code"].map(lambda c: stations[c]["zone"] if c in stations else "")
                    st.dataframe(hdf[["code","name","state","zone"]], hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# VIEW 4: Resilience Simulator (NEW)
# ═══════════════════════════════════════════════════════════════════
elif view == "💥 Resilience Simulator":
    st.subheader("Network Resilience Simulator")
    st.caption(
        "Remove a station and see how it affects the network: "
        "isolated regions, component splits, path length increase."
    )
    st.warning("⚠️ Simulation uses sampling for speed. Results are approximate.")

    station_options = {f"{s['name']} ({code})": code for code, s in stations.items()}
    sorted_opts = sorted(station_options.keys())
    target_label = st.selectbox("Station to remove", sorted_opts,
                                index=sorted_opts.index("New Delhi (NDLS)")
                                if "New Delhi (NDLS)" in sorted_opts else 0)

    if st.button("Simulate removal", type="primary"):
        code = station_options[target_label]
        with st.spinner("Simulating…"):
            result = simulate_removal(code)

        if "error" in result:
            st.error(result["error"])
        else:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Connected components before", result["original_components"])
            r2.metric("Connected components after", result["new_components"],
                      delta=result["components_added"],
                      delta_color="inverse")
            r3.metric("Avg path before", f"{result['avg_path_orig_km']} km")
            r4.metric("Avg path after", f"{result['avg_path_new_km']} km",
                      delta=f"+{result['avg_path_increase_pct']}%",
                      delta_color="inverse")

            if result["components_added"] > 0:
                st.error(
                    f"🚨 Removing **{result['name']}** splits the network into "
                    f"**{result['new_components']} disconnected components**. "
                    f"This is a critical vulnerability."
                )
            elif result["avg_path_increase_pct"] > 10:
                st.warning(
                    f"⚠️ Removing **{result['name']}** increases average journey distance by "
                    f"**{result['avg_path_increase_pct']}%**."
                )
            else:
                st.success(
                    f"✅ The network remains well-connected after removing **{result['name']}**. "
                    f"Path length increases by only {result['avg_path_increase_pct']}%."
                )

    st.divider()
    st.subheader("Most Vulnerable Stations (by Betweenness)")
    st.caption("High betweenness stations are the most likely to cause disruption if removed.")
    vuln = df_cent.sort_values("betweenness", ascending=False).head(15)
    fig_v = px.bar(vuln, x="betweenness", y="name", orientation="h",
                   color="zone", color_discrete_map=ZONE_COLORS,
                   labels={"betweenness": "Betweenness Centrality", "name": ""},
                   hover_data=["degree", "state"])
    fig_v.update_layout(height=450, plot_bgcolor="white", margin=dict(t=10, b=10))
    st.plotly_chart(fig_v, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# VIEW 5: Zone Comparison (NEW)
# ═══════════════════════════════════════════════════════════════════
elif view == "🗂️ Zone Comparison":
    st.subheader("Railway Zone Comparison")

    z1, z2 = st.columns(2)
    with z1:
        st.markdown("**Stations per zone**")
        fig_z1 = px.bar(zone_stats, x="zone", y="stations",
                        color="zone", color_discrete_map=ZONE_COLORS,
                        labels={"stations": "Stations", "zone": "Zone"})
        fig_z1.update_layout(showlegend=False, plot_bgcolor="white", height=300)
        st.plotly_chart(fig_z1, use_container_width=True)

    with z2:
        st.markdown("**Average betweenness centrality per zone**")
        fig_z2 = px.bar(zone_stats.sort_values("avg_betweenness", ascending=False),
                        x="zone", y="avg_betweenness",
                        color="zone", color_discrete_map=ZONE_COLORS,
                        labels={"avg_betweenness": "Avg Betweenness", "zone": "Zone"})
        fig_z2.update_layout(showlegend=False, plot_bgcolor="white", height=300)
        st.plotly_chart(fig_z2, use_container_width=True)

    st.subheader("Zone Summary Table")
    st.dataframe(zone_stats, use_container_width=True, hide_index=True)

    # Radar chart
    from math import pi
    metrics = ["stations", "avg_degree", "avg_betweenness"]
    labels = ["Stations", "Avg Degree", "Avg Betweenness"]

    fig_radar = go.Figure()
    for _, row in zone_stats.iterrows():
        vals = [row[m] for m in metrics]
        # normalise to 0-1
        maxvals = zone_stats[metrics].max()
        norm_vals = [v / maxvals[m] if maxvals[m] > 0 else 0 for v, m in zip(vals, metrics)]
        norm_vals += norm_vals[:1]
        angles = [n / len(labels) * 2 * pi for n in range(len(labels))] + [0]
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=labels + [labels[0]],
            fill="toself",
            name=row["zone"],
            line=dict(color=ZONE_COLORS.get(row["zone"], "#6B7280")),
            opacity=0.5,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=500,
    )
    st.subheader("Zone Radar (normalised)")
    st.plotly_chart(fig_radar, use_container_width=True)
