"""Visualize an OpenSearch-backed knowledge graph through the LightRAG API.

The example retrieves nodes and edges from the server's ``/graphs`` endpoint
and renders that API response as standalone HTML. It never reads storage files
or connects directly to OpenSearch.

Prerequisites:
    1. LightRAG Server running with OpenSearch storage:
       lightrag-server --host 0.0.0.0 --port 9621

    2. Documents already indexed (e.g., via the WebUI or API)

Usage:
    # Fetch graph data through the API and generate standalone HTML
    python examples/graph_visual_with_opensearch.py

    # Custom server URL and output file
    python examples/graph_visual_with_opensearch.py --server http://localhost:9621 --output my_graph.html
"""

import argparse
import os
import sys
import webbrowser

import pipmaster as pm

if not pm.is_installed("requests"):
    pm.install("requests")
if not pm.is_installed("pyvis"):
    pm.install("pyvis")

import requests
from pyvis.network import Network


def fetch_graph(server_url: str, label: str = "*", max_nodes: int = 300) -> dict:
    """Fetch knowledge graph data from LightRAG Server API."""
    url = f"{server_url}/graphs"
    params = {"label": label, "max_nodes": max_nodes}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def generate_html(graph_data: dict, output_file: str) -> str:
    """Generate an interactive HTML visualization from graph data."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        print("No nodes found in the graph. Index some documents first.")
        sys.exit(1)

    print(f"Building visualization: {len(nodes)} nodes, {len(edges)} edges")

    net = Network(height="100vh", notebook=False, cdn_resources="in_line")

    # Add nodes with colors based on entity type
    import hashlib

    for node in nodes:
        node_id = node.get("id", "")
        props = node.get("properties", {})
        entity_type = props.get("entity_type", "unknown")
        description = props.get("description", "")

        # Deterministic color from entity type
        color_hash = int(hashlib.md5(entity_type.encode()).hexdigest()[:6], 16)
        color = f"#{color_hash:06x}"

        net.add_node(
            node_id,
            label=node_id,
            title=f"[{entity_type}] {description[:200]}"
            if description
            else entity_type,
            color=color,
        )

    # Add edges
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        props = edge.get("properties", {})
        rel_type = edge.get("type", "")
        description = props.get("description", "")

        net.add_edge(
            source,
            target,
            title=f"[{rel_type}] {description[:200]}" if description else rel_type,
            label=rel_type,
        )

    net.save_graph(output_file)
    print(f"Graph saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an OpenSearch-backed graph through the LightRAG API"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:9621",
        help="LightRAG Server URL (default: http://localhost:9621)",
    )
    parser.add_argument(
        "--output",
        default="knowledge_graph_opensearch.html",
        help="Output HTML file (default: knowledge_graph_opensearch.html)",
    )
    parser.add_argument(
        "--label",
        default="*",
        help="Starting node label, or '*' for all nodes (default: *)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=300,
        help="Maximum nodes to fetch (default: 300)",
    )
    args = parser.parse_args()

    # Verify server is running
    try:
        requests.get(f"{args.server}/health", timeout=5)
    except requests.ConnectionError:
        print(f"Error: Cannot connect to LightRAG Server at {args.server}")
        print("Start the server first: lightrag-server --host 0.0.0.0 --port 9621")
        sys.exit(1)

    graph_data = fetch_graph(args.server, args.label, args.max_nodes)
    output = generate_html(graph_data, args.output)
    webbrowser.open(f"file://{os.path.abspath(output)}")


if __name__ == "__main__":
    main()
