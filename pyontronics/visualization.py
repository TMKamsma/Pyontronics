try:
    import networkx as nx
except ImportError:
    nx = None

import matplotlib.pyplot as plt
from pyontronics import EchoStateNetwork


def visualize_reservoir(esn: EchoStateNetwork, draw_labels=False):
    """
    Visualizes the ESN as a directed graph with NetworkX.
    """

    if nx is None:
        raise ImportError(
            "networkx is required for visualization"
            "Install it with 'pip install pyontronics[graph]'"
        )

    G = nx.DiGraph()
    input_nodes = [f"inp_{i}" for i in range(esn.input_dim)]
    reservoir_nodes = [f"res_{i}" for i in range(esn.reservoir_size)]
    output_nodes = [f"out_{i}" for i in range(esn.output_dim)]

    # Add all nodes
    G.add_nodes_from(input_nodes)
    G.add_nodes_from(reservoir_nodes)
    G.add_nodes_from(output_nodes)

    # Input -> Reservoir edges
    for i in range(esn.reservoir_size):
        for j in range(esn.input_dim):
            w = esn.W_in[i, j]
            if w != 0:
                G.add_edge(input_nodes[j], reservoir_nodes[i], weight=w)

    # Reservoir -> Reservoir edges
    for i in range(esn.reservoir_size):
        for j in range(esn.reservoir_size):
            w = esn.W_res[i, j]
            if w != 0:
                G.add_edge(reservoir_nodes[j], reservoir_nodes[i], weight=w)

    # Reservoir -> Output edges
    if esn.W_out is not None:
        for i in range(esn.output_dim):
            for j in range(esn.reservoir_size):
                w = esn.W_out[i, j]
                if w != 0:
                    G.add_edge(reservoir_nodes[j], output_nodes[i], weight=w)

    # Layout for drawing
    pos = {}
    for idx, node in enumerate(input_nodes):
        pos[node] = (0, -(idx - (len(input_nodes) - 1) / 2) * 0.1)
    for idx, node in enumerate(output_nodes):
        pos[node] = (2, -(idx - (len(output_nodes) - 1) / 2) * 0.1)

    # Spring layout for reservoir in the middle
    pos_res = nx.spring_layout(G.subgraph(reservoir_nodes), k=0.9, scale=0.5)
    for node, coord in pos_res.items():
        pos[node] = (coord[0] + 1, coord[1])

    plt.figure(figsize=(9, 7))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_color="lightblue",
        node_size=500,
        edgecolors="black",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=reservoir_nodes,
        node_color="lightgreen",
        node_size=500,
        edgecolors="black",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=output_nodes,
        node_color="lightcoral",
        node_size=500,
        edgecolors="black",
    )

    # Separate edges by type for coloring
    in2res, res2res, res2out = [], [], []
    w_in2res, w_res2res, w_res2out = [], [], []

    for src, dst, data in G.edges(data=True):
        w = data["weight"]
        if src in input_nodes and dst in reservoir_nodes:
            in2res.append((src, dst))
            w_in2res.append(w)
        elif src in reservoir_nodes and dst in reservoir_nodes:
            res2res.append((src, dst))
            w_res2res.append(w)
        elif src in reservoir_nodes and dst in output_nodes:
            res2out.append((src, dst))
            w_res2out.append(w)

    def _edge_colors(weights, base_color):
        """
        Returns RGBA colors with alpha scaled by |weight|.
        """
        import matplotlib.colors as mcolors

        if weights:
            max_w = max(abs(w) for w in weights)
        else:
            max_w = 1e-9

        colors = []
        for w in weights:
            alpha = 0.1 + 0.9 * (abs(w) / max_w)
            rgba = list(mcolors.to_rgba(base_color))
            rgba[-1] = alpha
            colors.append(rgba)
        return colors

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=in2res,
        edge_color=_edge_colors(w_in2res, "lightblue"),
        arrowstyle="-|>",
        arrowsize=10,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=res2res,
        edge_color=_edge_colors(w_res2res, "green"),
        arrowstyle="-|>",
        arrowsize=10,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=res2out,
        edge_color=_edge_colors(w_res2out, "red"),
        arrowstyle="-|>",
        arrowsize=10,
    )

    if draw_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("ESN Visualization")
    plt.axis("off")
    plt.show()
