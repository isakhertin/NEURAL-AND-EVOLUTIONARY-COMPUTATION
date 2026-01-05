# utils.py
# Utility functions for loading graphs and helper operations

from collections import defaultdict


class Graph:
    """
    Graph representation using an adjacency list.
    """

    def __init__(self, num_vertices, edges):
        self.num_vertices = num_vertices
        self.edges = edges
        self.adj_list = self._build_adj_list()

    def _build_adj_list(self):
        adj = defaultdict(list)
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj


def load_col_graph(file_path):
    """
    Loads a graph from a .col file.

    Parameters:
        file_path (str): path to the .col file

    Returns:
        Graph object
    """
    num_vertices = 0
    edges = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue

            parts = line.split()

            if parts[0] == 'p':
                # Example: p edge 50 120
                num_vertices = int(parts[2])

            elif parts[0] == 'e':
                # Example: e 1 2  (convert to 0-based)
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                edges.append((u, v))

    if num_vertices == 0:
        raise ValueError("Invalid .col file: number of vertices not found.")

    return Graph(num_vertices, edges)
    
def normalize_colors(chromosome):
    mapping = {}
    next_color = 0
    normalized = []

    for c in chromosome:
        if c not in mapping:
            mapping[c] = next_color
            next_color += 1
        normalized.append(mapping[c])

    return normalized

def count_colors(chromosome):
    """
    Counts how many unique colors are used in a chromosome.
    """
    return len(set(chromosome))
