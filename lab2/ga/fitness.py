# fitness.py
# Fitness and conflict evaluation for the Graph Coloring Problem

from utils import count_colors


def count_conflicts(chromosome, graph):
    """
    Counts the number of conflicts in a chromosome.

    Parameters:
        chromosome (list[int]): color assigned to each vertex
        graph (Graph): graph object

    Returns:
        int: number of conflicting edges
    """
    conflicts = 0
    for u, v in graph.edges:
        if chromosome[u] == chromosome[v]:
            conflicts += 1
    return conflicts


def fitness(chromosome, graph, penalty=200):
    """
    Computes the fitness of a chromosome.

    Lower fitness is better.

    Parameters:
        chromosome (list[int])
        graph (Graph)
        penalty (int): weight for conflicts

    Returns:
        float: fitness value
    """
    conflicts = count_conflicts(chromosome, graph)
    num_colors = count_colors(chromosome)

    return conflicts * penalty + num_colors
