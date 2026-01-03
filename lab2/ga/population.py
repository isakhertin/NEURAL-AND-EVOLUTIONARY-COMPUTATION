# population.py
# Creation of chromosomes and populations

import random
from utils import normalize_colors

def create_chromosome(num_vertices, max_colors):
    """
    Creates a random chromosome.

    Parameters:
        num_vertices (int): number of vertices in the graph
        max_colors (int): maximum number of colors allowed

    Returns:
        list[int]: chromosome
    """
    return [random.randint(0, max_colors - 1) for _ in range(num_vertices)]


def create_population(pop_size, num_vertices, max_colors):
    """
    Creates an initial population of chromosomes.

    Parameters:
        pop_size (int): number of individuals
        num_vertices (int)
        max_colors (int)

    Returns:
        list[list[int]]: population
    """
    return [
        normalize_colors(create_chromosome(num_vertices, max_colors))
        for _ in range(pop_size)
    ]
