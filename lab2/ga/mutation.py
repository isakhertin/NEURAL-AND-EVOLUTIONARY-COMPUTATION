# mutation.py
# Mutation operators for the Genetic Algorithm

import random
from ga.fitness import count_conflicts


def random_reset_mutation(chromosome, max_colors, mutation_rate=0.1):
    """
    Random reset mutation.

    Parameters:
        chromosome (list[int])
        max_colors (int)
        mutation_rate (float)

    Returns:
        list[int]: mutated chromosome
    """
    new_chromosome = chromosome[:]

    for i in range(len(new_chromosome)):
        if random.random() < mutation_rate:
            new_chromosome[i] = random.randint(0, max_colors - 1)

    return new_chromosome


def conflict_based_mutation(chromosome, graph, max_colors):
    """
    Conflict-based mutation with safe color handling.
    """
    new_chromosome = chromosome[:]

    # Find vertices involved in conflicts
    conflict_vertices = set()
    for u, v in graph.edges:
        if new_chromosome[u] == new_chromosome[v]:
            conflict_vertices.add(u)
            conflict_vertices.add(v)

    if not conflict_vertices:
        return new_chromosome

    vertex = random.choice(list(conflict_vertices))
    current_color = new_chromosome[vertex]

    # Allowed colors
    possible_colors = list(range(max_colors))

    # If current color is invalid, just assign a valid one
    if current_color not in possible_colors:
        new_chromosome[vertex] = random.choice(possible_colors)
        return new_chromosome

    # Otherwise choose a different color
    possible_colors.remove(current_color)
    new_chromosome[vertex] = random.choice(possible_colors)

    return new_chromosome