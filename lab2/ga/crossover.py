# crossover.py
# Crossover operators for the Genetic Algorithm

import random


def one_point_crossover(parent1, parent2):
    """
    One-point crossover.

    Parameters:
        parent1 (list[int])
        parent2 (list[int])

    Returns:
        tuple(list[int], list[int]): two offspring
    """
    length = len(parent1)
    if length < 2:
        return parent1[:], parent2[:]

    point = random.randint(1, length - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def uniform_crossover(parent1, parent2, swap_prob=0.5):
    """
    Uniform crossover.

    Parameters:
        parent1 (list[int])
        parent2 (list[int])
        swap_prob (float): probability of choosing gene from parent1

    Returns:
        tuple(list[int], list[int]): two offspring
    """
    child1 = []
    child2 = []

    for g1, g2 in zip(parent1, parent2):
        if random.random() < swap_prob:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)

    return child1, child2
