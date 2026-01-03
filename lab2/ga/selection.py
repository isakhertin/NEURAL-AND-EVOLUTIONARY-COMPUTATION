# selection.py
# Selection operators for the Genetic Algorithm

import random


def tournament_selection(population, fitness_values, tournament_size=3):
    """
    Tournament selection.

    Parameters:
        population (list[list[int]])
        fitness_values (list[float])
        tournament_size (int)

    Returns:
        list[int]: selected chromosome
    """
    selected_indices = random.sample(range(len(population)), tournament_size)

    best_index = selected_indices[0]
    best_fitness = fitness_values[best_index]

    for idx in selected_indices[1:]:
        if fitness_values[idx] < best_fitness:
            best_fitness = fitness_values[idx]
            best_index = idx

    return population[best_index]


def roulette_wheel_selection(population, fitness_values):
    """
    Roulette wheel selection (fitness proportional).

    Lower fitness => higher probability.

    Parameters:
        population (list[list[int]])
        fitness_values (list[float])

    Returns:
        list[int]: selected chromosome
    """
    # Convert fitness to selection probability (inverse fitness)
    epsilon = 1e-6  # avoid division by zero
    inverse_fitness = [1.0 / (f + epsilon) for f in fitness_values]

    total = sum(inverse_fitness)
    probabilities = [f / total for f in inverse_fitness]

    selected_index = random.choices(
        range(len(population)),
        weights=probabilities,
        k=1
    )[0]

    return population[selected_index]
