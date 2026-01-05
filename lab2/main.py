# main.py
# Main execution file for the Genetic Algorithm

import random
import matplotlib.pyplot as plt

from utils import load_col_graph, normalize_colors
from ga.population import create_population
from ga.fitness import fitness, count_conflicts
from ga.selection import tournament_selection, roulette_wheel_selection
from ga.crossover import one_point_crossover, uniform_crossover
from ga.mutation import random_reset_mutation, conflict_based_mutation


def genetic_algorithm(
    graph,
    pop_size=100,
    max_generations=500,
    max_colors=None,
    selection_method="tournament",
    crossover_method="one_point",
    mutation_method="random",
    mutation_rate=0.1,
    elite_size=1
):
    """
    Runs the Genetic Algorithm for Graph Coloring.
    """

    if max_colors is None:
        max_colors = graph.num_vertices

    # Create initial population
    population = create_population(
        pop_size,
        graph.num_vertices,
        max_colors
    )

    current_max_colors = max_colors
    best_fitness_history = []

    for generation in range(max_generations):
        # Evaluate fitness
        fitness_values = [
            fitness(individual, graph)
            for individual in population
        ]

        # Track best individual
        best_index = fitness_values.index(min(fitness_values))
        best_individual = normalize_colors(population[best_index])
        best_fit = fitness_values[best_index]

        conflicts = count_conflicts(best_individual, graph)
        num_colors = len(set(best_individual))

        #current_max_colors = max_colors
        best_fitness_history.append(best_fit)
        if conflicts == 0 and num_colors < current_max_colors:
            current_max_colors = num_colors
        """
        print(
            f"Generation {generation}: "
            f"Reducing max colors to {current_max_colors}"
        )

        print(
            f"Generation {generation}: "
            f"Best fitness = {best_fit}"
        )
        """

        # Elitism
        new_population = [best_individual[:] for _ in range(elite_size)]


        # Generate new population
        while len(new_population) < pop_size:
            # Selection
            if selection_method == "tournament":
                parent1 = tournament_selection(
                    population, fitness_values
                )
                parent2 = tournament_selection(
                    population, fitness_values
                )
            else:
                parent1 = roulette_wheel_selection(
                    population, fitness_values
                )
                parent2 = roulette_wheel_selection(
                    population, fitness_values
                )


            # Crossover
            if crossover_method == "one_point":
                child1, child2 = one_point_crossover(
                    parent1, parent2
                )
            else:
                child1, child2 = uniform_crossover(
                    parent1, parent2
                )
            child1 = normalize_colors(child1)
            child2 = normalize_colors(child2)
            # Mutation
            if mutation_method == "random":
                child1 = random_reset_mutation(child1, current_max_colors, mutation_rate)
                child2 = random_reset_mutation(child2, current_max_colors, mutation_rate)
            else:
                child1 = conflict_based_mutation(child1, graph, current_max_colors)
                child2 = conflict_based_mutation(child2, graph, current_max_colors)


            child1 = normalize_colors(child1)
            child2 = normalize_colors(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)


        population = new_population

    return best_individual, best_fitness_history


def plot_fitness(fitness_history, label=None):
    """
    Plots fitness history 
    """
    plt.plot(fitness_history, label=label)

def run_experiments(graph_path, max_generations, label_size):
    graph = load_col_graph(graph_path)

    experiments = [
        # 1
        dict(
            pop_size=100,
            max_generations=max_generations,
            selection_method="tournament",
            crossover_method="one_point",
            mutation_method="random",
            mutation_rate=0.10,
            elite_size=1
        ),
        # 2
        dict(
            pop_size=150,
            max_generations=max_generations,
            selection_method="tournament",
            crossover_method="uniform",
            mutation_method="random",
            mutation_rate=0.15,
            elite_size=2
        ),
        # 3
        dict(
            pop_size=150,
            max_generations=max_generations,
            selection_method="roulette",
            crossover_method="one_point",
            mutation_method="random",
            mutation_rate=0.10,
            elite_size=2
        ),
        # 4
        dict(
            pop_size=200,
            max_generations=max_generations,
            selection_method="roulette",
            crossover_method="uniform",
            mutation_method="random",
            mutation_rate=0.20,
            elite_size=1
        ),
        # 5
        dict(
            pop_size=150,
            max_generations=max_generations,
            selection_method="tournament",
            crossover_method="uniform",
            mutation_method="conflict",
            mutation_rate=0.15,
            elite_size=2
        ),
        # 6
        dict(
            pop_size=200,
            max_generations=max_generations,
            selection_method="tournament",
            crossover_method="one_point",
            mutation_method="conflict",
            mutation_rate=0.10,
            elite_size=2
        ),
    ]

    plt.figure()

    for i, params in enumerate(experiments, start=1):
        random.seed(42 + i)
        print(f"\n=== Experiment {i} ===")
        print(params)

        best_solution, fitness_history = genetic_algorithm(
            graph,
            pop_size=params["pop_size"],
            max_generations=params["max_generations"],
            max_colors=20,
            selection_method=params["selection_method"],
            crossover_method=params["crossover_method"],
            mutation_method=params["mutation_method"],
            mutation_rate=params["mutation_rate"],
            elite_size=params["elite_size"]
        )

        conflicts = count_conflicts(best_solution, graph)
        num_colors = len(set(best_solution))

        print(f"Result:")
        print(f"  Conflicts: {conflicts}")
        print(f"  Colors: {num_colors}")
        print(f"  Best fitness: {min(fitness_history)}")

        #if i == 5:
        #    plot_specific_fitness(fitness_history)

        plot_fitness(
            fitness_history,
            label=f"Exp {i}"
        )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(f'Fitness evolution – all experiments ({label_size})')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_specific_fitness(fitness_history):
    """
    Plots the evolution of fitness.
    """
    plt.figure()
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Evolution")
    plt.grid(True)
    plt.show()

def plot_evolution(graph_path):
    graph = load_col_graph(graph_path)
    #random.seed(42 )

    best_solution, fitness_history = genetic_algorithm(
        graph,
        max_colors=20,
    
        pop_size=150,
        max_generations=140,
        selection_method="roulette",
        crossover_method="one_point",
        mutation_method="random",
        mutation_rate=0.10,
        elite_size=2
    )

    conflicts = count_conflicts(best_solution, graph)
    num_colors = len(set(best_solution))
    print("Conflicts:", conflicts)
    print("Colors:", num_colors)
    print("Best fitness:", min(fitness_history))

    plt.figure()
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Evolution (MEDIUM) - Best configuration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_large_best.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    # ===== CHANGE THIS PATH =====
    graph_path = "data/test_data.col"
    graph_path = "data/SMALL_myciel3.col"
    graph_path = "data/LARGE_david.col"
    graph_path = "data/MEDIUM_myciel4.col"


    #run_experiments(graph_path, max_generations=800, label_size= 'LARGE')
    plot_evolution(graph_path)

    """
    graph = load_col_graph(graph_path)

    best_solution, fitness_history = genetic_algorithm(
        graph,
        pop_size=150,
        max_generations=10,
        max_colors=20,
        selection_method="tournament",
        crossover_method="uniform",
        mutation_method="conflict",
        mutation_rate=0.15,
        elite_size=2
    )

    plot_fitness(fitness_history)
    """