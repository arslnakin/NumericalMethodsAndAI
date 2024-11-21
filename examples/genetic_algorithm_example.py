from algorithms.genetic_algorithm import GeneticAlgorithm

# Example usage for Genetic Algorithm
genetic_algorithm = GeneticAlgorithm(
    population_size=20,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    x_range=(-5, 5)
)
frames = genetic_algorithm.optimize()
genetic_algorithm.plot_animation(frames, output="genetic_algorithm.gif")
