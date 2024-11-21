import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, x_range):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.x_range = x_range

    def objective_function(self, x):
        return x**2 + 4 * np.sin(5 * x) + np.cos(10 * x)

    def initialize_population(self):
        return np.random.uniform(self.x_range[0], self.x_range[1], self.population_size)

    def evaluate_fitness(self, population):
        return self.objective_function(population)

    def select_parents(self, population, fitness):
        adjusted_fitness = 1 / (fitness + 1e-10)
        adjusted_fitness -= adjusted_fitness.min() + 1e-5
        probabilities = adjusted_fitness / adjusted_fitness.sum()
        indices = np.random.choice(len(population), size=2, p=probabilities)
        return population[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand()
            return alpha * parent1 + (1 - alpha) * parent2, alpha * parent2 + (1 - alpha) * parent1
        return parent1, parent2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.uniform(-1, 1)
            individual += mutation
            individual = np.clip(individual, self.x_range[0], self.x_range[1])
        return individual

    def optimize(self):
        population = self.initialize_population()
        frames = []

        for _ in range(self.generations):
            fitness = self.evaluate_fitness(population)
            frames.append((population.copy(), fitness.copy()))
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            population = np.array(new_population)

        return frames

    def plot_animation(self, frames, output="genetic_algorithm.gif"):
        x = np.linspace(self.x_range[0], self.x_range[1], 500)
        y = self.objective_function(x)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, label="Objective Function", color="blue")
        scatter = ax.scatter([], [], color="red", label="Population")
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(min(y) - 1, max(y) + 1)
        ax.set_title("Genetic Algorithm Optimization")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid()

        def update(frame):
            population, _ = frame
            scatter.set_offsets(np.column_stack((population, self.objective_function(population))))
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
        ani.save(output, writer="imagemagick")
        plt.close()
