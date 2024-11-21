from algorithms.monte_carlo import MonteCarloPi

# Example usage for Monte Carlo Pi
monte_carlo = MonteCarloPi(num_points=10000, num_frames=50)
monte_carlo.plot_animation(output="monte_carlo_simulation.gif")
