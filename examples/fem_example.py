from algorithms.fem import FEMSimulation

# Example usage for FEM
fem = FEMSimulation(x_range=(0, 1), y_range=(0, 1), num_points_x=10, num_points_y=10, scale=0.1)
fem.plot_animation(num_frames=20, output="fem_simulation.gif")
