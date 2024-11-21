from algorithms.pinn import PINNPendulum
import numpy as np

# Example usage for PINN
t_data = np.linspace(0, 10, 100).reshape(-1, 1)
pinn = PINNPendulum(g=9.81, L=1.0, theta0=3.14/4, t_data=t_data, epochs=1000, learning_rate=0.01)
pinn.train()
pinn.plot_animation(output="pinn_pendulum_animation.gif")
