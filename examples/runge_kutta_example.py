from algorithms.runge_kutta import RungeKuttaPendulum

# Example usage for Runge Kutta Pendulum
runge_kutta = RungeKuttaPendulum(theta0=3.14/4, omega0=0, length=1.0, g=9.81, dt=0.02, t_end=10.0)
runge_kutta.plot_energy_animation(output="runge_kutta_energy.gif")
