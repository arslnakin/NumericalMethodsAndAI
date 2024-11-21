import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RungeKuttaPendulum:
    def __init__(self, theta0, omega0, length, g, dt, t_end):
        self.theta0 = theta0
        self.omega0 = omega0
        self.length = length
        self.g = g
        self.dt = dt
        self.t_end = t_end
        self.thetas, self.omegas = self.solve_motion()

    def solve_motion(self):
        def f(state, t):
            theta, omega = state
            dtheta = omega
            domega = -self.g / self.length * np.sin(theta)
            return np.array([dtheta, domega])

        times = np.arange(0, self.t_end, self.dt)
        thetas, omegas = [], []
        state = np.array([self.theta0, self.omega0])

        for t in times:
            thetas.append(state[0])
            omegas.append(state[1])
            k1 = f(state, t) * self.dt
            k2 = f(state + 0.5 * k1, t + 0.5 * self.dt) * self.dt
            k3 = f(state + 0.5 * k2, t + 0.5 * self.dt) * self.dt
            k4 = f(state + k3, t + self.dt) * self.dt
            state += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return np.array(thetas), np.array(omegas)

    def calculate_energy(self):
        height = -self.length * np.cos(self.thetas)
        potential_energy = self.g * (height + self.length)
        kinetic_energy = 0.5 * (self.length * self.omegas)**2
        total_energy = potential_energy + kinetic_energy
        return potential_energy, kinetic_energy, total_energy

    def plot_energy_animation(self, output="runge_kutta_energy.gif"):
        potential_energy, kinetic_energy, total_energy = self.calculate_energy()
        times = np.arange(0, len(self.thetas) * self.dt, self.dt)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, times[-1])
        ax.set_ylim(0, max(total_energy) * 1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        ax.set_title("Pendulum Energy Over Time")
        pe_line, = ax.plot([], [], label="Potential Energy", color="blue")
        ke_line, = ax.plot([], [], label="Kinetic Energy", color="red")
        te_line, = ax.plot([], [], label="Total Energy", color="green")
        ax.legend()

        def update(frame):
            pe_line.set_data(times[:frame], potential_energy[:frame])
            ke_line.set_data(times[:frame], kinetic_energy[:frame])
            te_line.set_data(times[:frame], total_energy[:frame])
            return pe_line, ke_line, te_line

        ani = animation.FuncAnimation(fig, update, frames=len(times), interval=self.dt * 1000, blit=True)
        ani.save(output, writer="imagemagick")
        plt.close()
