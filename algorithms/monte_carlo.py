import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MonteCarloPi:
    def __init__(self, num_points, num_frames):
        self.num_points = num_points
        self.num_frames = num_frames
        self.frames = self.generate_frames()

    def generate_frames(self):
        x = np.random.uniform(-1, 1, self.num_points)
        y = np.random.uniform(-1, 1, self.num_points)
        distances = x**2 + y**2
        inside_circle = distances <= 1

        points_per_frame = self.num_points // self.num_frames
        frames = []
        for i in range(self.num_frames):
            frame_x = x[:points_per_frame * (i + 1)]
            frame_y = y[:points_per_frame * (i + 1)]
            frame_inside = inside_circle[:points_per_frame * (i + 1)]
            frames.append((frame_x, frame_y, frame_inside))

        return frames

    def plot_animation(self, output="monte_carlo_simulation.gif"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

        def update(frame):
            ax.clear()
            frame_x, frame_y, frame_inside = frame
            ax.scatter(frame_x[frame_inside], frame_y[frame_inside], color='blue', s=1, label="Inside Circle")
            ax.scatter(frame_x[~frame_inside], frame_y[~frame_inside], color='red', s=1, label="Outside Circle")
            ax.add_patch(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--'))
            pi_estimate = 4 * sum(frame_inside) / len(frame_inside)
            ax.set_title(f"Monte Carlo Simulation\nPoints: {len(frame_x)} | Estimated π: {pi_estimate:.4f}")
            ax.legend()

        ani = animation.FuncAnimation(fig, update, frames=self.frames, repeat=False)
        ani.save(output, writer="imagemagick")
        plt.close()
