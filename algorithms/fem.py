import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation

class FEMSimulation:
    def __init__(self, x_range, y_range, num_points_x, num_points_y, scale=0.1):
        self.x_range = x_range
        self.y_range = y_range
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.scale = scale
        self.points, self.triang = self.create_mesh()

    def create_mesh(self):
        x = np.linspace(self.x_range[0], self.x_range[1], self.num_points_x)
        y = np.linspace(self.y_range[0], self.y_range[1], self.num_points_y)
        xv, yv = np.meshgrid(x, y)
        points = np.column_stack((xv.ravel(), yv.ravel()))
        triang = tri.Triangulation(points[:, 0], points[:, 1])
        return points, triang

    def apply_deformation(self, frame, num_frames):
        deformed_points = self.points.copy()
        deformed_points[:, 1] += self.scale * frame / num_frames * np.sin(2 * np.pi * self.points[:, 0])
        return deformed_points

    def plot_animation(self, num_frames=20, output="fem_simulation.gif"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(self.points[:, 0].min() - 0.1, self.points[:, 0].max() + 0.1)
        ax.set_ylim(self.points[:, 1].min() - 0.1, self.points[:, 1].max() + 0.1)
        ax.set_aspect('equal')

        def update(frame):
            ax.clear()
            current_points = self.apply_deformation(frame, num_frames)
            ax.triplot(current_points[:, 0], current_points[:, 1], self.triang.triangles, color='blue')
            ax.set_title(f"FEM Deformation - Frame {frame + 1}")
            ax.grid()

        ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
        ani.save(output, writer="imagemagick")
        plt.close()
