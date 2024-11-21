import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GradientDescent:
    def __init__(self, func, grad, initial_x, learning_rate, tolerance, max_iter):
        self.func = func
        self.grad = grad
        self.initial_x = initial_x
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.x_values = []
        self.f_values = []
        self.frames = []

    def optimize(self):
        x = self.initial_x
        self.x_values.append(x)
        self.f_values.append(self.func(x))
        iterations = 0

        while True:
            gradient = self.grad(x)
            if abs(gradient) < self.tolerance or iterations >= self.max_iter:
                break
            x = x - self.learning_rate * gradient
            self.x_values.append(x)
            self.f_values.append(self.func(x))
            iterations += 1
            self.frames.append((self.x_values[:], self.f_values[:]))

        return x

    def plot_animation(self, x_range=(-10, 10), output="gradient_descent.gif"):
        x_vals = np.linspace(x_range[0], x_range[1], 500)
        y_vals = self.func(x_vals)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, y_vals, label="f(x)", color="blue")
        scatter, = ax.plot([], [], 'ro', label="Steps")
        path_line, = ax.plot([], [], 'r--', alpha=0.6)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(0, max(y_vals) * 1.1)
        ax.set_title("Gradient Descent Optimization")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid()

        def init():
            scatter.set_data([], [])
            path_line.set_data([], [])
            return scatter, path_line

        def update(frame):
            x_path, y_path = frame
            scatter.set_data(x_path, y_path)
            path_line.set_data(x_path, y_path)
            return scatter, path_line

        ani = animation.FuncAnimation(fig, update, frames=self.frames, interval=500, blit=True)
        ani.save(output, writer="imagemagick")
        plt.close()
