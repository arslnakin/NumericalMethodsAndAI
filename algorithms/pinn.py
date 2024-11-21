import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PINNPendulum:
    def __init__(self, g, L, theta0, t_data, epochs=1000, learning_rate=0.01):
        self.g = g
        self.L = L
        self.theta0 = theta0
        self.t_data = t_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.loss_history = []
        self.frames = []

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='tanh'),
            tf.keras.layers.Dense(20, activation='tanh'),
            tf.keras.layers.Dense(20, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def loss_fn(self, t):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            theta = self.model(t)
            dtheta_dt = tape1.gradient(theta, t)
        d2theta_dt2 = tape1.gradient(dtheta_dt, t)
        del tape1
        residual = d2theta_dt2 + (self.g / self.L) * tf.sin(theta)
        return tf.reduce_mean(tf.square(residual))

    def train(self):
        t = tf.convert_to_tensor(self.t_data, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = self.loss_fn(t)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.loss_history.append(loss.numpy())

            if epoch % 50 == 0:
                t_test = np.linspace(0, 10, 100).reshape(-1, 1)
                theta_pred = self.model(t_test).numpy()
                self.frames.append((t_test, theta_pred, loss.numpy()))
                print(f"Epoch {epoch}: Loss = {loss.numpy()}")

    def plot_animation(self, output="pinn_pendulum_animation.gif"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 1)
        true_line, = ax.plot([], [], label="True Solution (approx)", color="blue")
        pred_line, = ax.plot([], [], label="PINN Prediction", color="red", linestyle="--")
        loss_text = ax.text(5, 0.8, '', fontsize=12, ha='center')
        ax.set_title("PINN Solution for Pendulum")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Theta (radians)")
        ax.legend()
        ax.grid()

        def init():
            true_line.set_data([], [])
            pred_line.set_data([], [])
            loss_text.set_text('')
            return true_line, pred_line, loss_text

        def update(frame):
            t_test, theta_pred, loss = frame
            true_line.set_data(t_test, np.sin(t_test))
            pred_line.set_data(t_test, theta_pred)
            loss_text.set_text(f"Loss: {loss:.6f}")
            return true_line, pred_line, loss_text

        ani = animation.FuncAnimation(fig, update, frames=self.frames, init_func=init, interval=200, blit=True)
        ani.save(output, writer="imagemagick")
        plt.close()
