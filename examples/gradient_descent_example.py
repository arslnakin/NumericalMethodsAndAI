from algorithms.gradient_descent import GradientDescent

# Example usage for Gradient Descent
def func(x):
    return x**2 + 4*x + 4

def grad(x):
    return 2*x + 4

gradient_descent = GradientDescent(
    func=func,
    grad=grad,
    initial_x=10.0,
    learning_rate=0.1,
    tolerance=1e-6,
    max_iter=100
)
gradient_descent.optimize()
gradient_descent.plot_animation(output="gradient_descent.gif")
