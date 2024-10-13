import numpy as np
from scipy.optimize import minimize


class PolynomialRegressionBase:

    def __init__(self, num_points=15, sigma=0.1):
        self.x_values = np.linspace(start=-1, stop=1, num=num_points)
        self.sigma = sigma
        self.y_true = self.f(self.x_values)
        self.y_observed = self.y_true + self.gaussian_noise(self.sigma)

    def f(self, x):
        " Define polynomial "
        return 2 * (x ** 3) - (x ** 2) + x - 5

    def gaussian_noise(self, mu=0, sigma=1):
        " Gaussian Noise "
        return np.random.normal(mu, sigma, len(self.x_values))


