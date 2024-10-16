from polynomial_regression_base import PolynomialRegressionBase
import numpy as np
from scipy.optimize import minimize
class PolynomialRegressionMLE(PolynomialRegressionBase):

    def __init__(self, num_points=15, sigma=0.1):
        super().__init__(num_points, sigma)


    def negative_log_likelihood(self, params):
        """Calculate the negative log-likelihood for Gaussian errors."""
        a3, a2, a1, a0 = params
        y_model = a3 * self.x_values ** 3 + a2 * self.x_values ** 2 + a1 * self.x_values + a0
        residuals = self.y_observed - y_model

        # Negative log-likelihood (proportional to Chi-Squared)
        neg_log_likelihood = 0.5 * np.sum((residuals ** 2) / (self.sigma ** 2))
        return neg_log_likelihood

    def minimize_mle(self):
        """Minimize the negative log-likelihood to find the best-fit polynomial coefficients."""
        initial_guess = np.random.rand(4)  # Random initial guess for coefficients
        result = minimize(self.negative_log_likelihood, x0=initial_guess)
        self.best_fit_params_mle = result.x
        print("Best-fit parameters (MLE):", self.best_fit_params_mle)
        return self.best_fit_params_mle
