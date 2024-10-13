import PolynomialRegressionBase
from scipy.optimize import minimize
import numpy as np



class PolynomialRegressionChiSquared(PolynomialRegressionBase):

    def chi_square(self, params):
        """ Calculate Chi-Squared based on model parameters (a3, a2, a1, a0) """
        a3, a2, a1, a0 = params
        y_model = a3 * self.x_values ** 3 + a2 * self.x_values ** 2 + a1 * self.x_values + a0
        return ((np.square(self.y_observed - y_model)) / (self.sigma ** 2)).sum()
        # todo: Check if it uses constant sigma or sigma_i

    def minimize_chi_square(self):
        initial_guess = np.random.rand(4)
        result = minimize(self.chi_square, x0=initial_guess, args=())

        print("Best-fit parameters (Chi-Squared Minimization):", result.x)
        return result.x