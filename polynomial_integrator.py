import numpy as np
import scipy.integrate as integrate

class PolynomialIntegrator:

    def __init__(self, coefficients):
        """
        :param coefficients: array of coefficients for a cubic polynomial
        """
        self.coefficients = coefficients

    def cubic_function(self, x):
        """
        Evaluates cubic polynomial for a given x
        :param x: the value at which we want to evaluate cubic polynomial
        :return: The result of ax^3 + bx^2 + cx + d
        """

        a, b, c, d = self.coefficients
        return (a * x**3) + (b * x**2) + (c * x) + d

    def integrate_polynomial(self, lower_bound = -1, upper_bound = 1):
        """
        Integrate the cubic polynimal between lower_bound and upper_bound
        :param lower_bound: the lower bound of integration
        :param upper_bound: the upper bound of integration
        :return: The result of the integral
        """
        result = integrate.quad(self.cubic_function, lower_bound, upper_bound)
        return result