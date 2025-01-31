import numpy as np
from scipy.integrate import quad

class GinfActivator:
    def __init__(self, V_min=-2, V_max=2, num_points=200):
        """
        Initializes the GinfActivator and precomputes the lookup table.
        
        Parameters:
        - V_min (float): Minimum V value for the lookup table.
        - V_max (float): Maximum V value for the lookup table.
        - num_points (int): Number of points in the lookup table.
        """

        # Sorry, a mathematician wrote this code.

        self.Rb = 200 * 10**-9  
        self.Rt = 50 * 10**-9  
        self.L = 10 * 10**-6   
        self.D = 1.0 * 10**-9   
        self.epsilon = 80.23 * 8.8541878128 * 10**-12  
        self.psi0 = -0.0102464  
        self.eta = 0.001009347337765476  
        self.kB = 1.38064852 * 10**-23
        self.T = 293.15
        self.eCharge = 1.60217662 * 10**-19
        self.Na = 6.0221409 * 10**23
        self.cb = 0.1 * self.Na
        self.sig = -0.0015 * 10**18 
        self.w = self.eCharge * self.D * self.eta / (self.kB * self.T * self.epsilon * self.psi0)
        self.Du = self.sig / (2 * self.cb * self.Rt)
        self.dg = -2 * self.w * ((self.Rb - self.Rt) / self.Rb) * self.Du

        self.V_values = np.linspace(V_min, V_max, num_points)
        self.ginf_values = np.array([self.dg * self._compute_ginf(V) for V in self.V_values])

    def _R(self, x):
        """ Computes R(x, L) """
        return self.Rb - (self.Rb - self.Rt) * x / self.L

    def _Q(self, V):
        """ Computes Q(V) """
        return np.pi * self.Rt * self.Rb * self.epsilon * self.psi0 / (self.eta * self.L) * V

    def _P(self, V):
        """ Computes P(V) """
        return self._Q(V) * self.L / (self.D * np.pi * self.Rt**2)

    def _g(self, x, V):
        """ Computes g(x, V) """
        R_xL = self._R(x)
        Pe_V = self._P(V)

        term1 = (x * self.Rt) / (self.L * R_xL)
        term2 = (np.exp(Pe_V * ((x * self.Rt**2) / (self.L * self.Rb * R_xL))) - 1) / (np.exp(Pe_V * (self.Rt / self.Rb)) - 1)

        return term1 - term2

    def _compute_ginf(self, V):
        """ Computes the integral of g(x, V) from 0 to L """
        result, _ = quad(self._g, 0, self.L, args=(V,))
        return result
    
    def get_lookup_table(self):
        return self.V_values, self.ginf_values

    def activate(self, V):
        """
        Fast activation function using the lookup table with linear interpolation.
        """
        if np.any(V < self.V_values[0]) or np.any(V > self.V_values[-1]):
            raise ValueError(f"Some input values are out of lookup table range [{self.V_values[0]}, {self.V_values[-1]}]")

        # Perform interpolation for each value in V
        return np.interp(V, self.V_values, self.ginf_values)