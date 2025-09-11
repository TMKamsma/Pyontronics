import numpy as np
from scipy.integrate import quad


class GinfActivator:
    """
    Conductance as derived in:
    Iontronic neuromorphic signaling with conical microfluidic memristors
    TM Kamsma, WQ Boon, T Ter Rele, C Spitoni, R van Roij
    Physical Review Letters 130 (26), 268401
    """
    
    def __init__(self, V_min=-2, V_max=2, resolution=200, offset=False, g0_shift=True):
        """
        Initializes the GinfActivator and precomputes the lookup table.

        Parameters:
        - V_min (float): Minimum V value for the lookup table.
        - V_max (float): Maximum V value for the lookup table.
        - resolution (int): Number of points in the lookup table.
        - offset (bool): Offset the activator mean to 0
        """

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
        self.w = (
            self.eCharge
            * self.D
            * self.eta
            / (self.kB * self.T * self.epsilon * self.psi0)
        )
        self.Du = self.sig / (2 * self.cb * self.Rt)
        self.dg = -2 * self.w * ((self.Rb - self.Rt) / self.Rb) * self.Du

        self.V_values = np.linspace(V_min, V_max, resolution)
        self.ginf_values = np.array(
            [((self.dg * self._compute_ginf(V)) / self.L) for V in self.V_values]
        )

        if offset:
            self.ginf_values -= np.mean(self.ginf_values)
            
        if not g0_shift:
            self.ginf_values += 1
            
    def _R(self, x):
        """Computes R(x, L)"""
        return self.Rb - (self.Rb - self.Rt) * x / self.L

    def _Q(self, V):
        """Computes Q(V)"""
        return (
            np.pi
            * self.Rt
            * self.Rb
            * self.epsilon
            * self.psi0
            / (self.eta * self.L)
            * V
        )

    def _P(self, V):
        """Computes P(V)"""
        return self._Q(V) * self.L / (self.D * np.pi * self.Rt**2)

    def _g(self, x, V):
        """Computes g(x, V)"""
        R_xL = self._R(x)
        Pe_V = self._P(V)

        term1 = (x * self.Rt) / (self.L * R_xL)
        term2 = (np.exp(Pe_V * ((x * self.Rt**2) / (self.L * self.Rb * R_xL))) - 1) / (
            np.exp(Pe_V * (self.Rt / self.Rb)) - 1
        )

        return term1 - term2

    def _compute_ginf(self, V):
        """Computes the integral of g(x, V) from 0 to L"""
        result, _ = quad(self._g, 0, self.L, args=(V,))
        return result

    def get_lookup_table(self):
        return self.V_values, self.ginf_values

    def activate(self, V):
        """
        Fast activation function using the lookup table with linear interpolation.
        """
        return np.interp(
            np.clip(V, self.V_values[0], self.V_values[-1]),
            self.V_values,
            self.ginf_values,
        )


class NCNM_activator:
    """
    Conductance as derived in:
    Brain-inspired computing with fluidic iontronic nanochannels
    TM Kamsma, J Kim, K Kim, WQ Boon, C Spitoni, J Park, R van Roij
    Proceedings of the National Academy of Sciences 121 (18), e2320242121
    """
    def __init__(self, V_min=-2, V_max=2, resolution=200, offset=False, tanh_transform = True, g0_shift=True):
        """
        Initializes the GinfActivator and precomputes the lookup table.

        Parameters:
        - V_min (float): Minimum V value for the lookup table.
        - V_max (float): Maximum V value for the lookup table.
        - resolution (int): Number of points in the lookup table.
        - offset (bool): Offset the activator mean to 0
        - tanh_transform: ad hoc approach to convert linear NCNM conductance equation to sigmoidal shape
        """

        self.Rb = 100 * 10**-6
        self.Rt = 5 * 10**-6
        self.L = 150 * 10**-6
        self.H = 5 * 10**-6
        self.D = 1.0 * 10**-9
        self.epsilon = 80.23 * 8.8541878128 * 10**-12
        self.psi0 = -0.039
        self.eta = 0.001009347337765476
        self.kB = 1.38064852 * 10**-23
        self.T = 293.15
        self.eCharge = 1.60217662 * 10**-19
        self.Na = 6.0221409 * 10**23
        self.cb = 10 * self.Na
        self.c_eb = 8 * self.Na
        self.c_et = 8.85 * self.Na
        self.sig = -0.0015 * 10**18
        self.lambda_D = 3.1 * 10**-9
        self.Rpore = 2 * self.lambda_D
        self.g0 = (
            2*self.cb
            * self.D
            * self.eta*self.epsilon
            * 2*(self.Rb-self.Rt)*self.H
            / (self.kB * self.T * self.L *np.log(self.Rb/self.Rt))
            * (1+4*self.lambda_D/self.Rpore*(np.cosh(self.eCharge*self.psi0/(2*self.kB*self.T))-1))
        )

        self.V_values = np.linspace(V_min, V_max, resolution)
        self.ginf_values = np.array(
            [((self._compute_ginf(V)) / (2*self.cb*self.L)) for V in self.V_values]
        ) - 1

        if offset:
            self.ginf_values -= np.mean(self.ginf_values)
        
        if not g0_shift:
            self.ginf_values += 1
        
        if tanh_transform:
            self.ginf_values = np.tanh(self.ginf_values)

    def _R(self, x):
        """Computes R(x, L)"""
        return self.Rb - (self.Rb - self.Rt) * x / self.L

    def _g(self, x, V):
        """Computes g(x, V)"""
        R_xL = self._R(x)

        term1 = 2*self.cb
        term2 = -self.c_eb*(self.eCharge*V)/(self.kB*self.T)*(self.Rb-self.Rt)/self.Rb*(np.log(self.Rb/self.Rt)*np.log(self.c_et/self.c_eb*(self.L-x)/self.L+x/self.L)+np.log(R_xL/self.Rt)*np.log(self.c_et/self.c_eb))/((np.log(self.Rb/self.Rt))**2*(self.c_eb/self.c_et-self.Rt/self.Rb))

        return term1 - term2

    def _compute_ginf(self, V):
        """Computes the integral of g(x, V) from 0 to L"""
        result, _ = quad(self._g, 0, self.L, args=(V,))
        return result

    def get_lookup_table(self):
        return self.V_values, self.ginf_values

    def activate(self, V):
        """
        Fast activation function using the lookup table with linear interpolation.
        """
        return np.interp(
            np.clip(V, self.V_values[0], self.V_values[-1]),
            self.V_values,
            self.ginf_values,
        )
