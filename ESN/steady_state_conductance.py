import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Constants (fill in the values)
Rb = 200*10**-9  # Replace with your value
Rt = 50*10**-9  # Replace with your value
L = 10*10**-6   # Replace with your value
D = 1.0*10**-9   # Replace with your value
epsilon = 80.23*8.8541878128*10**-12  # Replace with your value
psi0 = -0.0102464  # Replace with your value
eta = 0.001009347337765476  # Replace with your value
kB = 1.38064852*10**-23
T = 293.15
eCharge = 1.60217662*10**-19
Na = 6.0221409*10**23
cb = 0.1*Na
sig = -0.0015*10**18 
w = eCharge*D*eta/(kB*T*epsilon*psi0)
Du = sig/(2*cb*Rt)
dg = -2*w*((Rb-Rt)/Rb)*Du

# Function definitions
def R(x, L):
    """
    Function R[x, L] = Rb - (Rb - Rt) * x / L
    """
    return Rb - (Rb - Rt) * x / L

def Q(V):
    """
    Function Q[V] = -pi * Rt * Rb * epsilon * psi0 / (eta * L)
    """
    return np.pi * Rt * Rb * epsilon * psi0 / (eta * L)*V

def P(V):
    """
    Function P[V] = Q[V] * L / (D * pi * Rt**2)
    """
    return Q(V) * L / (D * np.pi * Rt**2)

def g(x, V):
    """
    Your function g(x, V).
    """
    R_xL = R(x, L)  # Compute R[x, L]
    Pe_V = P(V)     # Compute P[V]
    
    term1 = (x * Rt) / (L * R_xL)
    term2 = (np.exp(Pe_V * ((x * Rt**2) / (L * Rb * R_xL))) - 1) / (np.exp(Pe_V * (Rt / Rb)) - 1)
    
    return term1 - term2

# Define the integral function ginf(V)
def ginf(V, L):
    """
    Computes ginf(V) = integral of g(x, V) from 0 to L.
    
    Parameters:
    V (float): Parameter V in g(x, V).
    L (float): Upper limit of integration.
    
    Returns:
    float: The value of the integral.
    """
    # Integrate g(x, V) from 0 to L
    result, _ = quad(g, 0, L, args=(V,))
    return result

# Define the range of V values
V_values = np.linspace(-2, 2, 200)  # 100 points between -1 and 1

# Compute ginf(V) for each V in the range
ginf_values = [dg*ginf(V, L)/L for V in V_values]

# Plot ginf(V) vs V
plt.figure(figsize=(8, 6))
plt.plot(V_values, ginf_values, label=r"$g_{\text{inf}}(V)$", color="blue", linewidth=2)
plt.xlabel(r"$V$", fontsize=14)
plt.ylabel(r"$g_{\text{inf}}(V)$", fontsize=14)
plt.title(r"$g_{\text{inf}}(V) = \int_0^L g(x, V) \, dx$", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.show()