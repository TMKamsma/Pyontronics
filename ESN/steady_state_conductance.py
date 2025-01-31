import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define the function g(x, V)
def g(x, V):
    """
    Example function g(x, V). Replace this with your actual function.
    For demonstration, we use a simple function: g(x, V) = sin(x) * exp(-V * x)
    """
    return np.sin(x) * np.exp(-V * x)

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

# Define the range of V values and the upper limit L
V_values = np.linspace(-1, 1, 100)  # 100 points between -1 and 1
L = 10.0  # Upper limit of integration

# Compute ginf(V) for each V in the range
ginf_values = [ginf(V, L) for V in V_values]

# Plot ginf(V) vs V
plt.figure(figsize=(8, 6))
plt.plot(V_values, ginf_values, label=r"$g_{\text{inf}}(V)$", color="blue", linewidth=2)
plt.xlabel(r"$V$", fontsize=14)
plt.ylabel(r"$g_{\text{inf}}(V)$", fontsize=14)
plt.title(r"$g_{\text{inf}}(V) = \int_0^L g(x, V) \, dx$", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.show()