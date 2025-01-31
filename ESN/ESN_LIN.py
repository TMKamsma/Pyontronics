import numpy as np


class EchoStateNetwork:
    def __init__(
        self,
        input_dim,
        reservoir_size,
        output_dim,
        leaking_rate=1,
        step_size=0.3,
        time_scale=1,
        spectral_radius=0.9,
        sparsity=0.5,
        input_scaling=1.0,
        regularization=1e-4,
    ):
        """
        Initialize the Echo State Network with leaky integrator neurons

        Parameters:
        input_dim (int): Dimension of input data
        reservoir_size (int): Number of neurons in the reservoir
        output_dim (int): Dimension of output data
        leaking_rate (float): (a) Self coupling constant
        step_size (float): (d) Time step size
        time_scale (float): (c) Scale of time evolution
        spectral_radius (float): Spectral radius of reservoir weight matrix
        sparsity (float): Proportion of recurrent weights set to zero
        input_scaling (float): Scaling factor for input weights
        regularization (float): Regularization coefficient for ridge regression
        """
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.leaking_rate = leaking_rate
        self.d = step_size
        self.c = time_scale
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.regularization = regularization

        # Initialize weight matrices
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights
        self.W_out = None  # Output weights

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize input weights
        self.W_in = (
            np.random.rand(self.reservoir_size, self.input_dim) - 0.5
        ) * self.input_scaling

        # Initialize reservoir weights with desired sparsity
        self.W_res = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        # Set sparsity
        sparsity_mask = np.random.rand(*self.W_res.shape) < self.sparsity
        self.W_res[sparsity_mask] = 0

        # Adjust spectral radius
        max_eig = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        if max_eig != 0:
            self.W_res *= self.spectral_radius / max_eig

    def fit(self, inputs, targets, washout=100):
        """
        Train the ESN using given input and target data

        Parameters:
        inputs (np.ndarray): Input data (n_samples, input_dim)
        targets (np.ndarray): Target outputs (n_samples, output_dim)
        washout (int): Number of initial timesteps to discard
        """
        n_samples = inputs.shape[0]
        states = np.zeros((n_samples, self.reservoir_size))
        x = np.zeros(self.reservoir_size)

        # Collect reservoir states
        for t in range(n_samples):
            u = inputs[t]
            x = (1 - self.leaking_rate * self.d / self.c) * x + (self.d / self.c) * np.tanh(
                np.dot(self.W_in, u) + np.dot(self.W_res, x)
            )
            states[t] = x

        # Discard washout period
        states = states[washout:]
        targets = targets[washout:]

        # Train output weights using ridge regression
        X = states
        Y = targets

        # Solve (X^T X + λI) W_out = X^T Y
        X_T = X.T
        A = np.dot(X_T, X) + self.regularization * np.eye(self.reservoir_size)
        B = np.dot(X_T, Y)
        self.W_out = np.linalg.solve(A, B).T

    def predict(self, inputs, initial_state=None):
        """
        Generate predictions using the trained ESN

        Parameters:
        inputs (np.ndarray): Input data (n_samples, input_dim)
        initial_state (np.ndarray): Initial reservoir state

        Returns:
        np.ndarray: Output predictions (n_samples, output_dim)
        """
        n_samples = inputs.shape[0]
        states = np.zeros((n_samples, self.reservoir_size))
        outputs = np.zeros((n_samples, self.output_dim))
        x = (
            initial_state.copy()
            if initial_state is not None
            else np.zeros(self.reservoir_size)
        )

        for t in range(n_samples):
            u = inputs[t]
            x = (1 - self.leaking_rate * self.d / self.c) * x + (self.d / self.c) * np.tanh(
                np.dot(self.W_in, u) + np.dot(self.W_res, x)
            )
            states[t] = x
            outputs[t] = np.dot(self.W_out, x)

        return outputs


def mackey_glass(tau=17, n=1000, beta=0.2, gamma=0.1, n_samples=5000, dt=1.0):
    """
    Generate Mackey-Glass time series
    Parameters:
    tau (int): Time delay
    n (int): Number of points to generate
    beta, gamma (float): Equation parameters
    n_samples (int): Number of samples to keep
    dt (float): Time step size
    """
    history_len = tau * 10  # Initialize sufficient history
    values = np.zeros(history_len + n)

    # Initial condition (constant history)
    values[:history_len] = 1.1

    delay_steps = int(tau / dt)
    if delay_steps <= 0:
        delay_steps = 1

    for t in range(history_len, history_len + n - 1):
        x_tau = values[t - delay_steps]
        dx_dt = beta * x_tau / (1 + x_tau**10) - gamma * values[t]
        values[t + 1] = values[t] + dx_dt * dt

    # Discard transient and return requested number of samples
    return values[history_len : history_len + n_samples]
