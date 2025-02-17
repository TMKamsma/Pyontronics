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
        activation=np.tanh,
        check_for_ESP=True,
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
        activation (func): Activation function for ESN nodes
        """

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.leaking_rate = leaking_rate
        self.step_size = step_size
        self.time_scale = time_scale
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.regularization = regularization
        self.activation = activation

        if leaking_rate * (step_size / time_scale) > 1:
            raise ValueError(
                "Invalid parameter combination: leaking_rate * (step_size / time_scale) must be ≤ 1."
            )

        # Initialize weight matrices
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights
        self.W_out = None  # Output weights

        self._initialize_weights()
        
        # Check for Echo State Property (Proposition 1, Jager et al. 2007, Neural Networks)
        U, S, Vh = np.linalg.svd(self.W_res)
        
        if check_for_ESP:
            if abs(1-self.step_size/self.time_scale*(self.leaking_rate-np.max(S))) > 1:
                raise ValueError(
                    f"Invalid parameter combination: abs(1-step_size/time_scale*(leaking_rate-np.max(S))) must be < 1 and now is {abs(1-self.step_size/self.time_scale*(self.leaking_rate-np.max(S)))}. Echo State Property not guaranteed."
                )
        
    def _initialize_weights(self):
        # Initialize input weights
        self.W_in = (
            np.random.rand(self.reservoir_size, self.input_dim) - 0.5
        ) * self.input_scaling

        # Initialize reservoir weights with desired sparsity
        self.W_res = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        sparsity_mask = np.random.rand(*self.W_res.shape) < self.sparsity
        self.W_res[sparsity_mask] = 0
        
        # Adjust spectral radius
        max_eig = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        if max_eig != 0:
            self.W_res *= self.spectral_radius / max_eig
            
        
        
    def _apply_reservoir_dynamics(self, x, u):
        return (1 - self.leaking_rate * self.step_size / self.time_scale) * x + (
            self.step_size / self.time_scale
        ) * self.activation(np.dot(self.W_in, u) + np.dot(self.W_res, x))

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
            x = self._apply_reservoir_dynamics(x, u)
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
            x = self._apply_reservoir_dynamics(x, u)
            states[t] = x
            outputs[t] = np.dot(self.W_out, x)

        return outputs

    def visualize_reservoir(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        for i in range(self.reservoir_size):
            G.add_node(i)

        for i in range(self.reservoir_size):
            for j in range(self.reservoir_size):
                weight = self.W_res[i, j]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)

        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=400, node_color="skyblue", edgecolors="black")

        edges = G.edges(data=True)
        edge_colors = [d["weight"] for (_, _, d) in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle="-|>", alpha=0.8, edge_color=edge_colors, edge_cmap=plt.cm.Blues)

        plt.title("Echo State Network Reservoir Visualization")
        plt.show()