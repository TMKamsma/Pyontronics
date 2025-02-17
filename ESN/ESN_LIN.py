import numpy as np
import random

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
        guarantee_ESP=True,
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
        guarantee_ESP (bool): Guarantee Echo State properties
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
        self.guarantee_ESP = guarantee_ESP

        if leaking_rate * (step_size / time_scale) > 1:
            raise ValueError(
                "Invalid parameter combination: leaking_rate * (step_size / time_scale) must be ≤ 1."
            )
            
        if guarantee_ESP:
            if self.spectral_radius >= 1:
                raise ValueError(
                    "Invalid spectral radius: spectral radius must be < 1 to guarantee ESP. Decrease spectral radius or turn off guarantee_ESP."
                )

        # Initialize weight matrices
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights
        self.W_out = None  # Output weights

        self._initialize_weights()

        # if self.guarantee_ESP:
        #     # Check for Echo State Property guarantee (Proposition 1, Jager et al. 2007, Neural Networks)
        #     _, S, _ = np.linalg.svd(self.W_res)
        #     if (
        #         abs(
        #             1
        #             - self.step_size / self.time_scale * (self.leaking_rate - np.max(S))
        #         )
        #         > 1
        #     ):
        #         raise ValueError(
        #             f"Invalid parameter combination: abs(1-step_size/time_scale*(leaking_rate-np.max(S))) must be < 1 and now is {round(abs(1 - self.step_size / self.time_scale * (self.leaking_rate - np.max(S))),4)}. Echo State Property not guaranteed."
        #         )

    def _initialize_weights(self, seed=42):
        np.random.seed(seed)

        # Initialize input weights
        self.W_in = (
            np.random.rand(self.reservoir_size, self.input_dim) - 0.5
        ) * self.input_scaling

        # Initialize reservoir weights with desired sparsity
        
        # If guarantee_ESP is true, use ESP recipe from Yildiz et al., 2012, Neural Networks
        # guarantee_ESP step 1: generate matrix with elements w ≥ 0
        if self.guarantee_ESP:
            matrix_centering = 0
        else:
            matrix_centering = -0.5
        
        self.W_res = np.random.rand(self.reservoir_size, self.reservoir_size) + matrix_centering
        sparsity_mask = np.random.rand(*self.W_res.shape) < self.sparsity
        self.W_res[sparsity_mask] = 0

        # Adjust spectral radius (also step 2 of guarantee_ESP)
        max_eig = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        if max_eig != 0:
            self.W_res *= self.spectral_radius / max_eig
        
        # guarantee_ESP step 3: switch the sign of each matrix elements with probability of 0.5
        if self.guarantee_ESP:
            for i in range(len(self.W_res)):
                for j in range(len(self.W_res[i])):
                    if random.random() < 0.5:
                        self.W_res[i][j]=-self.W_res[i][j]
                    
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

    def visualize_reservoir(self, draw_labels=False):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        input_nodes = [f"inp_{i}" for i in range(self.input_dim)]
        reservoir_nodes = [f"res_{i}" for i in range(self.reservoir_size)]
        output_nodes = [f"out_{i}" for i in range(self.output_dim)]
        
        G.add_nodes_from(input_nodes)
        G.add_nodes_from(reservoir_nodes)
        G.add_nodes_from(output_nodes)
        
        for i in range(self.reservoir_size):
            for j in range(self.input_dim):
                w = self.W_in[i, j]
                if w != 0:
                    G.add_edge(input_nodes[j], reservoir_nodes[i], weight=w)
        
        for i in range(self.reservoir_size):
            for j in range(self.reservoir_size):
                w = self.W_res[i, j]
                if w != 0:
                    G.add_edge(reservoir_nodes[j], reservoir_nodes[i], weight=w)
        
        for i in range(self.output_dim):
            for j in range(self.reservoir_size):
                w = self.W_out[i, j]
                if w != 0:
                    G.add_edge(reservoir_nodes[j], output_nodes[i], weight=w)
        
        pos = {}
        for idx, node in enumerate(input_nodes):
            pos[node] = (0, -idx)
        for idx, node in enumerate(output_nodes):
            pos[node] = (2, -idx)
        
        reservoir_subgraph = G.subgraph(reservoir_nodes)
        pos_res = nx.spring_layout(reservoir_subgraph, k=0.9, scale=0.5)
        for node, (x_coord, y_coord) in pos_res.items():
            pos[node] = (x_coord + 1, y_coord)
        
        plt.figure(figsize=(9, 7))
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                            node_color="lightblue", node_size=500, edgecolors="black")
        nx.draw_networkx_nodes(G, pos, nodelist=reservoir_nodes,
                            node_color="lightgreen", node_size=500, edgecolors="black")
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes,
                            node_color="lightcoral", node_size=500, edgecolors="black")

        in2res_edges = []
        res2res_edges = []
        res2out_edges = []
        in2res_weights = []
        res2res_weights = []
        res2out_weights = []
        
        for (src, dst, data) in G.edges(data=True):
            w = data["weight"]
            if src in input_nodes and dst in reservoir_nodes:
                in2res_edges.append((src, dst))
                in2res_weights.append(w)
            elif src in reservoir_nodes and dst in reservoir_nodes:
                res2res_edges.append((src, dst))
                res2res_weights.append(w)
            elif src in reservoir_nodes and dst in output_nodes:
                res2out_edges.append((src, dst))
                res2out_weights.append(w)

        def get_edge_colors_and_alphas(weights, base_color):
            """ Return list of RGBA colors with alpha scaled by |weight|. """
            max_w = max(abs(w) for w in weights) if weights else 1e-9
            colors = []
            for w in weights:
                alpha = 0.1 + 0.9 * (abs(w) / max_w)
                import matplotlib.colors as mcolors
                rgba = list(mcolors.to_rgba(base_color))
                rgba[-1] = alpha
                colors.append(rgba)
            return colors

        in2res_colors = get_edge_colors_and_alphas(in2res_weights, "lightblue")
        res2res_colors = get_edge_colors_and_alphas(res2res_weights, "green")
        res2out_colors = get_edge_colors_and_alphas(res2out_weights, "red")
        
        nx.draw_networkx_edges(G, pos,
                            edgelist=in2res_edges,
                            edge_color=in2res_colors,
                            arrowstyle="-|>",
                            arrowsize=10)
        nx.draw_networkx_edges(G, pos,
                            edgelist=res2res_edges,
                            edge_color=res2res_colors,
                            arrowstyle="-|>",
                            arrowsize=10)
        nx.draw_networkx_edges(G, pos,
                            edgelist=res2out_edges,
                            edge_color=res2out_colors,
                            arrowstyle="-|>",
                            arrowsize=10)
        
        if draw_labels:
            nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("ESN Visualization: Input (left), Reservoir (middle), Output (right)")
        plt.axis("off")
        plt.show()
