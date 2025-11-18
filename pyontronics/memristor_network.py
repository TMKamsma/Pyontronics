import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Optional, Callable

class MemristorNetwork:
    def __init__(self, num_vertices: int, num_edges: int, ground_vertices: List[int] = [0], dt: float = 0.01, activation=np.tanh):
        """
        Initialize a memristor network with Kirchhoff's law constraints.
        
        Parameters:
        num_vertices (int): Number of vertices in the circuit
        num_edges (int): Number of memristors (edges) in the circuit  
        ground_vertices (list): Index of the ground vertex (default: 0)
        dt (float): Time step for simulation (default: 0.01)
        activation: steady state conductance of the memristors
        """
        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.ground_vertices: Set[int] = set(ground_vertices)
        self.dt = dt
        
        # Initialize connectivity matrix (incidence matrix)
        self.incidence_matrix = np.zeros((num_vertices, num_edges))
        
        # Initialize memristor parameters
        self.conductances = np.ones(num_edges)  # Initial conductances g(t)
        self.g0 = np.ones(num_edges)  # Maximum conductances g₀
        self.tau = np.ones(num_edges)  # Memory timescales τ
        self.orientations = np.ones(num_edges)  # +1 or -1 for activation(±V)
        self.activation = activation # Set steady state conductance function
        
        # Initialize vertex voltages
        self.voltages = np.zeros(num_vertices)
        
        # Set of vertices with imposed voltage (excluding ground)
        self.imposed_vertices: Set[int] = set()
        self.voltage_function: Optional[Callable[[float], Dict[int, float]]] = None
        
        # Output training parameters
        self.W_out: Optional[np.ndarray] = None  # Output weight matrix
        self.use_conductances = True  # Use conductances (True) or voltages (False) for prediction
        self.prediction_window = 10  # Prediction steps ahead (multiple of dt)
        self.regularization = 1e-4  # Ridge regression regularization
        
        # Attributes for pre-calculated masks and indices
        self._free_mask: np.ndarray = np.ones(self.num_vertices, dtype=bool)
        self._imposed_mask: np.ndarray = np.zeros(self.num_vertices, dtype=bool)
        self._update_masks() # Initial calculation
    
    def set_memristor_parameters(self, g0_values=None, tau_values=None, orientations=None):
        """
        Set memristor parameters for each edge.
        
        Parameters:
        g0_values: Array of maximum conductances g₀ for each memristor
        tau_values: Array of timescales τ for each memristor  
        orientations: Array of +1/-1 for activation(±V) orientation
        """
        if g0_values is not None:
            if len(g0_values) != self.num_edges:
                raise ValueError(f"g0_values must have length {self.num_edges}")
            self.g0 = np.array(g0_values)
        
        if tau_values is not None:
            if len(tau_values) != self.num_edges:
                raise ValueError(f"tau_values must have length {self.num_edges}")
            self.tau = np.array(tau_values)
        
        if orientations is not None:
            if len(orientations) != self.num_edges:
                raise ValueError(f"orientations must have length {self.num_edges}")
            self.orientations = np.array(orientations)
        
    def set_connectivity(self, edge_list: List[Tuple[int, int, int]]):
        """
        Set the connectivity of the network using an edge list.
        Automatically sets direction from lower to higher vertex index.
        """
        # Reset incidence matrix
        self.incidence_matrix = np.zeros((self.num_vertices, self.num_edges))
        
        for vertex_i, vertex_j, edge_idx in edge_list:
            if edge_idx >= self.num_edges:
                raise ValueError(f"Edge index {edge_idx} exceeds number of edges")
            if vertex_i >= self.num_vertices or vertex_j >= self.num_vertices:
                raise ValueError("Vertex index out of bounds")
            if vertex_i == vertex_j:
                raise ValueError(f"Edge {edge_idx} cannot connect a vertex to itself.")
            
            # Always set direction: lower index → higher index
            source = min(vertex_i, vertex_j)
            target = max(vertex_i, vertex_j)
            
            self.incidence_matrix[source, edge_idx] = 1
            self.incidence_matrix[target, edge_idx] = -1
            
            # Automatic orientation: +1 if i < j, -1 if i > j
            # This means activation(V) uses voltage drop in natural direction
            if vertex_i < vertex_j:
                self.orientations[edge_idx] = 1
            else:
                self.orientations[edge_idx] = -1
    
    def set_imposed_voltage_vertices(self, vertices: List[int]):
        """
        Set which vertices have imposed voltages.
        
        Parameters:
        vertices: List of vertex indices with imposed voltage
        """
        self.imposed_vertices = set(vertices)
        self.imposed_vertices -= self.ground_vertices

        # Update masks whenever imposed vertices are changed    
        self._update_masks()
        
    def set_imposed_voltages(self, voltage_function):
        """
        Set a function that returns imposed voltages at each time step.
        
        Parameters:
        voltage_function: Function that takes time and returns a dictionary
                         {vertex_index: voltage_value}
        """
        self.voltage_function = voltage_function
    
    def _update_masks(self):
        """Pre-calculates boolean masks for free, imposed, and ground nodes."""
        # Reset masks
        self._free_mask[:] = True
        self._imposed_mask[:] = False
        
        # Ground vertex is not free
        if self.ground_vertices:
            self._free_mask[list(self.ground_vertices)] = False
        
        # Imposed vertices are not free
        for v in self.imposed_vertices:
            self._free_mask[v] = False
            self._imposed_mask[v] = True
    
    def build_conductance_matrix(self) -> csc_matrix:
        """
        Build the conductance matrix (nodal admittance matrix) G for the circuit.
        G = A * diag(conductances) * A^T, where A is incidence matrix.
        """
        # Create diagonal conductance matrix
        G_diag = diags(self.conductances)
        
        # Compute G = A * G_diag * A^T
        A_sparse = csc_matrix(self.incidence_matrix)
        G = A_sparse @ G_diag @ A_sparse.T
        
        return G
    
    def solve_circuit(self, imposed_voltages: Dict[int, float]):
        """
        Solve the circuit using Kirchhoff's current law.
        Handles imposed voltages by elimination.
        """
        G = self.build_conductance_matrix()
        
        # Reduced system for free nodes using the pre-calculated mask
        G_reduced = G[self._free_mask, :][:, self._free_mask]
        
        rhs_reduced = np.zeros(np.sum(self._free_mask))
        
        # Use the pre-calculated mask to find imposed nodes affecting the free ones
        imposed_indices = np.where(self._imposed_mask)[0]
        
        for v_imposed in imposed_indices:
            voltage = imposed_voltages.get(v_imposed, 0.0)
            rhs_reduced -= G[self._free_mask, v_imposed].toarray().flatten() * voltage
        
        # Solve for free voltages
        free_voltages = spsolve(G_reduced, rhs_reduced)
        
        # Reconstruct full solution
        voltages = np.zeros(self.num_vertices)
        voltages[self._free_mask] = free_voltages
        
        # Set imposed voltages
        for v, voltage in imposed_voltages.items():
            voltages[v] = voltage
        
        # voltages[self.ground_vertices] = 0.0
        
        # Calculate currents
        currents = self.conductances * (self.incidence_matrix.T @ voltages)
        
        powers = currents * (self.incidence_matrix.T @ voltages)
        
        return voltages, currents, powers
    
    def update_memristors(self, voltages: np.ndarray, currents: np.ndarray, time: float):
        """
        Update memristor conductances using the equation:
        τ·dg/dt = g₀·(activation(orientation·V)) - g(t)
        """
        # Calculate voltage drops across each memristor
        voltage_drops = self.incidence_matrix.T @ voltages
        
        # Apply orientations: V_effective = orientation * V_drop
        effective_voltages = self.orientations * voltage_drops
        
        # Calculate the right-hand side of the ODE
        dg_dt = (self.g0 * (self.activation(effective_voltages)) - self.conductances) / self.tau
        
        # Update conductances using Euler integration
        self.conductances += dg_dt * self.dt
        
        # Ensure conductances stay positive and below 2*g₀ (physical limits)
        self.conductances = np.clip(self.conductances, 1e-2, 2 * self.g0)
    
    def simulate(self, total_time, save_interval=0.01):
        """
        Run the simulation for the given total time.
        
        Returns:
        time_points: Array of time points
        voltage_history: List of voltage arrays at each time point
        current_history: List of current arrays at each time point
        conductance_history: List of conductance arrays at each time point
        """
        if save_interval < self.dt:
            raise ValueError(f"save_interval cannot be smaller than network time step {self.dt}")
        
        if save_interval % self.dt != 0:
            raise ValueError(f"save_interval must be a multiple of network time step {self.dt}")
        
        num_steps = int(total_time / self.dt)
        save_every = int(save_interval / self.dt)
        
        time_points = []
        voltage_history = []
        current_history = []
        conductance_history = []
        powers_history = []
        
        for step in range(num_steps):
            time = step * self.dt
            
            # Get imposed voltages at current time
            imposed_voltages = self.voltage_function(time)
            
            # Solve circuit
            voltages, currents, powers = self.solve_circuit(imposed_voltages)
            
            # Update memristor states
            self.update_memristors(voltages, currents, time)
            
            # Store results at save intervals
            if step % save_every == 0:
                time_points.append(time)
                voltage_history.append(voltages.copy())
                current_history.append(currents.copy())
                conductance_history.append(self.conductances.copy())
                powers_history.append(powers)
        
        return (np.array(time_points), 
                np.array(voltage_history), 
                np.array(current_history), 
                np.array(conductance_history),
                np.array(powers_history))
    
    def set_prediction_parameters(self, use_conductances=True, prediction_window=10, regularization=1e-4):
        """
        Set parameters for output prediction.
        
        Parameters:
        use_conductances: If True, use memristor conductances as features;
                         if False, use free node voltages as features
        prediction_window: Number of steps ahead to predict (multiple of dt)
        regularization: Ridge regression regularization coefficient
        """
        self.use_conductances = use_conductances
        self.prediction_window = prediction_window
        self.regularization = regularization
    
    def get_features(self) -> np.ndarray:
        """
        Extract features for prediction based on current network state.
        
        Returns:
        features: Array of either conductances or free node voltages
        """
        if self.use_conductances:
            return self.conductances.copy()
        else:
            # Get voltages of free nodes (not ground and not imposed)
            return self.voltages[self._free_mask]
    
    def train_output(self, total_time):
        """
        Train output weights to predict future input values from network states.
        
        Parameters:
        total_time: Total simulation time for training
        
        Returns:
        W_out: Trained output weight matrix
        """
        num_steps = int(total_time / self.dt)
        
        # Collect features and targets
        features_list = []
        targets_list = []
        
        # Reset network to initial state
        self.conductances = np.ones(self.num_edges)
        self.voltages = np.zeros(self.num_vertices)
        
        for step in range(num_steps - self.prediction_window):
            time = step * self.dt
            
            # Get imposed voltages using the voltage_function (same as simulate)
            imposed_voltages = self.voltage_function(time)
            
            # Solve circuit
            self.voltages, currents, powers = self.solve_circuit(imposed_voltages)
            
            # Update memristors
            self.update_memristors(self.voltages, currents, time)
            
            # Store features for prediction
            features = self.get_features()
            features_list.append(features)
            
            # Store target (future input value to predict)
            future_time = (step + self.prediction_window) * self.dt
            future_imposed_voltages = self.voltage_function(future_time)
            target = future_imposed_voltages[list(self.imposed_vertices)[0]]
            targets_list.append(target)
        
        # Convert to arrays
        X = np.array(features_list)  # (n_samples, n_features)
        y = np.array(targets_list)   # (n_samples,)
        
        # Add bias term
        X = np.column_stack([X, np.ones(X.shape[0])])
        
        # Ridge regression: (XᵀX + λI)W_out = Xᵀy
        X_T = X.T
        A = X_T @ X + self.regularization * np.eye(X.shape[1])
        b = X_T @ y
        self.W_out = np.linalg.solve(A, b)
        
        return self.W_out
    
    def predict(self, total_time):
        """
        Generate predictions using trained output weights.
        
        Parameters:
        total_time: Total simulation time for prediction
        
        Returns:
        predictions: Array of predicted values (n_steps,)
        time_points: Array of time points
        """
        if self.W_out is None:
            raise ValueError("Output weights must be trained first")
        
        num_steps = int(total_time / self.dt)
        # predictions = np.full(num_steps, np.nan)
        powers_history = np.zeros((num_steps, self.num_edges))
        time_points = np.arange(num_steps) * self.dt
        
        # Pre-allocate the feature matrix X to avoid dynamic memory allocation
        # Shape: (num_steps, num_features + 1 bias)
        num_features = self.num_edges if self.use_conductances else np.sum(self._free_mask)
        X = np.ones((num_steps, num_features + 1)) # Initialize with 1s (bias column is already set)
        
        # Reset network to initial state
        self.conductances = np.ones(self.num_edges)
        self.voltages = np.zeros(self.num_vertices)
        
        for step in range(num_steps):
            time = step * self.dt
            
            # Get imposed voltages using the voltage_function (same as simulate)
            imposed_voltages = self.voltage_function(time)
            
            # Solve circuit
            self.voltages, currents, powers = self.solve_circuit(imposed_voltages)
            
            # Log the power per device
            powers_history[step] = powers
            
            # Update memristors
            self.update_memristors(self.voltages, currents, time)
            
            # Get features and make prediction
            X[step, :-1] = self.get_features()
        
        # X shape: (N, M+1), W_out shape: (M+1,) -> Result shape: (N,)
        predictions = X @ self.W_out
        
        return predictions, time_points, np.array(powers_history)
    
    def evaluate_prediction(self,
                            predictions: np.ndarray,
                            target_signal: np.ndarray,
                            time_points: np.ndarray,
                            powers: Optional[np.ndarray] = None,
                            time_window: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Evaluate prediction performance for a specified time window.

        Calculates MSE, RMSE, and NRMSE (normalized by the standard deviation of the true data).

        Args:
            predictions (np.ndarray): The array of predicted values.
            target_signal (np.ndarray): The array of true target values.
            time_points (np.ndarray): The array of time points corresponding to the signals.
            time_window (Optional[Tuple[float, float]]): A tuple (start_time, end_time)
                to calculate metrics for. If None, the entire series is used.

        Returns:
            Dict[str, float]: A dictionary containing 'mse', 'rmse', and 'nrmse'.
        """
        # 1. Handle potential NaNs in predictions
        valid_mask = ~np.isnan(predictions)
        
        # 2. Apply time window if specified
        if time_window:
            start_time, end_time = time_window
            if start_time >= end_time:
                raise ValueError("start_time must be less than end_time.")
            
            time_mask = (time_points >= start_time) & (time_points <= end_time)
            final_mask = valid_mask & time_mask
        else:
            final_mask = valid_mask
            
        predictions_sliced = predictions[final_mask]
        target_sliced = target_signal[final_mask]
        
        if len(target_sliced) == 0:
            print("Warning: No data in the specified time window to evaluate.")
            return {'mse': np.nan, 'rmse': np.nan, 'nrmse': np.nan}
            
        # 3. Calculate metrics on the sliced data
        error = predictions_sliced - target_sliced
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        
        target_std = np.std(target_sliced)
        if target_std < 1e-9: # Avoid division by zero for flat signals
            nrmse = np.inf if rmse > 0 else 0.0
        else:
            nrmse = rmse / target_std
            
        if powers.any():
            total_power = np.sum(powers,axis=1)
            total_energy = np.sum(total_power)*self.dt
            
        return mse, rmse, nrmse, total_power, total_energy
    
    def plot_network_schematic(self, current_voltages=None, current_conductances=None, input_seed=42):
        """
        Create a schematic visualization of the memristor network.
        
        Parameters:
        current_voltages: Array of current voltages for coloring vertices
        current_conductances: Array of current conductances for edge weights
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add vertices
        for i in range(self.num_vertices):
            G.add_node(i)
        
        # Add edges (memristors)
        for edge_idx in range(self.num_edges):
            # Find connected vertices for this edge
            connections = np.where(self.incidence_matrix[:, edge_idx] != 0)[0]
            if len(connections) == 2:
                v1, v2 = connections
                # Determine direction based on incidence matrix
                if self.orientations[edge_idx] == 1:
                    source, target = v1, v2
                else:
                    source, target = v2, v1
                G.add_edge(source, target, weight=edge_idx)
        
        # Set up the plot
        plt.figure(figsize=(6, 4.8))
        
        # Choose layout
        pos = nx.spring_layout(G, seed=input_seed)  # Consistent layout
        
        # Prepare node colors and labels
        node_colors = []
        node_labels = {}
        
        for node in G.nodes():
            if node in self.ground_vertices:
                node_colors.append('lightcoral')  # Red for ground
                node_labels[node] = f"V{node}\n(GND)"
            elif node in self.imposed_vertices:
                node_colors.append('lightgreen')  # Green for input
                node_labels[node] = f"V{node}\n(Input)"
            else:
                node_colors.append('lightblue')   # Blue for free nodes
                node_labels[node] = f"V{node}"
            
            # Add voltage values if provided
            if current_voltages is not None:
                node_labels[node] += f"\n{current_voltages[node]:.2f}V"
        
        # Prepare edge weights and labels
        edge_widths = []
        edge_labels = {}
        
        for edge in G.edges(data=True):
            edge_idx = edge[2]['weight']
            if current_conductances is not None:
                # Scale width by conductance
                width = 1 + 4 * (current_conductances[edge_idx] / np.max(current_conductances))
                edge_widths.append(width)
            else:
                edge_widths.append(2)
            
            edge_labels[(edge[0], edge[1])] = f"M{edge_idx}"
            
            # Add conductance values if provided
            if current_conductances is not None:
                edge_labels[(edge[0], edge[1])] += f"\n{current_conductances[edge_idx]:.3f}S"
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.6, edge_color='gray',
                              arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        
        plt.title("Memristor Network Schematic\n"
                  f"Grounds: {sorted(list(self.ground_vertices))}, "
                  f"Inputs: {sorted(list(self.imposed_vertices))}")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()