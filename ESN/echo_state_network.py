import numpy as np
import random
from tqdm import tqdm


class EchoStateNetwork:
    def __init__(
        self,
        input_dim,
        reservoir_size,
        output_dim,
        leaking_rate=1.0,
        step_size=0.3,
        time_scale=1.0,
        spectral_radius=0.9,
        sparsity=0.5,
        input_scaling=1.0,
        regularization=1e-4,
        washout=100,
        weight_seed=42,
        activation=np.tanh,
        guarantee_ESP=True,
        progress_bar=True,
    ):
        """
        Echo State Network with optional guarantee of the Echo State Property (ESP).
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
        self.washout = washout
        self.weight_seed = weight_seed
        self.activation = activation
        self.guarantee_ESP = guarantee_ESP
        self.progress_bar = not progress_bar

        self.training_mode = None

        self._check_parameters()

        # Weight matrices
        self.W_in = None
        self.W_res = None
        self.W_out = None

        self._initialize_all_weights()

    def _check_parameters(self):
        """Checks whether the chosen parameters are valid."""
        if self.leaking_rate * (self.step_size / self.time_scale) > 1:
            raise ValueError("leaking_rate * (step_size / time_scale) must be ≤ 1.")

        # Equivalent to step 2 of Yildiz et al. 2012 algorithm
        if self.guarantee_ESP and self.spectral_radius >= self.leaking_rate:
            raise ValueError(
                "spectral_radius must be < leaking_rate if guarantee_ESP is True."
            )

    def _initialize_all_weights(self):
        """Initializes input and reservoir weights, then adjusts spectral radius."""
        np.random.seed(self.weight_seed)

        # Input weights
        self.W_in = self._initialize_input_weights()

        # Reservoir weights
        self.W_res = self._initialize_reservoir_weights()

        # Adjust spectral radius
        self._apply_spectral_radius()

        # Sign switching if guaranteeing ESP (step 3 of Yildiz et al. 2012 algorithm)
        if self.guarantee_ESP:
            self._random_sign_switch()

        # print(np.max(np.abs(np.linalg.eigvals((self.step_size / self.time_scale)*np.absolute(self.W_res)-(1-self.leaking_rate * (self.step_size / self.time_scale))*np.identity(self.reservoir_size)))))
        # print(np.max(np.abs(np.linalg.eigvals(self.W_res))))

        max_eig_M = np.max(
            np.abs(
                np.linalg.eigvals(
                    (self.step_size / self.time_scale) * np.absolute(self.W_res)
                    - (1 - self.leaking_rate * (self.step_size / self.time_scale))
                    * np.identity(self.reservoir_size)
                )
            )
        )

        # Notify if ESP is still guaranteed
        if self.guarantee_ESP is False:
            if max_eig_M < 1:
                print("This initilization has the echo state property")
            else:
                print("This initilization might not have the echo state property")

    def _initialize_input_weights(self):
        """Creates input weight matrix of shape (reservoir_size, input_dim)."""
        return (
            np.random.rand(self.reservoir_size, self.input_dim) - 0.5
        ) * self.input_scaling

    def _initialize_reservoir_weights(self):
        """
        Creates a sparse reservoir weight matrix of shape
        (reservoir_size, reservoir_size).
        """
        # If guaranteeing ESP, shift matrix to be initially ≥ 0 for (step 1 Yildiz et al. 2012 algorithm)
        shift = 0.0 if self.guarantee_ESP else -0.5
        W_res = np.random.rand(self.reservoir_size, self.reservoir_size) + shift

        # Apply sparsity
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        W_res[mask] = 0
        return W_res

    def _apply_spectral_radius(self):
        """
        Adjusts the reservoir matrix to have the desired spectral radius.
        """
        max_eig = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        if max_eig != 0:
            self.W_res *= self.spectral_radius / max_eig

    def _random_sign_switch(self):
        """
        With probability 0.5, flips the sign of each reservoir weight.
        Only called when guarantee_ESP=True.
        """
        for i in range(self.reservoir_size):
            for j in range(self.reservoir_size):
                if random.random() < 0.5:
                    self.W_res[i, j] = -self.W_res[i, j]

    def _apply_reservoir_dynamics(self, x, u):
        """
        Updates the reservoir state using the leaky integrator equation.
        """
        alpha =  (self.step_size / self.time_scale)
        return (1 - self.leaking_rate *alpha) * x + alpha * self.activation(
            np.dot(self.W_in, u) + np.dot(self.W_res, x)
        )

    def fit(self, inputs, targets):
        """
        Train in two possible modes:
          - Per-timestep mode: inputs is a 2D np.array of shape (n_steps, input_dim)
                               => one target per time step
          - Single-label mode: inputs is a list of pulses (each pulse is shape (T, input_dim))
                               => one target per pulse
        """
        # Detect which mode to use
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            # -------------------------------------------------
            # Mode: Per-timestep
            # -------------------------------------------------
            self.training_mode = "timestep"

            n_steps = inputs.shape[0]
            extended_length = n_steps + self.washout

            # Build extended input by prepending washout zeros
            extended_inputs = np.zeros((extended_length, self.input_dim))
            extended_inputs[self.washout :] = inputs

            # Collect reservoir states over extended input
            states = np.zeros((extended_length, self.reservoir_size))
            x = np.zeros(self.reservoir_size)

            for t in tqdm(
                range(extended_length),
                desc="Training (time-step)",
                disable=self.progress_bar,
                total=extended_length,
            ):
                x = self._apply_reservoir_dynamics(x, extended_inputs[t])
                states[t] = x

            # Discard the first `washout` states
            states = states[self.washout :]

            # Targets for each time step must align with the last states
            # so we do not discard anything from 'targets' if it already had shape (n_steps, output_dim).
            # Make sure 'targets' also has shape (n_steps, output_dim).
            if len(targets) != n_steps:
                raise ValueError(
                    "Mismatch: inputs has n_steps={} but targets has {} rows.".format(
                        n_steps, len(targets)
                    )
                )

        elif isinstance(inputs, list):
            # -------------------------------------------------
            # Mode: Single-label per pulse (variable lengths)
            # -------------------------------------------------
            self.training_mode = "pulses"

            n_pulses = len(inputs)
            states = np.zeros((n_pulses, self.reservoir_size))

            for i, pulse in tqdm(
                enumerate(inputs),
                desc="Training (pulses)",
                disable=self.progress_bar,
                total=len(inputs),
            ):
                # Convert each pulse to array shape (T, input_dim)
                pulse_array = np.array(pulse).reshape(-1, self.input_dim)
                T = pulse_array.shape[0]

                # Build extended pulse with zeros at the front
                extended_pulse = np.zeros((T + self.washout, self.input_dim))
                extended_pulse[self.washout :] = pulse_array

                # Run reservoir
                x = np.zeros(self.reservoir_size)
                for t in range(T + self.washout):
                    x = self._apply_reservoir_dynamics(x, extended_pulse[t])

                # Store final state in 'states'
                states[i] = x

            # 'targets' should have shape (n_pulses, output_dim)
            if len(targets) != n_pulses:
                raise ValueError(
                    "Mismatch: got n_pulses={} but targets has {} rows.".format(
                        n_pulses, len(targets)
                    )
                )

        else:
            raise ValueError(
                "Inputs must be a 2D NumPy array (per-timestep) or list of pulses (single-label)."
            )

        self._solve_ridge(states, targets)

    def _solve_ridge(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Solves the ridge regression problem for W_out."""
        A = states.T @ states + self.regularization * np.eye(self.reservoir_size)
        B = states.T @ targets
        self.W_out = np.linalg.solve(A, B).T

    def predict(self, inputs, initial_state=None):
        """
        Prediction in the same two modes:
          - Per-timestep: 2D NumPy (n_steps, input_dim) => outputs shape (n_steps, output_dim)
          - Single-label: list of pulses => outputs shape (n_pulses, output_dim)
        """
        if self.training_mode == "timestep":
            if not (isinstance(inputs, np.ndarray) and inputs.ndim == 2):
                raise ValueError(
                    "ESN was trained in per-timestep mode, but got a different type of inputs."
                )

            n_steps = inputs.shape[0]
            extended_length = n_steps + self.washout
            extended_inputs = np.zeros((extended_length, self.input_dim))
            extended_inputs[self.washout :] = inputs

            x = (
                np.zeros(self.reservoir_size)
                if initial_state is None
                else initial_state.copy()
            )
            all_states = np.zeros((extended_length, self.reservoir_size))

            for t in range(extended_length):
                if t < len(extended_inputs)/4:
                    x = self._apply_reservoir_dynamics(x, extended_inputs[t])
                else:
                    previous_network_output = self.W_out @ all_states[t-1].T
                    x = self._apply_reservoir_dynamics(x, previous_network_output)
                all_states[t] = x

            # Discard the first washout states
            trimmed_states = all_states[self.washout :]

            # Compute output at every time step
            outputs = (self.W_out @ trimmed_states.T).T  # shape: (n_steps, output_dim)
            return outputs

        elif self.training_mode == "pulses":
            if not isinstance(inputs, list):
                raise ValueError(
                    "ESN was trained in single-label-pulse mode, but got non-list inputs."
                )

            n_pulses = len(inputs)
            outputs = np.zeros((n_pulses, self.output_dim))

            for i, pulse in enumerate(inputs):
                pulse_array = np.array(pulse).reshape(-1, self.input_dim)
                T = pulse_array.shape[0]

                extended_pulse = np.zeros((T + self.washout, self.input_dim))
                extended_pulse[self.washout :] = pulse_array

                x = (
                    np.zeros(self.reservoir_size)
                    if initial_state is None
                    else initial_state.copy()
                )
                for t in range(T + self.washout):
                    x = self._apply_reservoir_dynamics(x, extended_pulse[t])

                # Final reservoir state => one prediction
                outputs[i] = self.W_out @ x

            return outputs

        else:
            raise ValueError("Network wasn't trained yet or mode is unknown.")

    def get_physical_length(self):
        tau = self.time_scale / self.leaking_rate
        D = 1 * 10**-9
        Length = np.sqrt(12 * D * tau)
        return Length * 10**6
