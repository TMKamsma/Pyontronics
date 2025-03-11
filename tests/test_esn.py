import pytest
import numpy as np

from ESN import EchoStateNetwork


@pytest.fixture
def default_esn():
    """Returns a small ESN instance with default parameters suitable for testing."""
    return EchoStateNetwork(
        input_dim=2,
        reservoir_size=10,
        output_dim=1,
        leaking_rate=0.8,
        step_size=0.3,
        time_scale=1.0,
        spectral_radius=0.8,
        sparsity=0.2,
        input_scaling=1.0,
        regularization=1e-4,
        washout=3,
        weight_seed=42,
        activation=np.tanh,
        guarantee_ESP=False,
        progress_bar=False,
    )


def test_init_basic(default_esn):
    """Check that ESN initializes correctly and weight matrices have expected shapes."""
    esn = default_esn
    assert esn.input_dim == 2
    assert esn.reservoir_size == 10
    assert esn.output_dim == 1

    # Check if initial weights are not None
    assert esn.W_in is not None
    assert esn.W_res is not None
    # W_out should still be None before training
    assert esn.W_out is None

    assert esn.W_in.shape == (esn.reservoir_size, esn.input_dim)
    assert esn.W_res.shape == (esn.reservoir_size, esn.reservoir_size)


def test_parameter_check_fail():
    """Check that invalid parameters (leaking_rate * (step_size/time_scale) > 1) raise an error."""
    with pytest.raises(ValueError):
        EchoStateNetwork(
            input_dim=1,
            reservoir_size=5,
            output_dim=1,
            leaking_rate=2.0,  # Too high
            step_size=1.0,
            time_scale=1.0,
        )


def test_guarantee_ESP_fail():
    """Check that guarantee_ESP=True with spectral_radius >= leaking_rate raises an error."""
    with pytest.raises(ValueError):
        EchoStateNetwork(
            input_dim=1,
            reservoir_size=5,
            output_dim=1,
            leaking_rate=0.8,
            spectral_radius=0.8,
            guarantee_ESP=True,
        )


def test_fit_timestep_mode(default_esn):
    """Train ESN in per-timestep mode and check that W_out is assigned."""
    esn = default_esn
    n_steps = 50
    inputs = np.random.randn(n_steps, esn.input_dim)
    targets = np.random.randn(n_steps, esn.output_dim)

    esn.fit(inputs, targets)
    assert esn.W_out is not None
    assert esn.W_out.shape == (esn.output_dim, esn.reservoir_size)
    assert esn.training_mode == "timestep"


def test_predict_timestep_mode(default_esn):
    """Train and then predict in timestep mode, verifying the output shape."""
    esn = default_esn
    n_steps = 30
    inputs = np.random.randn(n_steps, esn.input_dim)
    targets = np.random.randn(n_steps, esn.output_dim)
    esn.fit(inputs, targets)

    test_steps = 10
    test_inputs = np.random.randn(test_steps, esn.input_dim)
    preds = esn.predict(test_inputs)
    assert preds.shape == (test_steps, esn.output_dim)


def test_fit_pulses_mode(default_esn):
    """Train ESN in pulses mode (one target per input sequence)."""
    esn = default_esn
    pulses = []
    lengths = [8, 10, 7, 5, 9]
    for L in lengths:
        pulses.append(np.random.randn(L, esn.input_dim))

    targets = np.random.randn(len(pulses), esn.output_dim)
    esn.fit(pulses, targets)

    assert esn.W_out is not None
    assert esn.training_mode == "pulses"


def test_predict_pulses_mode(default_esn):
    """Train ESN in pulses mode, then predict on similar pulses."""
    esn = default_esn
    pulses = []
    lengths = [5, 6, 7]
    for L in lengths:
        pulses.append(np.random.randn(L, esn.input_dim))
    targets = np.random.randn(len(pulses), esn.output_dim)

    esn.fit(pulses, targets)
    outputs = esn.predict(pulses)
    assert outputs.shape == (len(pulses), esn.output_dim)


def test_mismatched_lengths_timestep(default_esn):
    """Check error with mismatched input and target length in timestep mode."""
    esn = default_esn
    inputs = np.random.randn(20, esn.input_dim)
    targets = np.random.randn(19, esn.output_dim)  # mismatch
    with pytest.raises(ValueError):
        esn.fit(inputs, targets)


def test_mismatched_pulses(default_esn):
    """Check error when the number of pulses differs from the number of target rows."""
    esn = default_esn
    pulses = [np.random.randn(5, esn.input_dim) for _ in range(3)]
    targets = np.random.randn(2, esn.output_dim)  # mismatch
    with pytest.raises(ValueError):
        esn.fit(pulses, targets)
