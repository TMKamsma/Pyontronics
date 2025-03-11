import pytest
import numpy as np
from ESN import EchoStateNetwork, PulseEchoStateNetwork

@pytest.fixture
def default_esn():
    return EchoStateNetwork(
        input_dim=2,
        reservoir_size=10,
        output_dim=1,
        leaking_rate=0.8,
        step_size=0.3,
        time_scale=1.0,
        spectral_radius=0.8,
        sparsity=0.2,
        washout=3,
        weight_seed=42,
        guarantee_ESP=False,
        progress_bar=False,
    )

@pytest.fixture
def default_pulse_esn(default_esn):
    return PulseEchoStateNetwork(
        input_dim=default_esn.input_dim,
        reservoir_size=default_esn.reservoir_size,
        output_dim=default_esn.output_dim,
        leaking_rate=default_esn.leaking_rate,
        step_size=default_esn.step_size,
        time_scale=default_esn.time_scale,
        spectral_radius=default_esn.spectral_radius,
        sparsity=default_esn.sparsity,
        washout=default_esn.washout,
        weight_seed=default_esn.weight_seed,
        guarantee_ESP=default_esn.guarantee_ESP,
        progress_bar=not default_esn.progress_bar,
    )

# ===== Tests for EchoStateNetwork (Timestep Mode) =====

def test_init_basic(default_esn):
    esn = default_esn
    assert esn.input_dim == 2
    assert esn.reservoir_size == 10
    assert esn.output_dim == 1
    assert esn.W_in.shape == (10, 2)
    assert esn.W_res.shape == (10, 10)
    assert esn.W_out is None

def test_parameter_check_fail():
    with pytest.raises(ValueError):
        EchoStateNetwork(
            input_dim=1,
            reservoir_size=5,
            output_dim=1,
            leaking_rate=2.0,  # invalid
            step_size=1.0,
            time_scale=1.0,
        )

def test_guarantee_ESP_fail():
    with pytest.raises(ValueError):
        EchoStateNetwork(
            input_dim=1,
            reservoir_size=5,
            output_dim=1,
            leaking_rate=0.8,
            spectral_radius=0.8,  # invalid when guarantee_ESP=True
            guarantee_ESP=True,
        )

def test_fit_timestep_mode(default_esn):
    esn = default_esn
    inputs = np.random.randn(50, esn.input_dim)
    targets = np.random.randn(50, esn.output_dim)
    esn.fit(inputs, targets)
    assert esn.W_out is not None
    assert esn.W_out.shape == (1, 10)

def test_predict_timestep_mode(default_esn):
    esn = default_esn
    inputs = np.random.randn(30, esn.input_dim)
    targets = np.random.randn(30, esn.output_dim)
    esn.fit(inputs, targets)

    test_inputs = np.random.randn(10, esn.input_dim)
    preds = esn.predict(test_inputs)
    assert preds.shape == (10, esn.output_dim)

def test_mismatched_lengths_timestep(default_esn):
    esn = default_esn
    inputs = np.random.randn(20, esn.input_dim)
    targets = np.random.randn(19, esn.output_dim)
    with pytest.raises(ValueError):
        esn.fit(inputs, targets)

# ===== Tests for PulseEchoStateNetwork (Pulse Mode) =====

def test_init_pulse_esn(default_pulse_esn):
    esn = default_pulse_esn
    assert esn.input_dim == 2
    assert esn.reservoir_size == 10
    assert esn.output_dim == 1
    assert esn.W_in.shape == (10, 2)
    assert esn.W_res.shape == (10, 10)
    assert esn.W_out is None

def test_fit_pulses_mode(default_pulse_esn):
    esn = default_pulse_esn
    pulses = [np.random.randn(L, esn.input_dim) for L in [8, 10, 7, 5, 9]]
    targets = np.random.randn(len(pulses), esn.output_dim)
    esn.fit(pulses, targets)
    assert esn.W_out is not None
    assert esn.W_out.shape == (esn.output_dim, esn.reservoir_size)

def test_predict_pulses_mode(default_pulse_esn):
    esn = default_pulse_esn
    pulses = [np.random.randn(L, esn.input_dim) for L in [5, 6, 7]]
    targets = np.random.randn(len(pulses), esn.output_dim)
    esn.fit(pulses, targets)
    
    test_pulses = [np.random.randn(L, esn.input_dim) for L in [4, 5]]
    outputs = esn.predict(test_pulses)
    assert outputs.shape == (2, esn.output_dim)

def test_mismatched_pulses(default_pulse_esn):
    esn = default_pulse_esn
    pulses = [np.random.randn(5, esn.input_dim) for _ in range(3)]
    targets = np.random.randn(2, esn.output_dim)  # mismatch
    with pytest.raises(ValueError):
        esn.fit(pulses, targets)

def test_wrong_input_type_predict_pulse(default_pulse_esn):
    esn = default_pulse_esn
    pulses = [np.random.randn(5, esn.input_dim) for _ in range(3)]
    targets = np.random.randn(3, esn.output_dim)
    esn.fit(pulses, targets)

    wrong_input = np.random.randn(10, esn.input_dim)  # should be list
    with pytest.raises(ValueError):
        esn.predict(wrong_input)

# ===== Edge Case Tests =====

def test_empty_input_timestep(default_esn):
    esn = default_esn
    inputs = np.array([]).reshape(0, esn.input_dim)
    targets = np.array([]).reshape(0, esn.output_dim)
    with pytest.raises(ValueError):
        esn.fit(inputs, targets)

def test_empty_input_pulse(default_pulse_esn):
    esn = default_pulse_esn
    pulses = []
    targets = np.array([]).reshape(0, esn.output_dim)
    with pytest.raises(ValueError):
        esn.fit(pulses, targets)
