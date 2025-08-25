import pytest
import numpy as np
import sys
from pyontronics import EchoStateNetwork, PulseEchoStateNetwork
from pyontronics.activators import GinfActivator, NCNM_activator
from pyontronics import visualization

# ===== Fixtures =====

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

@pytest.fixture
def single_input_esn():
    """Returns an ESN with a single input dimension to allow teacher forcing tests."""
    return EchoStateNetwork(
        input_dim=1,
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
def ginf_activator_default():
    return GinfActivator()

@pytest.fixture
def ginf_activator_offset():
    return GinfActivator(offset=True)

@pytest.fixture
def ncnm_activator_default():
    return NCNM_activator()

@pytest.fixture
def ncnm_activator_offset():
    return NCNM_activator(offset=True)

@pytest.fixture
def ncnm_activator_linear():
    return NCNM_activator(tanh_transform=False)

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

def test_physical_length(default_esn):
    esn = default_esn
    assert esn.physical_length == pytest.approx(122, abs=1)

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

# ===== Teacher Forcing Tests =====
def test_teacher_ratio_all_teacher(single_input_esn):
    """
    Test that when teacher_ratio=1.0 (pure teacher forcing), the predict method 
    runs without error and returns an output of the correct shape.
    """
    esn = single_input_esn
    n_steps = 20
    inputs = np.random.randn(n_steps, esn.input_dim)
    targets = np.random.randn(n_steps, esn.output_dim)
    esn.fit(inputs, targets)
    
    test_inputs = np.random.randn(10, esn.input_dim)
    preds = esn.predict(test_inputs, teacher_ratio=1.0)
    assert preds.shape == (10, esn.output_dim)

def test_teacher_ratio_free_running(single_input_esn):
    """
    Test that when teacher_ratio=0.0 (free-running mode) the network produces 
    different predictions than when using full teacher forcing.
    """
    esn = single_input_esn
    n_steps = 20
    inputs = np.random.randn(n_steps, esn.input_dim)
    targets = np.random.randn(n_steps, esn.output_dim)
    esn.fit(inputs, targets)
    
    test_inputs = np.random.randn(10, esn.input_dim)
    preds_teacher = esn.predict(test_inputs, teacher_ratio=1.0)
    preds_free = esn.predict(test_inputs, teacher_ratio=0.0)
    
    # Assert that the two predictions differ.
    assert not np.allclose(preds_teacher, preds_free)

# ===== Tests for Activators =====

def test_ginf_lookup_table_shape(ginf_activator_default):
    V, g = ginf_activator_default.get_lookup_table()
    assert V.shape == g.shape
    assert len(V) == 200

def test_ginf_offset_mean_zero(ginf_activator_offset):
    _, g = ginf_activator_offset.get_lookup_table()
    assert np.isclose(np.mean(g), 0, atol=1e-10)

def test_ginf_activate_within_range(ginf_activator_default):
    V = np.linspace(-2, 2, 10)
    g = ginf_activator_default.activate(V)
    assert g.shape == V.shape
    assert np.all(np.isfinite(g))

def test_ginf_activate_out_of_bounds(ginf_activator_default):
    V = np.array([-10, 0, 10])
    g = ginf_activator_default.activate(V)
    # Should clip to min/max
    assert np.isfinite(g).all()
    assert g[0] == ginf_activator_default.ginf_values[0]
    assert g[-1] == ginf_activator_default.ginf_values[-1]

def test_ncnm_lookup_table_shape(ncnm_activator_default):
    V, g = ncnm_activator_default.get_lookup_table()
    assert V.shape == g.shape
    assert len(V) == 200

def test_ncnm_offset_mean_zero(ncnm_activator_offset):
    _, g = ncnm_activator_offset.get_lookup_table()
    assert np.isclose(np.mean(g), 0, atol=1e-10)

def test_ncnm_activate_within_range(ncnm_activator_default):
    V = np.linspace(-2, 2, 10)
    g = ncnm_activator_default.activate(V)
    assert g.shape == V.shape
    assert np.all(np.isfinite(g))

def test_ncnm_activate_out_of_bounds(ncnm_activator_default):
    V = np.array([-10, 0, 10])
    g = ncnm_activator_default.activate(V)
    assert np.isfinite(g).all()
    assert g[0] == ncnm_activator_default.ginf_values[0]
    assert g[-1] == ncnm_activator_default.ginf_values[-1]

def test_ncnm_linear_vs_tanh(ncnm_activator_linear, ncnm_activator_default):
    V = np.linspace(-2, 2, 10)
    g_linear = ncnm_activator_linear.activate(V)
    g_tanh = ncnm_activator_default.activate(V)
    # Should not be identical
    assert not np.allclose(g_linear, g_tanh)

def test_ginf_compute_ginf_scalar(ginf_activator_default):
    val = ginf_activator_default._compute_ginf(0.5)
    assert np.isscalar(val)
    assert np.isfinite(val)

def test_ncnm_compute_ginf_scalar(ncnm_activator_default):
    val = ncnm_activator_default._compute_ginf(0.5)
    assert np.isscalar(val)
    assert np.isfinite(val)

def test_ginf_repr_str(ginf_activator_default):
    # Just check that repr/str do not error
    assert isinstance(str(ginf_activator_default), str)
    assert isinstance(repr(ginf_activator_default), str)

def test_ncnm_repr_str(ncnm_activator_default):
    assert isinstance(str(ncnm_activator_default), str)
    assert isinstance(repr(ncnm_activator_default), str)

# ===== Visualization Tests =====

def test_visualize_reservoir_no_error(default_esn):
    esn = default_esn
    inputs = np.random.randn(50, esn.input_dim)
    targets = np.random.randn(50, esn.output_dim)
    esn.fit(inputs, targets)
    try:
        visualization.visualize_reservoir(esn)
    except Exception as e:
        pytest.fail(f"visualize_reservoir raised an exception: {e}")

def test_visualize_reservoir_with_labels_no_error(default_esn):
    esn = default_esn
    inputs = np.random.randn(50, esn.input_dim)
    targets = np.random.randn(50, esn.output_dim)
    esn.fit(inputs, targets)
    try:
        visualization.visualize_reservoir(default_esn, draw_labels=True)
    except Exception as e:
        pytest.fail(f"visualize_reservoir raised an exception: {e}")

def test_visualize_reservoir_raises_without_networkx(default_esn, monkeypatch):
    import importlib
    monkeypatch.setitem(sys.modules, "networkx", None)
    mod = importlib.reload(visualization)
    with pytest.raises(ImportError):
        mod.visualize_reservoir(default_esn)