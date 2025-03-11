import numpy as np
import optuna
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

from ESN import EchoStateNetwork, MackeyGlassGenerator, GinfActivator  # noqa: E402

ginf_activator = GinfActivator(V_min=-2, V_max=2, resolution=200, offset=True)


activation_functions = {
    "tanh": np.tanh,
    "ginf": ginf_activator.activate,
}


def objective(trial):
    tau = np.random.randint(10, 30)
    trial.set_user_attr("tau", tau)

    mg_series = MackeyGlassGenerator(tau=tau, n=10000, n_samples=5000)
    mg_series = (
        2 * (mg_series - mg_series.min()) / (mg_series.max() - mg_series.min()) - 1
    )

    inputs = mg_series[:-1].reshape(-1, 1)
    targets = mg_series[1:].reshape(-1, 1)

    train_len = 4000
    test_len = 1000
    train_x = inputs[:train_len]
    train_y = targets[:train_len]
    test_x = inputs[train_len : train_len + test_len]
    test_y = targets[train_len : train_len + test_len]

    reservoir_size = trial.suggest_int("reservoir_size", 1, 400, log=True)
    leaking_rate = trial.suggest_float("leaking_rate", 0.01, 1.0)
    step_size = trial.suggest_float("step_size", 0.0001, 1, log=True)
    time_scale = trial.suggest_float("time_scale", 0.1, 5.0)
    spectral_radius = trial.suggest_float("spectral_radius", 0.01, leaking_rate)
    sparsity = trial.suggest_float("sparsity", 0.01, 0.99)
    input_scaling = trial.suggest_float("input_scaling", 0.1, 10.0, log=True)
    regularization = trial.suggest_float("regularization", 1e-6, 1e-2, log=True)

    if leaking_rate * (step_size / time_scale) > 1:
        raise optuna.exceptions.TrialPruned("Leaking rate * step_size/time_scale > 1")

    if spectral_radius >= leaking_rate:
        raise optuna.exceptions.TrialPruned("Spectral_radius < Leaking_rate")

    esn = EchoStateNetwork(
        input_dim=1,
        reservoir_size=reservoir_size,
        output_dim=1,
        leaking_rate=leaking_rate,
        step_size=step_size,
        time_scale=time_scale,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        regularization=regularization,
        activation=ginf_activator.activate,
        guarantee_ESP=True,
        progress_bar=False,
    )
    esn.fit(train_x, train_y)

    predictions = esn.predict(test_x)
    return np.mean((predictions - test_y) ** 2)


try:
    study_name = "ESN optimization MG"
    storage = optuna.storages.RDBStorage(
        "sqlite:///optuna_esn.db", engine_kwargs={"connect_args": {"timeout": 20.0}}
    )

    optuna.delete_study(study_name=study_name, storage=storage)
except Exception:
    print("No database available")

study = optuna.create_study(
    direction="minimize",
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=600, n_jobs=-1)
