import numpy as np
import optuna
import polars as pl
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)
from ESN import EchoStateNetwork, GinfActivator  # noqa: E402

ginf_activator = GinfActivator(V_min=-2, V_max=2, resolution=200, offset=True)

data = pl.read_csv(r"data\ventilator-pressure-prediction\train.csv")[:24000]

data = data.to_dummies(columns=["R", "C"])

split = int(len(data) * 0.8)

train_x = data.drop('id','breath_id',"pressure")[:split].to_numpy()
train_y = data[["pressure"]][:split].to_numpy()
test_x = data.drop('id','breath_id',"pressure")[split:].to_numpy()
test_y = data[["pressure"]][split:].to_numpy()

data.drop('id','breath_id',"pressure")[:split].head()

def objective(trial):
    reservoir_size = trial.suggest_int("reservoir_size", 1, 100, log=True)
    leaking_rate = trial.suggest_float("leaking_rate", 0.01, 1.0)
    step_size = trial.suggest_float("step_size", 0.0001, 1, log=True)
    time_scale = trial.suggest_float("time_scale", 0.1, 5.0)
    spectral_radius = trial.suggest_float("spectral_radius", 0.01, leaking_rate)
    sparsity = trial.suggest_float("sparsity", 0.01, 0.99)
    input_scaling = trial.suggest_float("input_scaling", 0.1, 10.0, log=True)
    regularization = trial.suggest_float("regularization", 1e-6, 1e-2, log=True)
    washout = trial.suggest_int("washout", 1, 100, log=True)

    if leaking_rate * (step_size / time_scale) > 1:
        raise optuna.exceptions.TrialPruned("Leaking rate * step_size/time_scale > 1")

    if spectral_radius >= leaking_rate:
        raise optuna.exceptions.TrialPruned("Spectral_radius < Leaking_rate")

    esn = EchoStateNetwork(
        input_dim=train_x.shape[1],
        reservoir_size=reservoir_size,
        output_dim=train_y.shape[1],
        leaking_rate=leaking_rate,
        step_size=step_size,
        time_scale=time_scale,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        regularization=regularization,
        washout=washout,
        activation=ginf_activator.activate,
        guarantee_ESP=True,
        progress_bar=False,
    )

    esn.fit(train_x, train_y)

    predictions = esn.predict(test_x)
    return np.mean((predictions - test_y) ** 2)

try:
    study_name = "ESN optimization Ventilator"
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
study.optimize(objective, n_trials=100, n_jobs=-1)