import numpy as np
import pandas as pd
import optuna
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from pyontronics import EchoStateNetwork, BandPassNetwork, GinfActivator

ginf_activator = GinfActivator()

teacher_ratio = 1
washout = 100
weight_seed = 1
n_runs = 25

data = pd.read_csv(r"data\ventilator-pressure-prediction\train.csv")[:100000]
split = int(len(data) * 0.8)
predict_steps = 3
train_x = data[["pressure"]][0 : split - predict_steps].to_numpy()
train_y = data[["pressure"]][predict_steps:split].to_numpy()
test_x = data[["pressure"]][split : len(data) - predict_steps].to_numpy()
test_y = data[["pressure"]][split + predict_steps : len(data)].to_numpy()
train_x_uout = data[["pressure"]][:split].to_numpy()
train_y_uout = data[["u_out"]][:split].to_numpy()
test_x_uout = data[["pressure"]][split:].to_numpy()
test_y_uout = data[["u_out"]][split:].to_numpy()


def objective_esn(trial):
    reservoir_size = 200
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

    mse_mean = 0
    for _i in range(n_runs):
        try:
            # Initialize ESN
            print('run number ', _i)
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
                washout=washout,
                activation=ginf_activator.activate,
                weight_seed=weight_seed,
                progress_bar=False,
            )
            esn.fit(train_x, train_y)
            predictions = esn.predict(test_x, teacher_ratio=teacher_ratio)

            mse_mean += np.mean((test_y - predictions) ** 2)

        except np.linalg.LinAlgError:
            raise optuna.exceptions.TrialPruned(
                "Singular matrix in fit (bad hyperparameters)"
            )
    rmse = np.sqrt(mse_mean) / n_runs
    return rmse


def objective_bpn(trial):
    reservoir_size = 200
    leaking_rate = trial.suggest_float("leaking_rate", 0.01, 1.0)
    step_size = trial.suggest_float("step_size", 0.0001, 1, log=True)
    time_scale = trial.suggest_float("time_scale", 0.1, 5.0)
    spectral_radius = trial.suggest_float("spectral_radius", 0.01, leaking_rate)
    sparsity = trial.suggest_float("sparsity", 0.01, 0.99)
    input_scaling = trial.suggest_float("input_scaling", 0.1, 10.0, log=True)
    regularization = trial.suggest_float("regularization", 1e-6, 1e-2, log=True)
    time_scale_std = trial.suggest_float("time_scale_std", 0.1, 10.0, log=True)

    if leaking_rate * (step_size / time_scale) > 1:
        raise optuna.exceptions.TrialPruned("Leaking rate * step_size/time_scale > 1")
    if spectral_radius >= leaking_rate:
        raise optuna.exceptions.TrialPruned("Spectral_radius < Leaking_rate")

    mse_mean = 0
    for _ in range(n_runs):
        try:
            bpn = BandPassNetwork(
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
                washout=washout,
                activation=ginf_activator.activate,
                weight_seed=weight_seed,
                time_scale_std=time_scale_std,
                progress_bar=False,
            )
            bpn.fit(train_x, train_y)
            predictions = bpn.predict(test_y)

            mse = np.mean((test_x - predictions) ** 2)
            mse_mean += mse

            # if mse is nan, report all kinds of data to find the problem
            if np.isnan(mse):
                print("NaN detected in MSE")
                print("Test output:", test_y_uout)
                print("Predictions:", predictions)
                print("Reservoir size:", reservoir_size)
                print("Leaking rate:", leaking_rate)

        except np.linalg.LinAlgError:
            raise optuna.exceptions.TrialPruned(
                "Singular matrix in fit (bad hyperparameters)"
            )
    rmse = np.sqrt(mse_mean) / n_runs
    return rmse


for net_type, objective_fn in [
    ("ESN", objective_esn),
    ("BPN", objective_bpn),
]:
    study_name = f"{net_type} optimization VVP"
    storage = optuna.storages.RDBStorage(
        "sqlite:///optuna_esn.db", engine_kwargs={"connect_args": {"timeout": 20.0}}
    )
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
        print("Deleted study")
    except Exception:
        print("No database available")
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective_fn, n_trials=200, n_jobs=-1)
