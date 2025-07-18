import numpy as np
import optuna
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

from pyontronics import EchoStateNetwork, BandPassNetwork, GinfActivator

ginf_activator = GinfActivator()

reservoir_size=12
teacher_ratio=0.25
washout=0
weight_seed=1
test_num=100

# Generate synthetic data (simple sine wave prediction)
t = np.linspace(0, 80 * np.pi, 800)
dt_harmonic = 80 * np.pi/800
data_bpntest = np.sin(t) * np.cos(1.2*t)

# Create input/output pairs for time series prediction
inputs_bpntest = data_bpntest[:-1].reshape(-1, 1)
targets_bpntest = data_bpntest[1:].reshape(-1, 1)

def objective_esn(trial):
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
    for i in range(test_num):
        # Initialize ESN
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
        esn.fit(inputs_bpntest, targets_bpntest)
        predictions = esn.predict(inputs_bpntest, teacher_ratio=teacher_ratio)

        mse = (targets_bpntest[375,0]-predictions[375,0])**2
        if mse<100:
            mse_mean += mse
    rmse = np.sqrt(mse_mean)
    return rmse
    
def objective_bpn(trial):
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
    for i in range(test_num):
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
        bpn.fit(inputs_bpntest, targets_bpntest)
        predictions = bpn.predict(inputs_bpntest)

        mse = (targets_bpntest[375,0]-predictions[375,0])**2
        if mse<100:
            mse_mean += mse
    rmse = np.sqrt(mse_mean)
    return rmse

for net_type, objective_fn in [
    ("ESN", objective_esn),
    ("BPN", objective_bpn),
]:
    study_name = f"{net_type} optimization HWTS"
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
    study.optimize(objective_fn, n_trials=400, n_jobs=-1)