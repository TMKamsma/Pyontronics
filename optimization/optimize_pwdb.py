from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import numpy as np
import optuna
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)
from ESN import EchoStateNetwork, GinfActivator  # noqa: E402

ginf_activator = GinfActivator(V_min=-2, V_max=2, resolution=200, offset=True)

pathlist = Path(r"data/pulse-wave-database/PWs_csv/csv").glob("**/*.csv")
dfs = []
for path in pathlist:
    df = pd.read_csv(path)
    _, df["filename"], df["Type"] = path.name.split("_")
    df["Type"] = df["Type"].str.replace(r"\.csv$", "", regex=True)
    dfs.append(df)
final_df = pd.concat(dfs, ignore_index=True)

data = pd.read_csv(r"data/pulse-wave-database/m.csv")
data.index = data.index + 1
df = final_df.merge(data, left_on="Subject Number", right_index=True, how="left")
pulse_columns = [col for col in df.columns if col.startswith(" pt")]
df["pulse"] = df[pulse_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Drop the original pulse columns
df = df.drop(columns=pulse_columns)
df = df[df.Type == "P"]
df = df.dropna(subset=["SI"])
df = df.reset_index(drop=True)

df_merged = df[["filename", "SI", "pulse"]].copy()
df_merged["SI"] = (df_merged["SI"] - df_merged["SI"].min()) / (
    df_merged["SI"].max() - df_merged["SI"].min()
)
df_merged = shuffle(df_merged, random_state=42)

train_samples = []
test_samples = []

unique_filenames = df_merged["filename"].unique()
for filename in unique_filenames:
    class_subset = df_merged[df_merged["filename"] == filename]

    train_samples.append(class_subset.iloc[:100])
    test_samples.append(class_subset.iloc[100:110])

train_set = pd.concat(train_samples).sample(frac=1, random_state=42)
test_set = pd.concat(test_samples).sample(frac=1, random_state=42)

train_x = list(train_set["pulse"])
train_y = train_set["SI"].values.reshape(-1, 1)
test_x = list(test_set["pulse"])
test_y = test_set["SI"].values.reshape(-1, 1)


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
        guarantee_ESP=True,
        progress_bar=False,
    )

    n_data_points = trial.suggest_int("training_data", 100, len(train_x))

    esn.fit(train_x[:n_data_points], train_y[:n_data_points])

    predictions = esn.predict(test_x)
    return np.mean((predictions - test_y) ** 2)


try:
    study_name = "ESN optimization Pulse"
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
study.optimize(objective, n_trials=600, n_jobs=30)
