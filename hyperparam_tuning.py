import torch
import optuna
from global_utils import PosteriorMeanEstimator, train_NN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


study = optuna.create_study(
    study_name="nn_hyperparam_tuning",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    direction="minimize"
)


def objective(trial):
    # Suggest hyperparameters
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    estimator_width = trial.suggest_categorical("estimator_width", [32, 64, 128, 256])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    summary_stats = trial.suggest_categorical("summary_stats", [32, 64, 128, 256])
    summary_channels = trial.suggest_categorical("summary_channels", [32, 64, 128, 256])

    # Instantiate model with trial params
    model = PosteriorMeanEstimator(
        summary_stats=summary_stats,
        summary_kernel_size=kernel_size,
        summary_channels=summary_channels,
        estimator_width=estimator_width,
        output_dim=2,
        dropout=dropout
    ).to(device)

    n_batches_per_epoch = int(8192 / batch_size)

    # Train and return test loss
    _, _, test_losses = train_NN(
        NN=model,
        n_epochs=256,
        n_batches_per_epoch=n_batches_per_epoch,
        training_batch_size=batch_size,
        lr=lr,
        process="d",
        observations="s",
    )

    return test_losses[-1]  # Return final test loss

# Launch study
study.optimize(objective, n_trials=1000)

# Best trial
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
