import torch
import os
from global_utils import PosteriorMeanEstimator, train_NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Best hyperparameters found using Optuna
dropout = 0.002519145083582702
lr = 3.519559766293919e-05
estimator_width = 256
kernel_size = 7
batch_size = 32
summary_stats = 128
summary_channels = 32

model_path = "best_model.pth"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = PosteriorMeanEstimator(
        summary_stats=summary_stats,
        summary_kernel_size=kernel_size,
        summary_channels=summary_channels,
        estimator_width=estimator_width,
        output_dim=2,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("No saved model found — creating new model")
    model = PosteriorMeanEstimator(
        summary_stats=summary_stats,
        summary_kernel_size=kernel_size,
        summary_channels=summary_channels,
        estimator_width=estimator_width,
        output_dim=2,
        dropout=dropout
    )
model.to(device)

n_batches_per_epoch = int(8192 / batch_size)

# Train and return test loss
NN, train_losses, test_losses = train_NN(
    NN=model,
    n_epochs=6000,
    n_batches_per_epoch=n_batches_per_epoch,
    training_batch_size=batch_size,
    lr=lr,
    process="d",
    observations="s",
    device=device
)

torch.save(NN.state_dict(), "best_model.pth")