import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_data(N=300, days=100, batch_size=100, process="d", observations="s", beta=None, gamma=None):
    # Initialise tensor of shape (batch_size, 3, days)
    processData = torch.zeros(batch_size, 3, days)
    caseData = torch.zeros(batch_size, 1, days)


    D = torch.distributions.Uniform(low=2, high=6).sample((batch_size,))
    R0 = torch.distributions.Uniform(low=1.01, high=3).sample((batch_size,))

    theta = torch.zeros(batch_size, 2)


    if gamma is not None:
        theta[:, 1] = gamma
    else:
        theta[:, 1] = 1/D
    if beta is not None:
        theta[:, 0] = beta
    else:
        theta[:, 0] = R0 * theta[:, 1]


    # Set initial compartment populations
    processData[:, 0, 0] = 0.99 * N  # S
    processData[:, 1, 0] = 0.01 * N  # I
    processData[:, 2, 0] = 0              # R

    for day in range(days - 1):
        S = processData[:, 0, day]
        I = processData[:, 1, day]
        R = processData[:, 2, day]

        beta = theta[:, 0]
        gamma = theta[:, 1]

        if process[0].lower() == "d":
            new_infections = beta * S * I / N
            new_recoveries = gamma * I
        elif process[0].lower() == "s":
            new_infections = torch.distributions.Binomial(total_count=S, probs=beta * I / N).sample()
            new_recoveries = torch.distributions.Binomial(total_count=I, probs=gamma).sample()
        if observations[0].lower() == "d":
            caseData[:, 0, day] = torch.floor(new_infections)
        elif observations[0].lower() == "s":
            caseData[:, 0, day] = torch.distributions.Poisson(rate=new_infections).sample()


        processData[:, 0, day + 1] = S - new_infections
        processData[:, 1, day + 1] = I + new_infections - new_recoveries
        processData[:, 2, day + 1] = R + new_recoveries

    return theta, processData, caseData

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.nn(x)
class PosteriorMeanEstimator(nn.Module):
    def __init__(
        self,
        summary_stats=32,
        summary_kernel_size=3,
        summary_channels=64,
        estimator_width=128,
        output_dim=2,
        dropout=0.0
    ):
        super().__init__()

        # Convnet summary layer
        padding = summary_kernel_size // 2
        self.summary = nn.Sequential(
            nn.Conv1d(1, summary_channels, summary_kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(summary_channels, summary_channels, summary_kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(summary_channels, summary_stats, summary_kernel_size, padding=padding),
        )
        # MLP estimator layer
        self.estimator = MLP(summary_stats, output_dim, hidden_dim=estimator_width, dropout=dropout)

    def forward(self, x):
        # We add an extra dimension to the input to set the number of input channels to one
        x_summary = self.summary(x)
        # We sum the summary statistics across all observations, then apply the estimator
        x_estimate = self.estimator(x_summary.sum(-1))
        return x_estimate
    
def train_NN(NN=None, n_epochs=128, n_batches_per_epoch=128, training_batch_size=128, lr=0.00001, dropout=0.01, process="d", observations="s", device="cpu"):
    NN = PosteriorMeanEstimator(dropout=dropout).to(device) if NN is None else NN
    NN.to(device)
    y_test, _, x_test = generate_data(N=300, days=100, batch_size=1024, process=process, observations=observations, beta=None, gamma=None)
    y_test = y_test.to(device)
    x_test = x_test.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(NN.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        y_epoch_batches, _, x_epoch_batches = generate_data(
            N=300, days=100, batch_size=n_batches_per_epoch * training_batch_size, process=process, observations=observations, beta=None, gamma=None
        )
        y_epoch_batches = y_epoch_batches.to(device)
        x_epoch_batches = x_epoch_batches.to(device)

        for batch_idx in range(n_batches_per_epoch):
            batch_idx_start = batch_idx * training_batch_size
            batch_idx_end = batch_idx_start + training_batch_size
            y_batch = y_epoch_batches[batch_idx_start:batch_idx_end]
            x_batch = x_epoch_batches[batch_idx_start:batch_idx_end]

            optimizer.zero_grad()
            y_pred = NN(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / n_batches_per_epoch
        with torch.no_grad():
            y_pred_test = NN(x_test)
            test_loss = loss_fn(y_pred_test, y_test).item()

        train_losses.append(avg_epoch_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

    return NN, train_losses, test_losses

def evaluate_NN(NN, testSize=1024, device="cpu", x_test=None, y_test=None):
    NN.to(device)
    NN.eval()
    if x_test is None or y_test is None:
        y_test, _, x_test = generate_data(N=300, days=100, batch_size=testSize)
    y_test = y_test.to(device)
    x_test = x_test.to(device)

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        y_pred_test = NN(x_test)
        test_loss = loss_fn(y_pred_test, y_test).item()

    # Move to CPU and convert to NumPy
    y_test = y_test.detach().cpu().numpy()
    y_pred_test = y_pred_test.detach().cpu().numpy()

    # Separate beta and gamma
    beta_true, gamma_true = y_test[:, 0], y_test[:, 1]
    beta_pred, gamma_pred = y_pred_test[:, 0], y_pred_test[:, 1]


    # Create subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot beta (predicted on x-axis, true on y-axis)
    axes[0].scatter(beta_pred, beta_true, alpha=0.5)
    axes[0].plot([beta_pred.min(), beta_pred.max()],
                [beta_pred.min(), beta_pred.max()],
                'r--', label='Perfect Prediction')
    axes[0].set_xlabel('Predicted $\\beta$')
    axes[0].set_ylabel('True $\\beta$')
    axes[0].set_title('Predicted vs True $\\beta$')
    axes[0].grid(True)
    axes[0].legend()

    # Plot gamma
    axes[1].scatter(gamma_pred, gamma_true, alpha=0.5)
    axes[1].plot([gamma_pred.min(), gamma_pred.max()],
                [gamma_pred.min(), gamma_pred.max()],
                'r--', label='Perfect Prediction')
    axes[1].set_xlabel('Predicted $\\gamma$')
    axes[1].set_ylabel('True $\\gamma$')
    axes[1].set_title('Predicted vs True $\\gamma$')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return test_loss



def plot_losses(train_losses, test_losses, startfrom=0):
    train_losses = train_losses[startfrom:]
    test_losses = test_losses[startfrom:]
    epochs = range(1+startfrom, startfrom + len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
