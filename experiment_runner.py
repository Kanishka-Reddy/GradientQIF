# experiment_runner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import os

from qif_layer import QIFLayer
from custom_optimizer import QIFAwareOptimizer
from critical_init import qif_critical_init


class QIFNetwork(nn.Module):
    """
    Spiking Neural Network with QIF neurons.

    Uses our derived surrogate gradient for training.
    """

    def __init__(self, input_size, hidden_size, output_size, time_steps=50,
                 use_derived_grad=True):
        """
        Initialize QIF network.

        Args:
            input_size (int): Input dimensionality
            hidden_size (int): Number of hidden neurons
            output_size (int): Number of output neurons
            time_steps (int): Number of time steps to simulate
            use_derived_grad (bool): Use our derived surrogate gradient
        """
        super().__init__()

        self.time_steps = time_steps

        # Input encoding layer (converts continuous values to spike trains)
        self.input_layer = nn.Linear(input_size, input_size)

        # QIF layers
        self.qif1 = QIFLayer(input_size, hidden_size, use_derived_grad=use_derived_grad)
        self.qif2 = QIFLayer(hidden_size, output_size, use_derived_grad=use_derived_grad)

        # Apply critical initialization
        self.apply(lambda m: qif_critical_init(m) if isinstance(m, nn.Linear) else m)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_size]

        Returns:
            torch.Tensor: Network output
        """
        batch_size = x.size(0)

        # Reset states
        self.reset_states()

        # Initialize membrane potentials
        v1 = torch.zeros(batch_size, self.qif1.output_size, device=x.device)
        v2 = torch.zeros(batch_size, self.qif2.output_size, device=x.device)

        # Apply input encoding (convert to spike probability)
        x_encoded = torch.sigmoid(self.input_layer(x))

        # Initialize spike count for readout
        spike_count = torch.zeros(batch_size, self.qif2.output_size, device=x.device)

        # Simulate network for multiple time steps
        for t in range(self.time_steps):
            # Generate input spikes
            input_spikes = torch.bernoulli(x_encoded)

            # Forward pass through layers
            spikes1, v1 = self.qif1(input_spikes, v1)
            spikes2, v2 = self.qif2(spikes1, v2)

            # Accumulate output spikes
            spike_count += spikes2

        # Normalize spike count by time steps to get firing rate
        outputs = spike_count / self.time_steps

        return outputs

    def reset_states(self):
        """Reset all layer states."""
        self.qif1.reset_state()
        self.qif2.reset_state()


def train_qif_network(model, train_loader, test_loader, device,
                      optimizer_type='qif_aware', epochs=10, learning_rate=1e-3):
    """
    Train a QIF network.

    Args:
        model (nn.Module): Network to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Computation device
        optimizer_type (str): 'qif_aware' or 'adam'
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate

    Returns:
        dict: Training results
    """
    model = model.to(device)

    # Initialize optimizer
    if optimizer_type == 'qif_aware':
        optimizer = QIFAwareOptimizer(model.parameters(), lr=learning_rate, gain_scaling=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        # Record time
        epoch_time = time.time() - epoch_start

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')

    return history


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model performance.

    Args:
        model (nn.Module): Network to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Computation device

    Returns:
        tuple: (loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(data_loader), 100. * correct / total


def plot_results(history_derived, history_standard=None, save_path=None):
    """
    Plot training results.

    Args:
        history_derived (dict): Training history with derived gradient
        history_standard (dict, optional): Training history with standard gradient
        save_path (str, optional): Path to save figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training and test accuracy
    epochs = range(1, len(history_derived['train_acc']) + 1)

    ax1.plot(epochs, history_derived['train_acc'], 'b-', label='Derived (Train)')
    ax1.plot(epochs, history_derived['test_acc'], 'b--', label='Derived (Test)')

    if history_standard:
        ax1.plot(epochs, history_standard['train_acc'], 'r-', label='Standard (Train)')
        ax1.plot(epochs, history_standard['test_acc'], 'r--', label='Standard (Test)')

    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    # Plot training and test loss
    ax2.plot(epochs, history_derived['train_loss'], 'b-', label='Derived (Train)')
    ax2.plot(epochs, history_derived['test_loss'], 'b--', label='Derived (Test)')

    if history_standard:
        ax2.plot(epochs, history_standard['train_loss'], 'r-', label='Standard (Train)')
        ax2.plot(epochs, history_standard['test_loss'], 'r--', label='Standard (Test)')

    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Plot epoch times
    ax3.plot(epochs, history_derived['epoch_times'], 'b-', label='Derived')
    if history_standard:
        ax3.plot(epochs, history_standard['epoch_times'], 'r-', label='Standard')

    ax3.set_title('Training Time per Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (s)')
    ax3.legend()
    ax3.grid(True)

    # Plot convergence rate (ratio of test accuracy to epoch)
    derived_convergence = [acc / (i + 1) for i, acc in enumerate(history_derived['test_acc'])]
    ax4.plot(epochs, derived_convergence, 'b-', label='Derived')

    if history_standard:
        standard_convergence = [acc / (i + 1) for i, acc in enumerate(history_standard['test_acc'])]
        ax4.plot(epochs, standard_convergence, 'r-', label='Standard')

    ax4.set_title('Convergence Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy / Epoch')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def run_mnist_experiment():
    """Run experiment on MNIST dataset."""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Create models
    input_size = 28 * 28
    hidden_size = 256
    output_size = 10
    time_steps = 20

    model_derived = QIFNetwork(input_size, hidden_size, output_size, time_steps,
                               use_derived_grad=True)
    model_standard = QIFNetwork(input_size, hidden_size, output_size, time_steps,
                                use_derived_grad=False)

    # Train with derived surrogate gradient
    print("Training with derived surrogate gradient...")
    history_derived = train_qif_network(
        model_derived, train_loader, test_loader, device,
        optimizer_type='qif_aware', epochs=5, learning_rate=1e-3
    )

    # Train with standard surrogate gradient
    print("Training with standard surrogate gradient...")
    history_standard = train_qif_network(
        model_standard, train_loader, test_loader, device,
        optimizer_type='adam', epochs=5, learning_rate=1e-3
    )

    # Plot and save results
    os.makedirs('results', exist_ok=True)
    plot_results(history_derived, history_standard, 'results/mnist_comparison.png')

    return history_derived, history_standard, model_derived, model_standard


def run_validation_experiment():
    """
    Run experiment to validate the theoretical predictions:
    - Compare analytical vs. numerical firing rates
    - Verify gradient properties near critical point
    - Measure gradient flow through layers
    """
    # Set up Fokker-Planck solver
    from fokker_planck import FokkerPlanckQIF

    fp_solver = FokkerPlanckQIF(v_min=-10.0, v_max=10.0, n_points=1000, dt=0.1)

    # Experiment 1: Compare analytical vs. numerical firing rates
    mu_values = np.linspace(-1.0, 1.0, 21)
    sigma = 0.5

    analytical_rates = []
    numerical_rates = []

    print("Validating mean-field approximation...")
    for mu in tqdm(mu_values):
        # Analytical rate
        analytical = fp_solver.analytical_firing_rate(mu, sigma)
        analytical_rates.append(analytical)

        # Numerical rate from FP equation
        ss = fp_solver.steady_state(mu, sigma)
        numerical_rates.append(ss['firing_rate'])

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, analytical_rates, 'b-', label='Analytical')
    plt.plot(mu_values, numerical_rates, 'ro', label='Numerical')
    plt.xlabel('Mean Input Current (μ)')
    plt.ylabel('Firing Rate (ν)')
    plt.title('QIF Neuron Firing Rate: Analytical vs. Numerical')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/firing_rate_validation.png')
    plt.show()

    # Experiment 2: Verify gradient properties near critical point
    mu_values = np.linspace(-1.0, 1.0, 101)
    firing_rates = np.array([fp_solver.analytical_firing_rate(mu, sigma) for mu in mu_values])

    # Calculate numerical derivatives
    d_nu_d_mu = np.gradient(firing_rates, mu_values)

    # Find the critical point (maximum of derivative)
    critical_idx = np.argmax(d_nu_d_mu)
    critical_mu = mu_values[critical_idx]

    # Plot the gain curve
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, d_nu_d_mu, 'b-', label='Gain (dν/dμ)')
    plt.axvline(x=critical_mu, color='r', linestyle='--',
                label=f'Critical Point: μ ≈ {critical_mu:.3f}')
    plt.axvline(x=-sigma / 4, color='g', linestyle=':',
                label=f'Predicted Critical: μ ≈ {-sigma / 4:.3f}')
    plt.xlabel('Mean Input Current (μ)')
    plt.ylabel('Gain (dν/dμ)')
    plt.title('QIF Neuron Gain: Verifying Critical Point')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/critical_point_validation.png')
    plt.show()

    # Print critical point validation
    print(f"Numerically found critical point: μ ≈ {critical_mu:.4f}")
    print(f"Theoretically predicted: μ ≈ {-sigma / 4:.4f}")

    return {
        'mu_values': mu_values,
        'analytical_rates': analytical_rates,
        'numerical_rates': numerical_rates,
        'gain': d_nu_d_mu,
        'critical_mu': critical_mu
    }


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Run validation experiments
    validation_results = run_validation_experiment()

    # Run MNIST experiment
    history_derived, history_standard, model_derived, model_standard = run_mnist_experiment()