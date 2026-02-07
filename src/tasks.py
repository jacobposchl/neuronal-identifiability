"""
Cognitive task generators for RNN training and evaluation.

Each task provides:
- Trial generation (input/output sequences)
- RNN training interface
- Hidden state trajectory extraction
- Expected dynamics description
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


class TaskBase:
    """Base class for cognitive tasks."""
    
    def __init__(self, name, input_size, output_size):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
    
    def generate_trial(self, length, batch_size):
        """Generate input/output sequences. Must be implemented by subclass."""
        raise NotImplementedError
    
    def train_rnn(self, rnn, n_epochs, batch_size=32, lr=0.001, 
                  trial_length=100, verbose=True, save_path=None):
        """
        Train RNN on this task.
        
        Args:
            rnn: PyTorch RNN model
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            trial_length: Length of each trial sequence
            verbose: Print progress
            save_path: Path to save checkpoint (optional)
        
        Returns:
            rnn: Trained model
            history: Training history dict with 'loss' and 'accuracy'
        """
        optimizer = optim.Adam(rnn.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {'loss': [], 'accuracy': []}
        
        if verbose:
            print(f"\nTraining {rnn.__class__.__name__} on {self.name}")
            print(f"  Hidden size: {rnn.hidden_size}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Learning rate: {lr}")
            print("-" * 60)
        
        rnn.train()
        for epoch in range(n_epochs):
            # Generate batch
            inputs, targets = self.generate_trial(trial_length, batch_size)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = rnn(inputs, return_hidden_states=False)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute accuracy (task-specific)
            accuracy = self._compute_accuracy(outputs, targets)
            
            # Record history
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy)
            
            # Print progress
            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                      f"Loss={loss.item():.4f}, Accuracy={accuracy:.2%}")
        
        if verbose:
            print("-" * 60)
            print(f"Training complete! Final accuracy: {accuracy:.2%}\n")
        
        # Save checkpoint if requested
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'task': self.name
            }, save_path)
        
        rnn.eval()
        return rnn, history
    
    def extract_trajectories(self, rnn, n_trials=50, trial_length=200):
        """
        Extract hidden state trajectories from trained RNN.
        
        Args:
            rnn: Trained PyTorch RNN model
            n_trials: Number of trials to generate
            trial_length: Length of each trial
        
        Returns:
            hidden_states: (n_units, n_timesteps) - all trials concatenated
            inputs: (n_timesteps, input_size) - corresponding inputs
            outputs: (n_timesteps, output_size) - corresponding outputs
        """
        rnn.eval()
        all_hidden = []
        all_inputs = []
        all_outputs = []
        
        with torch.no_grad():
            for trial in range(n_trials):
                # Generate trial
                inputs, targets = self.generate_trial(trial_length, batch_size=1)
                
                # Forward pass
                outputs, hidden_states = rnn(inputs, return_hidden_states=True)
                
                # Store (remove batch dimension)
                all_hidden.append(hidden_states[0].numpy())  # (seq_len, hidden_size)
                all_inputs.append(inputs[0].numpy())  # (seq_len, input_size)
                all_outputs.append(outputs[0].numpy())  # (seq_len, output_size)
        
        # Concatenate all trials
        hidden_states = np.concatenate(all_hidden, axis=0)  # (n_trials*seq_len, hidden_size)
        inputs = np.concatenate(all_inputs, axis=0)
        outputs = np.concatenate(all_outputs, axis=0)
        
        # Transpose to (n_units, n_timesteps)
        hidden_states = hidden_states.T
        
        return hidden_states, inputs, outputs
    
    def _compute_accuracy(self, outputs, targets):
        """Compute task-specific accuracy. Override in subclass if needed."""
        # Default: mean squared error converted to correlation-based accuracy
        mse = torch.mean((outputs - targets) ** 2).item()
        # Normalize by target variance
        target_var = torch.var(targets).item()
        r2 = 1 - (mse / (target_var + 1e-10))
        return max(0, r2)  # R^2 as accuracy proxy
    
    def get_expected_dynamics(self):
        """Return description of expected dynamics. Override in subclass."""
        return "Unknown dynamics"


class FlipFlopTask(TaskBase):
    """
    N-bit memory register task (set-reset flip-flop).
    
    The network must maintain memory of N binary values. Each input channel
    can send a SET signal (+1 or -1) to directly set that bit, or no signal (0)
    to leave it unchanged. The network outputs the current state of all bits.
    
    Note: This implements SET semantics (input directly sets state), not TOGGLE.
    This is equivalent to an SR (set-reset) flip-flop in digital electronics.
    
    Expected dynamics: Contraction-dominant (stable attractors for each state)
    Expected unit types: Integrators >> Rotators (memory maintenance)
    
    Args:
        n_bits: Number of bits to store (default 3)
        flip_prob: Probability of set signal per bit per timestep (default 0.05)
    """
    
    def __init__(self, n_bits=3, flip_prob=0.05):
        super().__init__("MemoryRegister", n_bits, n_bits)
        self.n_bits = n_bits
        self.flip_prob = flip_prob
    
    def generate_trial(self, length=200, batch_size=32):
        """
        Generate memory register sequences.
        
        Returns:
            inputs: (batch, length, n_bits) - set signals (+1, -1, or 0)
            targets: (batch, length, n_bits) - current bit states (+1 or -1)
        """
        inputs = torch.zeros(batch_size, length, self.n_bits)
        targets = torch.zeros(batch_size, length, self.n_bits)
        
        for b in range(batch_size):
            # Initialize random state
            state = torch.randint(0, 2, (self.n_bits,)) * 2.0 - 1.0  # Random +1/-1
            
            for t in range(length):
                # Random updates: flip_prob chance to set each bit
                for bit in range(self.n_bits):
                    if np.random.rand() < self.flip_prob:
                        # Randomly set to +1 or -1
                        new_value = np.random.choice([1.0, -1.0])
                        inputs[b, t, bit] = new_value
                        state[bit] = new_value  # SET bit to input value
                    # else: input remains 0, state unchanged
                
                # Output current state
                targets[b, t] = state.clone()
        
        return inputs, targets
    
    def _compute_accuracy(self, outputs, targets):
        """Accuracy: fraction of correctly classified bits."""
        # Sign of output should match target
        correct = (torch.sign(outputs) == torch.sign(targets)).float()
        return torch.mean(correct).item()
    
    def get_expected_dynamics(self):
        return ("Contraction-dominant: 2^n_bits stable attractors\n"
                f"Expected: {2**self.n_bits} attractor states\n"
                "Unit distribution: Integrators (60-80%), Rotators (10-20%), "
                "Explorers (5-15%)")


class CyclingMemoryTask(TaskBase):
    """
    Cycling memory task with periodic dynamics.
    
    The network receives a sequence of n_patterns random patterns at the start,
    then must continuously cycle through them in order when prompted.
    Input channels: pattern presentation + cycle trigger
    
    Expected dynamics: Rotation-dominant (periodic cycling through states)
    Expected unit types: Rotators >> Integrators (oscillatory dynamics)
    
    Args:
        n_patterns: Number of patterns in cycle (default 4)
        pattern_dim: Dimensionality of each pattern (default 3)
        cycle_length: Timesteps per pattern during cycling (default 20)
    """
    
    def __init__(self, n_patterns=4, pattern_dim=3, cycle_length=20):
        super().__init__("CyclingMemory", pattern_dim + 1, pattern_dim)
        self.n_patterns = n_patterns
        self.pattern_dim = pattern_dim
        self.cycle_length = cycle_length
        
        # Generate fixed patterns (so network can learn them)
        torch.manual_seed(42)  # Reproducible patterns
        self.patterns = torch.randn(n_patterns, pattern_dim)
        self.patterns = self.patterns / torch.norm(self.patterns, dim=1, keepdim=True)
        torch.manual_seed(torch.seed())  # Reset to random
    
    def generate_trial(self, length=200, batch_size=32):
        """
        Generate cycling memory sequences.
        
        Returns:
            inputs: (batch, length, pattern_dim + 1)
                    [:, :, :pattern_dim] = pattern input
                    [:, :, pattern_dim] = cycle trigger
            targets: (batch, length, pattern_dim) - current pattern in cycle
        """
        inputs = torch.zeros(batch_size, length, self.pattern_dim + 1)
        targets = torch.zeros(batch_size, length, self.pattern_dim)
        
        # Use fixed patterns (initialized in __init__)
        patterns = self.patterns
        
        for b in range(batch_size):
            # Presentation phase (first n_patterns * 15 timesteps) - longer to help learning
            presentation_length = self.n_patterns * 15
            for i in range(self.n_patterns):
                t_start = i * 15
                t_end = (i + 1) * 15
                if t_end <= length:
                    inputs[b, t_start:t_end, :self.pattern_dim] = patterns[i]
                    targets[b, t_start:t_end] = patterns[i]
            
            # Cycling phase (remaining timesteps)
            if presentation_length < length:
                inputs[b, presentation_length:, self.pattern_dim] = 1.0  # Cycle trigger
                
                for t in range(presentation_length, length):
                    # Which pattern to output (cycles through)
                    pattern_idx = ((t - presentation_length) // self.cycle_length) % self.n_patterns
                    targets[b, t] = patterns[pattern_idx]
        
        return inputs, targets
    
    def get_expected_dynamics(self):
        return ("Rotation-dominant: Periodic cycling through n_patterns states\n"
                f"Expected: {self.n_patterns}-state limit cycle\n"
                "Unit distribution: Rotators (50-70%), Integrators (20-30%), "
                "Explorers (10-20%)")


class ContextIntegrationTask(TaskBase):
    """
    Context-dependent evidence integration task.
    
    The network receives a context cue (determines which input channel to integrate)
    followed by noisy evidence. It must integrate evidence from the cued channel
    and output a binary decision once a decision threshold is crossed.
    
    Expected dynamics: Mixed (integration + switching)
    Expected unit types: Balanced distribution with transition dynamics
    
    Args:
        n_contexts: Number of context options (default 2)
        evidence_noise: Noise level in evidence (default 0.3)
        decision_threshold: Threshold for decision (default 3.0)
    """
    
    def __init__(self, n_contexts=2, evidence_noise=0.3, decision_threshold=3.0):
        super().__init__("ContextIntegration", 
                        n_contexts + n_contexts,  # context + evidence channels
                        1)  # binary decision output
        self.n_contexts = n_contexts
        self.evidence_noise = evidence_noise
        self.decision_threshold = decision_threshold
    
    def generate_trial(self, length=150, batch_size=32):
        """
        Generate context-dependent integration sequences.
        
        Returns:
            inputs: (batch, length, 2*n_contexts)
                    [:, :, :n_contexts] = context cue (one-hot at start)
                    [:, :, n_contexts:] = evidence channels
            targets: (batch, length, 1) - accumulated evidence / decision
        """
        inputs = torch.zeros(batch_size, length, 2 * self.n_contexts)
        targets = torch.zeros(batch_size, length, 1)
        
        context_duration = 20
        
        for b in range(batch_size):
            # Random context
            context_idx = np.random.randint(0, self.n_contexts)
            
            # Context cue period
            inputs[b, :context_duration, context_idx] = 1.0
            
            # Evidence integration period
            evidence_sign = np.random.choice([-1, 1])
            accumulated = 0.0
            
            for t in range(context_duration, length):
                # Noisy evidence on the cued channel
                evidence = evidence_sign * 0.1 + np.random.randn() * self.evidence_noise
                inputs[b, t, self.n_contexts + context_idx] = evidence
                
                # Accumulate
                accumulated += evidence
                
                # Output: accumulated evidence (saturated at decision)
                targets[b, t, 0] = np.tanh(accumulated / self.decision_threshold)
        
        return inputs, targets
    
    def get_expected_dynamics(self):
        return ("Mixed dynamics: Contraction during integration, expansion at context switch\n"
                "Expected: Integration ramp + context-gated transitions\n"
                "Unit distribution: Integrators (40-50%), Rotators (20-30%), "
                "Explorers (20-30%)")


class SequentialMNISTTask(TaskBase):
    """
    Sequential MNIST digit classification task.
    
    Images are presented as 784-step sequences (one pixel per timestep, row-major order).
    The network must integrate visual information over time and output the digit class
    at the final timestep. This task requires temporal integration and pattern recognition.
    
    Expected dynamics: Contraction-dominant (evidence accumulation) + expansion (decision)
    Expected unit types: Integrators (50-60%), Explorers (20-30%), Rotators (10-20%)
    
    Args:
        data_root: Directory to store MNIST data (default: './data')
        train: Use training set (True) or test set (False)
        normalize: Whether to normalize pixel values to [-1, 1] (default: True)
    """
    
    def __init__(self, data_root='./data', train=True, normalize=True):
        super().__init__("SequentialMNIST", input_size=1, output_size=10)
        self.data_root = data_root
        self.train = train
        self.normalize = normalize
        
        # Import torchvision (already in requirements.txt)
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision required for MNIST task. Install with: pip install torchvision")
        
        # Load MNIST dataset
        transform_list = [transforms.ToTensor()]
        if normalize:
            # Normalize to [-1, 1] instead of [0, 1]
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        transform = transforms.Compose(transform_list)
        
        self.dataset = datasets.MNIST(
            root=data_root,
            train=train,
            transform=transform,
            download=True
        )
        
        # Pre-compute sequence length (28x28 = 784 pixels)
        self.seq_length = 28 * 28
        
        print(f"Loaded MNIST {'training' if train else 'test'} set: {len(self.dataset)} images")
    
    def generate_trial(self, length=None, batch_size=32):
        """
        Generate sequential MNIST trials.
        
        Args:
            length: Sequence length (ignored, always 784 for MNIST)
            batch_size: Number of images in batch
        
        Returns:
            inputs: (batch, 784, 1) - pixel values presented sequentially
            targets: (batch, 784, 10) - one-hot labels (only final timestep is non-zero)
        """
        # Override length to MNIST dimensions
        length = self.seq_length
        
        # Sample random batch from dataset
        indices = torch.randint(0, len(self.dataset), (batch_size,))
        
        inputs = torch.zeros(batch_size, length, 1)
        targets = torch.zeros(batch_size, length, 10)
        
        for i, idx in enumerate(indices):
            image, label = self.dataset[idx]
            
            # Flatten image to sequence (28x28 -> 784)
            pixel_sequence = image.view(-1)  # Shape: (784,)
            
            # Assign to inputs (one pixel per timestep)
            inputs[i, :, 0] = pixel_sequence
            
            # One-hot encode label - only at final timestep
            # This ensures loss is only computed on the classification decision
            targets[i, -1, label] = 1.0  # Only final timestep has target
        
        return inputs, targets
    
    def _compute_accuracy(self, outputs, targets):
        """
        Classification accuracy on final timestep.
        
        Args:
            outputs: (batch, seq_len, 10) network outputs
            targets: (batch, seq_len, 10) one-hot labels
        
        Returns:
            accuracy: Fraction of correctly classified digits
        """
        # Use final timestep for classification
        final_outputs = outputs[:, -1, :]  # (batch, 10)
        final_targets = targets[:, -1, :]  # (batch, 10)
        
        # Get predicted class (argmax)
        predictions = torch.argmax(final_outputs, dim=1)
        true_labels = torch.argmax(final_targets, dim=1)
        
        # Compute accuracy
        correct = (predictions == true_labels).float()
        accuracy = torch.mean(correct).item()
        
        return accuracy
    
    def get_expected_dynamics(self):
        return ("Contraction-dominant: Evidence accumulation over 784 timesteps\n"
                "Expected: Visual feature integration + classification decision\n"
                "Unit distribution: Integrators (50-60%), Explorers (20-30%), "
                "Rotators (10-20%)")


def get_task(task_name, **kwargs):
    """
    Factory function to get task by name.
    
    Args:
        task_name: 'flipflop', 'cycling', 'context', or 'mnist'
        **kwargs: Task-specific parameters
    
    Returns:
        Task instance
    """
    task_map = {
        'flipflop': FlipFlopTask,
        'cycling': CyclingMemoryTask,
        'context': ContextIntegrationTask,
        'mnist': SequentialMNISTTask
    }
    
    task_name_lower = task_name.lower()
    if task_name_lower not in task_map:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(task_map.keys())}")
    
    return task_map[task_name_lower](**kwargs)
