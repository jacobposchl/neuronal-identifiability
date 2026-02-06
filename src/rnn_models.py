"""
RNN architectures for deformation-based unit classification.

Provides vanilla RNN, LSTM, and GRU implementations with exposed hidden states
for Jacobian computation and deformation analysis.
"""

import torch
import torch.nn as nn
import numpy as np


class VanillaRNN(nn.Module):
    """
    Simple vanilla RNN with exposed hidden states.
    
    Architecture:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_ho @ h_t + b_o
    
    Args:
        input_size: Dimension of input features
        hidden_size: Number of hidden units
        output_size: Dimension of output
        nonlinearity: 'tanh' or 'relu'
        bias: Whether to use bias terms
    """
    
    def __init__(self, input_size, hidden_size, output_size, 
                 nonlinearity='tanh', bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, 
                         nonlinearity=nonlinearity, bias=bias)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size, bias=bias)
        
        # Store nonlinearity for single-step computation
        self.nonlinearity = torch.tanh if nonlinearity == 'tanh' else torch.relu
    
    def forward(self, x, return_hidden_states=True):
        """
        Forward pass through RNN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            return_hidden_states: Whether to return all hidden states
        
        Returns:
            outputs: (batch, seq_len, output_size)
            hidden_states: (batch, seq_len, hidden_size) if return_hidden_states else None
        """
        # Get hidden states from RNN
        hidden_states, _ = self.rnn(x)  # (batch, seq_len, hidden_size)
        
        # Compute outputs
        outputs = self.fc(hidden_states)  # (batch, seq_len, output_size)
        
        if return_hidden_states:
            return outputs, hidden_states
        return outputs
    
    def step(self, x, h):
        """
        Single timestep for Jacobian computation.
        
        Args:
            x: Input at time t, shape (batch, input_size) or (input_size,)
            h: Hidden state at time t-1, shape (batch, hidden_size) or (hidden_size,)
        
        Returns:
            h_next: Hidden state at time t, same shape as h
        """
        # Handle both batched and single inputs
        single_input = (x.dim() == 1)
        if single_input:
            x = x.unsqueeze(0)
            h = h.unsqueeze(0)
        
        # Get RNN parameters
        W_ih = self.rnn.weight_ih_l0  # (hidden_size, input_size)
        W_hh = self.rnn.weight_hh_l0  # (hidden_size, hidden_size)
        b_h = self.rnn.bias_ih_l0 + self.rnn.bias_hh_l0  # Combined bias
        
        # Compute next hidden state
        h_next = self.nonlinearity(x @ W_ih.t() + h @ W_hh.t() + b_h)
        
        if single_input:
            h_next = h_next.squeeze(0)
        
        return h_next
    
    def get_initial_hidden(self, batch_size=1):
        """Get zero initial hidden state."""
        return torch.zeros(batch_size, self.hidden_size)


class SimpleLSTM(nn.Module):
    """
    LSTM with exposed hidden and cell states.
    
    Architecture:
        i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
        f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
        g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
        o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
        y_t = W_ho @ h_t + b_o
    
    Args:
        input_size: Dimension of input features
        hidden_size: Number of hidden units
        output_size: Dimension of output
        bias: Whether to use bias terms
    """
    
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bias=bias)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size, bias=bias)
    
    def forward(self, x, return_hidden_states=True):
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            return_hidden_states: Whether to return all hidden states
        
        Returns:
            outputs: (batch, seq_len, output_size)
            hidden_states: (batch, seq_len, hidden_size) if return_hidden_states else None
        """
        # Get hidden states from LSTM
        hidden_states, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Compute outputs
        outputs = self.fc(hidden_states)  # (batch, seq_len, output_size)
        
        if return_hidden_states:
            return outputs, hidden_states
        return outputs
    
    def step(self, x, h, c):
        """
        Single timestep for Jacobian computation.
        
        Args:
            x: Input at time t, shape (batch, input_size) or (input_size,)
            h: Hidden state at time t-1, shape (batch, hidden_size) or (hidden_size,)
            c: Cell state at time t-1, shape (batch, hidden_size) or (hidden_size,)
        
        Returns:
            h_next: Hidden state at time t
            c_next: Cell state at time t
        """
        # Handle both batched and single inputs
        single_input = (x.dim() == 1)
        if single_input:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
            h = h.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
            c = c.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        else:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
            h = h.unsqueeze(0)  # (1, batch, hidden_size)
            c = c.unsqueeze(0)  # (1, batch, hidden_size)
        
        # Forward through LSTM
        _, (h_next, c_next) = self.lstm(x, (h, c))
        
        # Remove extra dimensions
        h_next = h_next.squeeze(0)
        c_next = c_next.squeeze(0)
        
        if single_input:
            h_next = h_next.squeeze(0)
            c_next = c_next.squeeze(0)
        
        return h_next, c_next
    
    def get_initial_hidden(self, batch_size=1):
        """Get zero initial hidden and cell states."""
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)
        return h, c


class SimpleGRU(nn.Module):
    """
    GRU with exposed hidden states.
    
    Architecture:
        r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
        z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
        n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1} + b_hn) + b_in)
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        y_t = W_ho @ h_t + b_o
    
    Args:
        input_size: Dimension of input features
        hidden_size: Number of hidden units
        output_size: Dimension of output
        bias: Whether to use bias terms
    """
    
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bias=bias)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size, bias=bias)
    
    def forward(self, x, return_hidden_states=True):
        """
        Forward pass through GRU.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            return_hidden_states: Whether to return all hidden states
        
        Returns:
            outputs: (batch, seq_len, output_size)
            hidden_states: (batch, seq_len, hidden_size) if return_hidden_states else None
        """
        # Get hidden states from GRU
        hidden_states, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        
        # Compute outputs
        outputs = self.fc(hidden_states)  # (batch, seq_len, output_size)
        
        if return_hidden_states:
            return outputs, hidden_states
        return outputs
    
    def step(self, x, h):
        """
        Single timestep for Jacobian computation.
        
        Args:
            x: Input at time t, shape (batch, input_size) or (input_size,)
            h: Hidden state at time t-1, shape (batch, hidden_size) or (hidden_size,)
        
        Returns:
            h_next: Hidden state at time t, same shape as h
        """
        # Handle both batched and single inputs
        single_input = (x.dim() == 1)
        if single_input:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
            h = h.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        else:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
            h = h.unsqueeze(0)  # (1, batch, hidden_size)
        
        # Forward through GRU
        _, h_next = self.gru(x, h)
        
        # Remove extra dimensions
        h_next = h_next.squeeze(0)
        
        if single_input:
            h_next = h_next.squeeze(0)
        
        return h_next
    
    def get_initial_hidden(self, batch_size=1):
        """Get zero initial hidden state."""
        return torch.zeros(batch_size, self.hidden_size)


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model, method='xavier'):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        method: 'xavier', 'kaiming', or 'orthogonal'
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_uniform_(param)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif method == 'orthogonal':
                nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
