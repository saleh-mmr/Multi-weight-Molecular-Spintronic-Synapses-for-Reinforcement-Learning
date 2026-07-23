"""Vectorized synaptic controller for efficient multi-weight conductance simulation."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


class SynapticWeightController:
    """
    High-performance vectorized replacement for SynapticWeightController (non-optimized).
    
    Key improvements:
      1. Tensor-based state storage (all crosspoint indices/noise in one tensor per parameter)
      2. Batch updates using PyTorch vectorized operations (no Python loops)
      3. Weight caching (only recompute if dirty flag or ap_index changed)
    
    State storage (per parameter):
      - bias_x, bias_noise: current index and noise of bias crosspoint
      - positive_x[i], positive_noise[i]: index and noise of i-th AP-selectable crosspoint
    
    The three AP-selectable crosspoints (n_problem=3) support multi-environment training
    where each environment switches the active AP state.
    """

    def __init__(self, model, g_ap, g_p, shift_parameter, g_bias, noise_stddev):
        """
        Initialize vectorized controller with neural network and device parameters.
        
        Creates per-parameter tensors for storing crosspoint indices and noise values.
        All tensors are allocated on the same device and dtype as model parameters.
        
        Args:
            model: PyTorch neural network
            g_ap (float): AP-state conductance coefficient
            g_p (float): P-state conductance coefficient
            shift_parameter (float): log10 offset
            g_bias (float): bias conductance coefficient
            noise_stddev (float): Gaussian noise std dev
        """
        self.model = model

        # Store conductance parameters for weight computation
        self.g_ap = float(g_ap)
        self.g_p = float(g_p)
        self.shift_parameter = float(shift_parameter)
        self.g_bias = float(g_bias)
        self.noise_stddev = float(noise_stddev)

        # Fixed: 3 AP-selectable crosspoints per parameter
        self.n_problem = 3
        self.scaling_factor = 1.0

        # Per-parameter state tensors
        # bias_x[name]: tensor with same shape as parameter, stores bias crosspoint indices
        # bias_noise[name]: tensor with same shape, stores bias crosspoint noise
        # positive_x[name]: tensor with shape (n_problem, *param.shape), stores AP-indices
        # positive_noise[name]: tensor with shape (n_problem, *param.shape), stores AP-noise
        self.bias_x = {}
        self.bias_noise = {}
        self.positive_x = {}
        self.positive_noise = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            dtype = param.dtype
            shape = param.shape

            # Compute equilibrium index (same as CrosspointState logic)
            initial_x = self._initial_index(device=device, dtype=dtype)

            # Allocate and initialize bias state tensors
            self.bias_x[name] = torch.full(
                shape,
                initial_x,
                dtype=dtype,
                device=device,
            )

            self.bias_noise[name] = self._draw_noise(
                shape,
                device=device,
                dtype=dtype,
            )

            # Allocate and initialize positive state tensors (one per AP-selectable branch)
            self.positive_x[name] = torch.full(
                (self.n_problem, *shape),
                initial_x,
                dtype=dtype,
                device=device,
            )

            self.positive_noise[name] = self._draw_noise(
                (self.n_problem, *shape),
                device=device,
                dtype=dtype,
            )

        # Caching: skip recomputation if ap_index unchanged and weights not updated
        self.current_loaded_ap_index = None
        self.weights_dirty = True

    def _initial_index(self, device, dtype):
        """
        Compute equilibrium crosspoint index.
        
        Solves for i such that G_ap(i) ≈ G_p(i):
            i^k ≈ i + shift_parameter
        where k = g_ap / g_p.
        
        Returns initial_index as a scalar tensor on the specified device.
        
        Args:
            device: torch device (CPU or GPU)
            dtype: torch dtype (float32 or float64)
        
        Returns:
            float: scalar equilibrium index
        """
        k = self.g_ap / self.g_p
        i = 1
        # Binary search could be faster, but scalar loop is fine for initialization
        while i ** k <= i + self.shift_parameter:
            i += 1
        return float(i)

    def _draw_noise(self, shape, device, dtype):
        """
        Draw a batch of Gaussian noise samples.
        
        If noise_stddev > 0: return N(0, noise_stddev) samples
        If noise_stddev <= 0: return zeros (no stochasticity)
        
        Vectorized equivalent of BaseCrosspoint.redraw_noise() for all crosspoints.
        
        Args:
            shape: tensor shape to fill
            device: torch device
            dtype: torch dtype
        
        Returns:
            torch.Tensor: noise samples with given shape
        """
        if self.noise_stddev > 0:
            return torch.normal(
                mean=0.0,
                std=self.noise_stddev,
                size=shape,
                device=device,
                dtype=dtype,
            )

        return torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, ap_index):
        """
        Update all crosspoint indices based on gradient signs (vectorized).
        
        For each parameter with gradient:
          1. Identify elements with positive gradient → increment bias index
          2. Identify elements with negative gradient → increment AP index at ap_index
        
        Non-finite (NaN/Inf) gradients are masked out.
        
        Single pass over model parameters with no loops over individual weights.
        Sets weights_dirty=True to trigger recomputation on next load_weights().
        
        Args:
            ap_index (int): which AP-selectable crosspoint (0-2) to update for negatives
        """
        assert 0 <= ap_index < self.n_problem

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            # Mask to identify elements that should be updated
            valid = torch.isfinite(grad)
            # Positive gradient: increase bias (reduce weight)
            pos = (grad > 0) & valid
            # Negative gradient: increase AP crosspoint (increase weight)
            neg = (grad < 0) & valid

            # Vectorized update: increment indices where condition is true
            # This is equivalent to a loop: for each True element, increment by 1.0
            if pos.any():
                self.bias_x[name][pos] += 1.0
                # Resample noise for updated elements
                self.bias_noise[name][pos] = self._draw_noise(
                    self.bias_noise[name][pos].shape,
                    device=param.device,
                    dtype=param.dtype,
                )

            # Increment AP crosspoint at ap_index for negative gradients
            if neg.any():
                self.positive_x[name][ap_index][neg] += 1.0
                self.positive_noise[name][ap_index][neg] = self._draw_noise(
                    self.positive_noise[name][ap_index][neg].shape,
                    device=param.device,
                    dtype=param.dtype,
                )

        self.weights_dirty = True

    @torch.no_grad()
    def load_weights(self, ap_index):
        """
        Compute and load all network parameters from physical device state.
        
        Vectorized weight computation:
        
        For each parameter:
          1. Compute bias conductance:  G_bias = g_bias * log10(x_bias) * (1 + noise_bias)
          2. For each AP-selectable crosspoint i:
             - If i == ap_index: G_ap = g_ap * log10(x_ap) * (1 + noise_ap)
             - Else: G_p = g_p * log10(x_p + shift) * (1 + noise_p)
          3. Sum all positive conductances: G_total = Σ G_i
          4. Final weight: w = scaling_factor * (G_total - G_bias)
        
        Caching optimization:
          - If ap_index unchanged and weights_dirty=False, skip computation
          - This avoids redundant weight recalculation within a learning episode
        
        Args:
            ap_index (int): which AP-selectable crosspoint (0-2) is "on"
        """
        assert 0 <= ap_index < self.n_problem

        # Skip recomputation if nothing changed
        if self.current_loaded_ap_index == ap_index and not self.weights_dirty:
            return

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Compute bias conductance for all elements in parallel
            x_bias = self.bias_x[name]
            noise_bias = self.bias_noise[name]
            g_bias = (self.g_bias * torch.log10(x_bias)) * (1.0 + noise_bias)

            # Compute total positive conductance (sum across n_problem branches)
            g_total = torch.zeros_like(param)

            for problem_index in range(self.n_problem):
                x_pos = self.positive_x[name][problem_index]
                noise_pos = self.positive_noise[name][problem_index]

                if problem_index == ap_index:
                    # AP-state branch (higher conductance for selected problem)
                    g = (self.g_ap * torch.log10(x_pos)) * (1.0 + noise_pos)
                else:
                    # P-state branch (lower conductance for non-selected problems)
                    g = (self.g_p * torch.log10(x_pos + self.shift_parameter)) * (1.0 + noise_pos)

                g_total += g

            # Compute final synaptic weight
            weight = self.scaling_factor * (g_total - g_bias)

            # Update model parameter in-place
            param.copy_(weight)

        self.current_loaded_ap_index = ap_index
        self.weights_dirty = False