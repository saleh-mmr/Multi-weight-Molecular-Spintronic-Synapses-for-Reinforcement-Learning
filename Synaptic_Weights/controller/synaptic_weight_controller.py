"""Object-based synaptic controller mapping gradients to crosspoint state updates."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from devices.multiWeightSynapse import MultiWeightSynapse, MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams


class SynapticWeightController:
    """
    Maps neural network parameters to physical multi-weight synapses.
    
    Each network weight/bias is paired with a MultiWeightSynapse object containing
    multiple crosspoint devices. Learning updates are translated into crosspoint
    index increments based on gradient sign.
    
    Update rule (sign-driven learning):
      - Positive gradient → increment bias crosspoint (decrease weight)
      - Negative gradient → increment positive crosspoint in AP state (increase weight)
    
    This object-oriented implementation maintains per-parameter synapse objects.
    For performance-critical applications, use synaptic_weight_controller_optimize.py.
    """

    def __init__(self, model, g_ap, g_p, shift_parameter, g_bias, noise_stddev):
        """
        Initialize controller with neural network and physical device parameters.
        
        Maps each network parameter (weight matrix or bias vector) to a corresponding
        MultiWeightSynapse. The n_problem=3 hardcodes three AP-selectable crosspoints
        per parameter (matching the three CartPole environments in training).
        
        Args:
            model: PyTorch neural network with named_parameters()
            g_ap (float): AP-state conductance coefficient
            g_p (float): P-state conductance coefficient
            shift_parameter (float): log10 offset to prevent log(0)
            g_bias (float): bias conductance coefficient
            noise_stddev (float): Gaussian noise standard deviation
        """
        self.model = model

        # Create physical device parameters shared across all synapses
        params = CrossPointParams(
            g_ap_coefficient=g_ap,
            g_p_coefficient=g_p,
            shift_parameter=shift_parameter,
            g_bias_coefficient=g_bias,
            noise_stddev=noise_stddev
        )
        # Specify multi-weight structure: 3 AP-selectable crosspoints per parameter
        spec = MultiWeightSynapseSpec(n_problem=3, scaling_factor=1)

        # Create synapse objects for each network parameter
        self.synapses = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            shape = param.shape
            # Weight matrices (2D): create 2D array of MultiWeightSynapse objects
            if len(shape) == 2:
                self.synapses[name] = [
                    [MultiWeightSynapse(spec, params) for _ in range(shape[1])]
                    for _ in range(shape[0])
                ]
            # Bias vectors (1D): create 1D array of MultiWeightSynapse objects
            elif len(shape) == 1:
                self.synapses[name] = [
                    MultiWeightSynapse(spec, params) for _ in range(shape[0])
                ]

    @torch.no_grad()
    def step(self, ap_index):
        """
        Update all crosspoint indices based on gradient signs.
        
        For each parameter with a gradient:
          1. Identify positive gradient elements → increase bias crosspoint
          2. Identify negative gradient elements → increase positive crosspoint at ap_index
        
        Non-finite (NaN/Inf) gradients are skipped. This implements discrete,
        sign-based learning where gradient magnitude is ignored (only sign matters).
        
        Args:
            ap_index (int): which AP-selectable crosspoint (0-2) is "on" in multi-weight
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            # Mask out non-finite gradients (NaN, Inf)
            valid = torch.isfinite(grad)
            # Positive gradient: increase bias (reduce weight)
            pos = (grad > 0) & valid
            # Negative gradient: increase AP crosspoint (increase weight)
            neg = (grad < 0) & valid

            # Update 2D weight matrices (shape: [out_features, in_features])
            if grad.ndim == 2:
                if pos.any():
                    for i in range(pos.shape[0]):
                        for j in range(pos.shape[1]):
                            if pos[i, j]:
                                self.synapses[name][i][j].increase_bias_crosspoint_index()
                if neg.any():
                    for i in range(neg.shape[0]):
                        for j in range(neg.shape[1]):
                            if neg[i, j]:
                                self.synapses[name][i][j].increase_positive_crosspoint_index(ap_index)
            # Update 1D bias vectors (shape: [out_features])
            elif grad.ndim == 1:
                if pos.any():
                    for i in range(pos.shape[0]):
                        if pos[i]:
                            self.synapses[name][i].increase_bias_crosspoint_index()
                if neg.any():
                    for i in range(neg.shape[0]):
                        if neg[i]:
                            self.synapses[name][i].increase_positive_crosspoint_index(ap_index)

    @torch.no_grad()
    def load_weights(self, ap_index):
        """
        Compute and load all network parameters from physical device states.
        
        For each synapse object, computes the weight as:
            weight = scaling_factor * (sum_of_conductances - bias_conductance)
        
        where which positive crosspoint is in AP state is determined by ap_index.
        This creates a parameterized family of weights indexed by ap_index (0-2),
        emulating environmental switching or selector signals in hardware.
        
        Args:
            ap_index (int): which AP-selectable crosspoint is active (0-2)
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip if no gradient (shouldn't happen after step(), but safe)
            grad = param.grad
            if grad is None:
                continue

            st = self.synapses[name]
            # Load weights for 2D matrices
            if param.ndim == 2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        param[i, j].copy_(
                            torch.tensor(st[i][j].weight(ap_index), dtype=param.dtype)
                        )
            # Load weights for 1D bias vectors
            elif param.ndim == 1:
                for i in range(param.shape[0]):
                    param[i].copy_(
                        torch.tensor(st[i].weight(ap_index), dtype=param.dtype)
                    )