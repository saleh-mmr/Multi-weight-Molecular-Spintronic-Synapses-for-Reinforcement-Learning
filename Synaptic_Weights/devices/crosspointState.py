"""Mutable per-crosspoint state storing index and one sampled noise realization."""

import numpy as np


class CrosspointState:
    """
    Encapsulates the mutable state of a single physical crosspoint device.
    
    Stores:
      - x: discrete index of the crosspoint (incremented by gradient updates)
      - noise_realization: one sample from Gaussian distribution (fixed per step)
    
    The index x determines conductance via logarithmic functions:
      G_ap = g_ap * log10(x)
      G_p  = g_p * log10(x + shift_parameter)
    """

    def __init__(self, params):
        """
        Initialize crosspoint state with equilibrium index.
        
        The equilibrium index is derived from solving:
            i^k = i + shift_parameter
        where k = g_ap_coefficient / g_p_coefficient.
        
        This ensures a balanced starting point before any gradient updates.
        
        Args:
            params (CrossPointParams): physical parameters defining conductance mapping
        """
        # Solve for equilibrium index: find i such that i^k ≈ i + shift
        k = params.g_ap_coefficient / params.g_p_coefficient
        i = 1
        while i**k <= i + params.shift_parameter:
            i += 1
        self.x = i
        
        # Sample one noise realization for this crosspoint
        # (redrawn each time update_state() is called in BaseCrosspoint)
        self.noise_realization = float(np.random.normal(0.0, params.noise_stddev))

    def update_noise(self, noise):
        """
        Replace current noise sample with a new one.
        
        Args:
            noise (float): newly sampled noise value
        """
        self.noise_realization = noise

    def increment_index(self):
        """
        Increment the crosspoint index (one step up the conductance curve).
        
        Returns:
            int: the new index value after increment
        """
        self.x += 1
        return self.x

    def get_state(self):
        """
        Retrieve the complete state for conductance calculation.
        
        Returns:
            tuple: (index, noise_realization)
        """
        return self.x, self.noise_realization