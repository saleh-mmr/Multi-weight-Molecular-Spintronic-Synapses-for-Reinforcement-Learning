"""Base crosspoint primitive with shared state and stochastic noise update logic."""

from .crosspointParams import CrossPointParams
from .crosspointState import CrosspointState
import numpy as np


class BaseCrosspoint:
    """
    Abstract base class for physical crosspoint devices.
    
    Provides:
      - Gaussian noise generation (can be zero if noise_stddev <= 0)
      - State update protocol: increment index and resample noise
    
    Subclasses (Magnetoresistive, NonMagnetoresistive) define specific conductance functions.
    """

    def __init__(self, params: CrossPointParams, state: CrosspointState):
        """
        Initialize a crosspoint with given parameters and mutable state.
        
        Args:
            params (CrossPointParams): device physics (conductance coefficients, noise)
            state (CrosspointState): mutable state (index, noise)
        """
        self.params = params
        self.state = state

    def redraw_noise(self, sigma):
        """
        Sample one new Gaussian noise value.
        
        Reflects the stochastic nature of physical device behavior:
        noise is multiplicative (1 + η) on conductance due to variability in
        materials, temperature, and other environmental factors.
        
        Args:
            sigma (float): standard deviation of Gaussian (if σ ≤ 0, returns 0)
        
        Returns:
            float: sampled noise value ∼ N(0, σ)
        """
        noise = float(np.random.normal(0.0, sigma)) if sigma > 0 else 0.0
        return noise

    def update_state(self):
        """
        Perform one device update: increment index and resample noise.
        
        Called after a gradient update to reflect physical device behavior
        (resistance change due to ion migration + thermal noise).
        """
        self.state.increment_index()
        sigma = self.params.get_noise_stddev()
        noise = self.redraw_noise(sigma)
        self.state.update_noise(noise)