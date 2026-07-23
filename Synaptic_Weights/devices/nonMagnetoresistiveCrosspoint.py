"""Bias crosspoint model that uses the non-magnetoresistive conductance branch."""

import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState


class NonMagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Models the bias (negative) conductance branch of the multi-weight synapse.
    
    Unlike the magnetoresistive crosspoint with P/AP states, this uses a single
    conductance formula without branching. It provides the bias term that is
    subtracted from the sum of positive conductances:
    
        weight = scaling_factor * (Σ G_positive - G_bias)
    """

    def conductance_p(self, state: CrosspointState) -> float:
        """
        Compute the bias-branch conductance: G_bias = g_bias * log10(x) * (1 + noise).
        
        Single branch (no P/AP switching like magnetoresistive).
        Subtracted from positive conductances to realize the synaptic weight.
        
        Args:
            state (CrosspointState): current crosspoint state
        
        Returns:
            float: bias conductance value
        """
        g_bias_coefficient = self.params.g_bias_coefficient
        index, noise = state.get_state()
        return float((g_bias_coefficient * np.log10(index)) * (1 + noise))