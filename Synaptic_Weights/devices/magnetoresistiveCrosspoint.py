"""Magnetoresistive positive crosspoint model with P/AP conductance branches."""

import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState


class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Models a magnetoresistive crosspoint with two device states:
      - Parallel (P): lower resistance (higher conductance)
      - Anti-parallel (AP): higher resistance (lower conductance)
    
    Used for positive synaptic weights that determine the multi-weight synapse output.
    The device can be driven to either state, and conductance depends on the current state.
    """

    def conductance_p(self, state: CrosspointState) -> float:
        """
        Compute parallel-state conductance: G_p = g_p * log10(x + shift) * (1 + noise).
        
        Lower conductance branch (when device is in parallel state).
        The shift parameter prevents log10 of zero and provides stability.
        Multiplicative noise (1 + η) reflects realistic device variability.
        
        Args:
            state (CrosspointState): current crosspoint state
        
        Returns:
            float: parallel conductance value
        """
        g_p_coefficient = self.params.g_p_coefficient
        shift_parameter = self.params.shift_parameter
        index, noise = state.get_state()
        return float((g_p_coefficient * np.log10(index + shift_parameter)) * (1 + noise))

    def conductance_ap(self, state: CrosspointState) -> float:
        """
        Compute anti-parallel-state conductance: G_ap = g_ap * log10(x) * (1 + noise).
        
        Higher conductance branch (when device is in anti-parallel state).
        This is the dominant branch for positive weights in multi-weight synapses.
        
        Args:
            state (CrosspointState): current crosspoint state
        
        Returns:
            float: anti-parallel conductance value
        """
        g_ap_coefficient = self.params.g_ap_coefficient
        index, noise = state.get_state()
        return float((g_ap_coefficient * np.log10(index)) * (1 + noise))
