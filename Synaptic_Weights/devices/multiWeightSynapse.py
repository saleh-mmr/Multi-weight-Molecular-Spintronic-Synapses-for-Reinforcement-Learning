"""Composes multiple physical crosspoints into one reusable multi-weight synapse."""

from .crosspointState import CrosspointState
from .magnetoresistiveCrosspoint import MagnetoresistiveCrosspoint
from .nonMagnetoresistiveCrosspoint import NonMagnetoresistiveCrosspoint


class MultiWeightSynapseSpec:
    """Configuration for a multi-weight synapse."""
    
    def __init__(self, n_problem, scaling_factor):
        """
        Initialize synapse specification.
        
        Args:
            n_problem (int): number of positive crosspoint branches
            scaling_factor (float): multiplier applied to the net conductance
        """
        self.n_problem = n_problem
        self.scaling_factor = scaling_factor


class MultiWeightSynapse:
    """
    Logically groups multiple physical crosspoints into one multi-weight synapse.
    
    Architecture:
      - One bias crosspoint (NonMagnetoresistive, always in conductance branch)
      - N positive crosspoints (Magnetoresistive, switchable between P and AP)
    
    Weight computation:
      1. Activate exactly one positive crosspoint in AP state (highest conductance)
      2. All other positive crosspoints in P state (lower conductance)
      3. Net weight = scaling_factor * (sum_of_P_and_one_AP - bias_conductance)
    
    This structure emulates a network where different environmental conditions
    or learning signals can drive which positive crosspoint is "active" (in AP state).
    """

    def __init__(self, multiweight_spec, crosspoint_params):
        """
        Initialize multi-weight synapse with physical devices.
        
        Creates:
          - One bias device (NonMagnetoresistive)
          - N positive devices (Magnetoresistive), each with independent state
        
        All states are initialized to equilibrium index by CrosspointState.__init__.
        
        Args:
            multiweight_spec (MultiWeightSynapseSpec): n_problem, scaling_factor
            crosspoint_params (CrossPointParams): physical device parameters
        """
        self.spec = multiweight_spec
        self.params = crosspoint_params
        
        # Initialize bias branch (fixed non-magnetoresistive crosspoint)
        self.bias_state = CrosspointState(self.params)
        self.bias_crosspoint = NonMagnetoresistiveCrosspoint(self.params, self.bias_state)

        # Initialize positive branches (switchable magnetoresistive crosspoints)
        self.positive_crosspoints_states = [CrosspointState(self.params) for _ in range(self.spec.n_problem)]
        self.positive_crosspoint = []
        for i in range(self.spec.n_problem):
            self.positive_crosspoint.append(
                MagnetoresistiveCrosspoint(self.params, self.positive_crosspoints_states[i])
            )

    def weight(self, ap_index):
        """
        Calculate synapse weight given which positive crosspoint is in AP state.
        
        Multi-state operation:
          - Device at ap_index: uses conductance_ap() (higher conductance)
          - All other devices: use conductance_p() (lower conductance)
          - Bias: always conductance_p() (subtracted from sum)
        
        Formula:
            weight = scaling_factor * (sum_conductances - bias_conductance)
        
        The ap_index effectively implements a weight multiplexer: by choosing
        which crosspoint is "on" (AP), the network selects different weight values
        without changing the physical devices themselves.
        
        Args:
            ap_index (int): which positive crosspoint (0 to n_problem-1) is in AP state
        
        Returns:
            float: computed synaptic weight
        """
        assert 0 <= ap_index < self.spec.n_problem, f"ap_index {ap_index} out of range [0, {self.spec.n_problem})"
        
        # Sum conductances: activate ap_index in AP state, others in P state
        g_total = 0
        for i in range(self.spec.n_problem):
            if i == ap_index:
                g_total += self.positive_crosspoint[i].conductance_ap(self.positive_crosspoints_states[i])
            else:
                g_total += self.positive_crosspoint[i].conductance_p(self.positive_crosspoints_states[i])
        
        # Subtract bias conductance
        g_bias = self.bias_crosspoint.conductance_p(self.bias_state)
        weight = self.spec.scaling_factor * (g_total - g_bias)
        return weight

    def increase_positive_crosspoint_index(self, index_positive_crosspoint):
        """
        Increment index of a positive crosspoint (physical device update operation).
        
        Simulates the effect of applying a write pulse to increase conductance.
        Increments index and resamples noise.
        
        Args:
            index_positive_crosspoint (int): which positive device to update
        """
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        self.positive_crosspoint[index_positive_crosspoint].update_state()

    def increase_bias_crosspoint_index(self):
        """
        Increment index of the bias crosspoint (physical device update operation).
        
        Simulates applying a write pulse to increase bias conductance.
        """
        self.bias_crosspoint.update_state()

    def get_positive_crosspoint_state(self, index_positive_crosspoint):
        """
        Retrieve state (index, noise) of a positive crosspoint.
        
        Args:
            index_positive_crosspoint (int): which positive device
        
        Returns:
            tuple: (index, noise_realization)
        """
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        return self.positive_crosspoint[index_positive_crosspoint].state.get_state()

    def get_bias_crosspoint_state(self):
        """
        Retrieve state (index, noise) of the bias crosspoint.
        
        Returns:
            tuple: (index, noise_realization)
        """
        return self.bias_crosspoint.state.get_state()

    def get_positive_crosspoint_conductance_p(self, index_positive_crosspoint):
        """
        Get the P-state (lower) conductance of a positive crosspoint.
        
        Args:
            index_positive_crosspoint (int): which positive device
        
        Returns:
            float: parallel conductance
        """
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        return self.positive_crosspoint[index_positive_crosspoint].conductance_p(
            self.positive_crosspoints_states[index_positive_crosspoint]
        )

    def get_positive_crosspoint_conductance_ap(self, index_positive_crosspoint):
        """
        Get the AP-state (higher) conductance of a positive crosspoint.
        
        Args:
            index_positive_crosspoint (int): which positive device
        
        Returns:
            float: anti-parallel conductance
        """
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        return self.positive_crosspoint[index_positive_crosspoint].conductance_ap(
            self.positive_crosspoints_states[index_positive_crosspoint]
        )

    def get_bias_crosspoint_conductance(self):
        """
        Get the conductance of the bias crosspoint.
        
        Returns:
            float: bias conductance
        """
        return self.bias_crosspoint.conductance_p(self.bias_state)

    def __str__(self):
        return f"MultiWeightSynapse"