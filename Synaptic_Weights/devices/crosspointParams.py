"""Parameter container for physical conductance equations in crosspoint devices."""


class CrossPointParams:
    """
    Holds the physical constants used by all crosspoint instances.
    
    These parameters define the conductance mapping from discrete state indices
    to continuous conductance values, plus noise injection into the hardware model.
    """

    def __init__(self, g_ap_coefficient, g_p_coefficient, shift_parameter, g_bias_coefficient, noise_stddev):
        """
        Initialize crosspoint parameters.
        
        Args:
            g_ap_coefficient (float): scaling factor for anti-parallel conductance (G_ap)
            g_p_coefficient (float): scaling factor for parallel conductance (G_p)
            shift_parameter (float): additive offset in log argument for G_p stability
            g_bias_coefficient (float): scaling factor for bias conductance
            noise_stddev (float): standard deviation of Gaussian noise injected during updates
        """
        self.g_ap_coefficient = g_ap_coefficient
        self.g_p_coefficient = g_p_coefficient
        self.shift_parameter = shift_parameter
        self.g_bias_coefficient = g_bias_coefficient
        self.noise_stddev = noise_stddev

    def get_noise_stddev(self):
        """Return the noise standard deviation for this parameter set."""
        return self.noise_stddev