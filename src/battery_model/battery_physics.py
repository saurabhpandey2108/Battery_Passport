"""
Battery Physics Model

This module implements the electrochemical battery model used for
voltage prediction based on estimated parameters.
"""

import numpy as np
from scipy.interpolate import interp1d


class BatteryModel:
    """
    Electrochemical battery model for voltage calculation.
    
    This class implements the battery physics equations used to calculate
    terminal voltage based on electrochemical parameters.
    """
    
    def __init__(self):
        """Initialize the battery model with default OCV curve."""
        self.setup_ocv_curve()
    
    def setup_ocv_curve(self):
        """Setup the Open Circuit Voltage (OCV) curve."""
        # OCV curve data points
        self.ocv_y = np.array([2.5, 2.5999, 2.757, 3.0026, 3.1401, 3.2088, 3.2383,
                              3.2726, 3.2972, 3.3119, 3.3119, 3.3365, 3.3709, 3.4887, 3.5])
        self.ocv_x = np.array([0, 0.18474, 0.71411, 3.5374, 7.243, 12.36, 20.124,
                              32.3, 44.828, 60.0004, 70.591, 84.708, 97.413, 99.707, 100])
        
        # Create interpolation function
        self.ocv_interp = interp1d(self.ocv_x, self.ocv_y, kind='linear', fill_value="extrapolate")
    
    def calculate_ocv(self, soc):
        """
        Calculate Open Circuit Voltage based on State of Charge.
        
        Args:
            soc (float): State of Charge (0-100%)
            
        Returns:
            float: Open Circuit Voltage (V)
        """
        # Ensure SOC is within reasonable bounds
        soc = np.clip(soc, 0, 100)
        return float(self.ocv_interp(soc))
    
    def calculate_voltage(self, C1, C2, R0, R1, R2, gamma1, M0, M, i, 
                         ir1=0, ir2=0, z=100, h=0, s=0):
        """
        Calculate terminal voltage using the battery model.
        
        This implements the electrochemical model:
        V = OCV(z) + Vh - R1*ir1 - R2*ir2 - R0*i
        
        Args:
            C1, C2 (float): Diffusion capacitances (F)
            R0 (float): Internal resistance (Ω)
            R1, R2 (float): Solution resistances (Ω)
            gamma1 (float): Hysteresis parameter
            M0, M (float): Hysteresis magnitude parameters
            i (float): Current (A)
            ir1, ir2 (float): RC circuit currents
            z (float): State of charge (%)
            h (float): Hysteresis state
            s (float): Sign state
            
        Returns:
            tuple: (voltage, ir1, ir2, z, h, s)
        """
        # Apply parameter constraints to ensure physical validity
        C1 = np.clip(C1, 3000, 450000)
        C2 = np.clip(C2, 3000, 450000)
        R0 = np.clip(R0, 1e-6, 10)
        R1 = np.clip(R1, 1e-6, 10)
        R2 = np.clip(R2, 1e-6, 10)
        gamma1 = np.clip(gamma1, 8, 1200)
        M0 = np.clip(M0, 0, 5)
        M = np.clip(M, 0, 8)
        
        # Model constants
        Ts = 0.0001  # Sampling time (s)
        Q = 10 * 3600  # Battery capacity (As)
        n = 1  # Efficiency factor
        
        # Small epsilon to prevent division by zero
        eps = 1e-10
        
        try:
            # Update state of charge
            z = z - Ts * n * i / (Q + eps)
            z = np.clip(z, 0, 100)  # Keep SOC within bounds
            
            # Update RC circuit currents (first-order dynamics)
            tau1 = (R1 + eps) * (C1 + eps)
            tau2 = (R2 + eps) * (C2 + eps)
            
            ir1 = np.exp(-Ts / tau1) * ir1 + (1 - np.exp(-Ts / tau1)) * i
            ir2 = np.exp(-Ts / tau2) * ir2 + (1 - np.exp(-Ts / tau2)) * i
            
            # Update hysteresis state
            u = np.exp(-abs(n * i * gamma1 * Ts / (Q + eps)))
            h = u * h - (1 - u) * (1.0 if i > 0 else 0.0)
            
            # Update sign state
            s = 1 if i > 0 else s
            
            # Calculate hysteresis voltage
            vh = M0 * s + M * h
            
            # Calculate terminal voltage
            v = self.calculate_ocv(z) + vh - R1 * ir1 - R2 * ir2 - R0 * i
            
            return round(v, 4), ir1, ir2, z, h, s
            
        except Exception as e:
            # Return safe default values if calculation fails
            return 3.0, ir1, ir2, z, h, s
    
    def simulate_discharge(self, params, current_profile, initial_soc=100):
        """
        Simulate battery discharge with given current profile.
        
        Args:
            params (tuple): Battery parameters (C1, C2, R0, R1, R2, gamma1, M0, M)
            current_profile (np.array): Current values over time
            initial_soc (float): Initial state of charge
            
        Returns:
            dict: Simulation results containing voltages, SOC, etc.
        """
        C1, C2, R0, R1, R2, gamma1, M0, M = params
        
        # Initialize state variables
        ir1, ir2 = 0, 0
        z = initial_soc
        h, s = 0, 0
        
        # Results storage
        voltages = []
        socs = []
        
        for i in current_profile:
            v, ir1, ir2, z, h, s = self.calculate_voltage(
                C1, C2, R0, R1, R2, gamma1, M0, M, i, ir1, ir2, z, h, s
            )
            voltages.append(v)
            socs.append(z)
        
        return {
            'voltages': np.array(voltages),
            'socs': np.array(socs),
            'currents': current_profile
        }
    
    def validate_parameters(self, params):
        """
        Validate if parameters are within physical bounds.
        
        Args:
            params (tuple): Battery parameters
            
        Returns:
            bool: True if parameters are valid
        """
        C1, C2, R0, R1, R2, gamma1, M0, M = params
        
        bounds = {
            'C1': (3000, 450000),
            'C2': (3000, 450000),
            'R0': (1e-6, 10),
            'R1': (1e-6, 10),
            'R2': (1e-6, 10),
            'gamma1': (8, 1200),
            'M0': (0, 5),
            'M': (0, 8)
        }
        
        param_values = [C1, C2, R0, R1, R2, gamma1, M0, M]
        param_names = list(bounds.keys())
        
        for i, (param_val, param_name) in enumerate(zip(param_values, param_names)):
            min_val, max_val = bounds[param_name]
            if not (min_val <= param_val <= max_val):
                return False
        
        return True