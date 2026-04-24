"""
Configuration settings for Battery Passport System
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')

# Results paths
TRAINED_MODELS_DIR = os.path.join(RESULTS_DIR, 'trained_models')
OPTIMIZATION_RESULTS_DIR = os.path.join(RESULTS_DIR, 'optimization_results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# Battery Model Configuration
BATTERY_CONFIG = {
    # Parameter bounds: [min, max]
    'parameter_bounds': {
        'C1': (3000, 450000),      # Diffusion capacitance 1 (F)
        'C2': (3000, 450000),      # Diffusion capacitance 2 (F)
        'R0': (0.001, 10),         # Internal resistance (Ω)
        'R1': (0.001, 10),         # Solution resistance 1 (Ω)
        'R2': (0.001, 10),         # Solution resistance 2 (Ω)
        'gamma1': (8, 1200),       # Hysteresis parameter
        'M0': (0.001, 5),          # Hysteresis magnitude 1
        'M': (0.001, 8)            # Hysteresis magnitude 2
    },
    
    # Target parameter values (for reference)
    'target_values': {
        'C1': 38000,
        'C2': 38000,
        'R0': 0.0082,
        'R1': 0.0158,
        'R2': 0.0158,
        'gamma1': 100,
        'M0': 0.002,
        'M': 0.05
    },
    
    # Battery physical constants
    'constants': {
        'sampling_time': 0.0001,    # Sampling time (s)
        'capacity': 10 * 3600,      # Battery capacity (As)
        'efficiency': 1,            # Efficiency factor
        'initial_soc': 100          # Initial state of charge (%)
    },
    
    # OCV curve data
    'ocv_curve': {
        'soc_points': [0, 0.18474, 0.71411, 3.5374, 7.243, 12.36, 20.124,
                      32.3, 44.828, 60.0004, 70.591, 84.708, 97.413, 99.707, 100],
        'voltage_points': [2.5, 2.5999, 2.757, 3.0026, 3.1401, 3.2088, 3.2383,
                          3.2726, 3.2972, 3.3119, 3.3119, 3.3365, 3.3709, 3.4887, 3.5]
    }
}

# Neural Network Configuration
NN_CONFIG = {
    'architecture': {
        'input_size': 14,           # Number of input features
        'hidden_layers': [64, 64, 64],  # Hidden layer sizes
        'output_size': 8,           # Number of battery parameters
        'activation': 'relu',       # Activation function
        'dropout_rate': 0.2         # Dropout rate for regularization
    },
    
    'training': {
        'epochs': 100,              # Maximum training epochs
        'batch_size': 32,           # Training batch size
        'learning_rate': 0.001,     # Initial learning rate
        'validation_split': 0.2,    # Fraction for validation
        'early_stopping_patience': 15,  # Early stopping patience
        'optimizer': 'adam',        # Optimizer type
        'beta1': 0.9,              # Adam beta1
        'beta2': 0.999,            # Adam beta2
        'epsilon': 1e-8            # Adam epsilon
    },
    
    'regularization': {
        'l1_lambda': 0.0,          # L1 regularization
        'l2_lambda': 0.0001,       # L2 regularization
        'gradient_clipping': 1.0    # Gradient clipping threshold
    }
}

# PSO Configuration
PSO_CONFIG = {
    'swarm': {
        'n_particles': 100,         # Number of particles
        'max_iterations': 200,      # Maximum iterations
        'tolerance': 1e-6,          # Convergence tolerance
        'early_stopping_patience': 20  # Early stopping patience
    },
    
    'parameters': {
        'inertia_weight': 0.729,    # Inertia weight (w)
        'cognitive_weight': 1.49445, # Cognitive weight (c1)
        'social_weight': 1.49445,   # Social weight (c2)
        'velocity_clamp': True,     # Whether to clamp velocity
        'velocity_max_factor': 0.2  # Max velocity as fraction of search space
    },
    
    'adaptive': {
        'adaptive_weights': False,   # Use adaptive weights
        'w_min': 0.4,               # Minimum inertia weight
        'w_max': 0.9                # Maximum inertia weight
    }
}

# Data Processing Configuration
DATA_CONFIG = {
    'preprocessing': {
        'window_size': 5,           # Window size for rolling features
        'normalize_features': True,  # Whether to normalize features
        'normalize_targets': False,  # Whether to normalize targets
        'remove_outliers': True,    # Whether to remove outliers
        'outlier_threshold': 3.0    # Outlier threshold (std deviations)
    },
    
    'features': {
        'use_moving_averages': True,    # Include moving averages
        'use_rms': True,               # Include RMS values
        'use_peak_to_peak': True,      # Include peak-to-peak values
        'use_derivatives': True,       # Include derivatives
        'use_higher_order': False     # Include higher-order features
    },
    
    'splitting': {
        'train_ratio': 0.7,         # Training data ratio
        'val_ratio': 0.15,          # Validation data ratio
        'test_ratio': 0.15,         # Test data ratio
        'shuffle': True,            # Shuffle data before splitting
        'random_seed': 42           # Random seed for reproducibility
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',                # Logging level
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,           # Save logs to file
    'log_dir': os.path.join(RESULTS_DIR, 'logs'),
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5               # Number of backup log files
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'name': 'battery_parameter_estimation',
    'description': 'Physics-informed neural network for battery parameter estimation',
    'version': '1.0.0',
    'save_models': True,
    'save_results': True,
    'save_plots': True,
    'random_seed': 42
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_gpu': False,              # Whether to use GPU (if available)
    'num_workers': 4,              # Number of worker processes
    'memory_limit': None           # Memory limit (None for no limit)
}

# Validation Configuration
VALIDATION_CONFIG = {
    'cross_validation': {
        'enabled': False,           # Whether to use cross-validation
        'folds': 5,                # Number of CV folds
        'stratified': False        # Whether to use stratified CV
    },
    
    'metrics': {
        'primary_metric': 'rmse',   # Primary optimization metric
        'additional_metrics': ['mae', 'mape', 'r2'],  # Additional metrics to track
        'voltage_tolerance': 0.01   # Acceptable voltage prediction error (V)
    }
}