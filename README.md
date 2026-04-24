# Battery Passport System

## Project Overview

A comprehensive Battery Passport System for monitoring electric vehicle (EV) battery health through real-time sensor data analysis. The system employs Physics-Informed Neural Networks (PINNs) to estimate internal battery electrochemical parameters.

## Key Features

- **Real-time Monitoring**: Continuous measurement of terminal voltage and current flow
- **Parameter Estimation**: Prediction of 8 critical battery parameters:
  - Diffusion Capacitance: C1, C2
  - Internal Resistance: R0
  - Solution Resistance: R1, R2
  - Hysteresis Parameters: γ (gamma1), M0, M
- **Machine Learning**: Custom neural network implementation with physics-informed constraints
- **Optimization**: Particle Swarm Optimization (PSO) for parameter tuning
- **Security**: Tamper-proof sensor module ensuring warranty compliance

## Battery Parameters

The system estimates the following electrochemical parameters:

1. **C1, C2** - Diffusion capacitances (F)
2. **R0** - Internal resistance (Ω)
3. **R1, R2** - Solution resistances (Ω)
4. **gamma1** - Hysteresis parameter
5. **M0, M** - Hysteresis magnitude parameters

These parameters provide insights into:
- State of Charge (SOC)
- State of Health (SOH)
- Voltage behavior prediction

## Project Structure

```
Battery_Passport_System/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── train.py                          # Main training script
├── demo.py                           # Demo and testing script
├── demo_simple.py                    # Simplified demo script
├── quick_test.py                     # Quick test script
├── project_info.py                   # Project information
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── neural_network/               # Neural network implementation
│   │   ├── __init__.py
│   │   └── deep_neural_network.py    # Custom NN from scratch
│   ├── optimization/                 # Optimization algorithms
│   │   ├── __init__.py
│   │   └── pso.py                    # Particle Swarm Optimization
│   ├── battery_model/                # Battery physics model
│   │   ├── __init__.py
│   │   └── battery_physics.py        # Electrochemical model
│   ├── data_processing/              # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py            # Data preprocessing
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       └── utils.py                  # Helper functions
│
├── config/                           # Configuration files
│   └── config.py                     # System configuration
│
├── data/                             # Data storage
│   ├── raw/                          # Raw sensor data (Excel files)
│   ├── processed/                    # Processed CSV data
│   │   └── battery_data.csv          # Main dataset
│   └── external/                     # External datasets
│
├── notebooks/                        # Jupyter notebooks
│   └── data_exploration.ipynb        # Data analysis notebook
│
├── models/                           # Saved model weights
├── results/                          # Experiment outputs
│   ├── trained_models/               # Saved models
│   ├── optimization_results/         # PSO results
│   ├── plots/                        # Generated plots
│   └── reports/                      # Analysis reports
│
├── tests/                            # Unit tests
├── docs/                             # Documentation
└── config/                           # Configuration files
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```bash
python train.py
```

## Usage

### Training the Neural Network
```python
from src.neural_network.deep_neural_network import DeepNeuralNetwork
from src.data_processing.data_loader import load_battery_data

# Load data
data = load_battery_data('data/processed/battery_data.csv')

# Initialize and train model
model = DeepNeuralNetwork(sizes=[14, 64, 64, 64, 8])
model.train(data, epochs=50, batch_size=32)
```

### Running PSO Optimization
```python
from src.optimization.pso import BatteryPSO

# Initialize PSO
pso = BatteryPSO(n_particles=100)

# Run optimization
best_params, best_score = pso.optimize(dataset)
```

## Data Format

The system expects CSV data with the following columns:
- Time: Timestamp
- Current_sense: Measured current (A)
- Terminal_voltage: Measured voltage (V)
- Additional derived features (moving averages, RMS, derivatives)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary and confidential. All rights reserved.

## Contact

For questions or support, please contact the development team.