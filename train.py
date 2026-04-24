"""
Main training script for Battery Passport System

This script orchestrates the training of neural networks for battery
parameter estimation using physics-informed constraints.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from config.config import *
from src.neural_network.deep_neural_network import DeepNeuralNetwork
from src.data_processing.data_loader import BatteryDataProcessor, load_battery_data
from src.battery_model.battery_physics import BatteryModel
from src.optimization.pso import BatteryPSO
from src.utils.utils import (
    setup_logging, calculate_metrics, plot_training_history,
    plot_parameter_comparison, plot_voltage_prediction,
    save_results, create_experiment_directory, set_random_seed,
    Timer, format_parameter_table
)


class BatteryParameterEstimator:
    """
    Main class for battery parameter estimation using neural networks.
    """
    
    def __init__(self, config_override=None):
        """
        Initialize the estimator.
        
        Args:
            config_override (dict): Configuration overrides
        """
        # Set random seed for reproducibility
        set_random_seed(EXPERIMENT_CONFIG['random_seed'])
        
        # Setup logging
        self.logger = setup_logging(LOGGING_CONFIG)
        self.logger.info("Initializing Battery Parameter Estimator")
        
        # Create experiment directory
        self.experiment_dir = create_experiment_directory(
            RESULTS_DIR, EXPERIMENT_CONFIG['name']
        )
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Initialize components
        self.data_processor = BatteryDataProcessor(
            window_size=DATA_CONFIG['preprocessing']['window_size']
        )
        self.battery_model = BatteryModel()
        self.neural_network = None
        self.pso_optimizer = None
        
        # Results storage
        self.results = {}
    
    def load_data(self, data_path=None):
        """Load and preprocess data."""
        with Timer("Data loading and preprocessing"):
            if data_path is None:
                data_path = os.path.join(PROCESSED_DATA_DIR, 'battery_data.csv')
            
            self.logger.info(f"Loading data from {data_path}")
            
            if not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load processed data
            self.dataset = load_battery_data(data_path)
            self.logger.info(f"Loaded dataset with shape: {self.dataset.shape}")
            
            # Split data
            train_size = int(len(self.dataset) * DATA_CONFIG['splitting']['train_ratio'])
            val_size = int(len(self.dataset) * DATA_CONFIG['splitting']['val_ratio'])
            
            if DATA_CONFIG['splitting']['shuffle']:
                np.random.shuffle(self.dataset)
            
            self.train_data = self.dataset[:train_size]
            self.val_data = self.dataset[train_size:train_size + val_size]
            self.test_data = self.dataset[train_size + val_size:]
            
            self.logger.info(f"Data split - Train: {len(self.train_data)}, "
                           f"Val: {len(self.val_data)}, Test: {len(self.test_data)}")
    
    def initialize_neural_network(self):
        """Initialize the neural network."""
        self.logger.info("Initializing neural network")
        
        arch_config = NN_CONFIG['architecture']
        sizes = [arch_config['input_size']] + arch_config['hidden_layers'] + [arch_config['output_size']]
        
        self.neural_network = DeepNeuralNetwork(
            sizes=sizes,
            activation=arch_config['activation'],
            dropout_rate=arch_config['dropout_rate']
        )
        
        self.logger.info(f"Neural network architecture: {sizes}")
    
    def train_neural_network(self):
        """Train the neural network."""
        if self.neural_network is None:
            self.initialize_neural_network()
        
        self.logger.info("Starting neural network training")
        
        with Timer("Neural network training"):
            training_config = NN_CONFIG['training']
            
            # Combine train and validation data for the custom training method
            combined_data = np.vstack([self.train_data, self.val_data])
            
            history = self.neural_network.train(
                dataset=combined_data,
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size'],
                l_rate=training_config['learning_rate'],
                validation_split=training_config['validation_split'],
                early_stopping_patience=training_config['early_stopping_patience']
            )
            
            self.results['nn_training_history'] = history
            self.logger.info(f"Training completed. Best validation RMSE: {history['best_val_loss']:.6f}")
    
    def evaluate_neural_network(self):
        """Evaluate neural network performance."""
        self.logger.info("Evaluating neural network performance")
        
        # Prepare test data
        X_test = self.test_data[:, 1:]  # Features
        y_test = self.test_data[:, 2].reshape(-1, 1)  # Target voltages
        
        # Make predictions
        predicted_params = self.neural_network.predict(X_test)
        
        # Calculate predicted voltages
        predicted_voltages = []
        for idx in range(len(X_test)):
            C1, C2, R0, R1, R2, gamma1, M0, M = predicted_params[idx]
            i = X_test[idx][1]  # Current value
            v, _, _, _, _, _ = self.battery_model.calculate_voltage(
                C1, C2, R0, R1, R2, gamma1, M0, M, i
            )
            predicted_voltages.append(v)
        
        predicted_voltages = np.array(predicted_voltages).reshape(-1, 1)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, predicted_voltages)
        
        self.results['nn_test_metrics'] = metrics
        self.results['predicted_parameters'] = predicted_params
        self.results['predicted_voltages'] = predicted_voltages
        self.results['actual_voltages'] = y_test
        
        self.logger.info(f"Test RMSE: {metrics['rmse']:.6f}")
        self.logger.info(f"Test MAE: {metrics['mae']:.6f}")
        self.logger.info(f"Test R²: {metrics['r2']:.6f}")
    
    def run_pso_optimization(self):
        """Run PSO optimization for comparison."""
        self.logger.info("Starting PSO optimization")
        
        with Timer("PSO optimization"):
            pso_config = PSO_CONFIG['swarm']
            param_config = PSO_CONFIG['parameters']
            
            self.pso_optimizer = BatteryPSO(
                n_particles=pso_config['n_particles'],
                w=param_config['inertia_weight'],
                c1=param_config['cognitive_weight'],
                c2=param_config['social_weight']
            )
            
            # Use a subset of data for PSO (for faster optimization)
            pso_data = self.train_data[::10]  # Use every 10th sample
            
            best_params, best_score = self.pso_optimizer.optimize(
                dataset=pso_data,
                max_iterations=pso_config['max_iterations'],
                tolerance=pso_config['tolerance'],
                verbose=True
            )
            
            self.results['pso_best_params'] = best_params
            self.results['pso_best_score'] = best_score
            self.results['pso_history'] = self.pso_optimizer.history
            
            self.logger.info(f"PSO completed. Best RMSE: {best_score:.6f}")
    
    def generate_plots(self):
        """Generate visualization plots."""
        self.logger.info("Generating visualization plots")
        
        plots_dir = os.path.join(self.experiment_dir, 'plots')
        
        # Plot training history
        if 'nn_training_history' in self.results:
            plot_training_history(
                self.results['nn_training_history'],
                save_path=os.path.join(plots_dir, 'training_history.png')
            )
        
        # Plot parameter comparison
        if 'predicted_parameters' in self.results:
            target_params = np.array(list(BATTERY_CONFIG['target_values'].values()))
            plot_parameter_comparison(
                self.results['predicted_parameters'],
                target_params,
                save_path=os.path.join(plots_dir, 'parameter_comparison.png')
            )
        
        # Plot voltage predictions
        if 'predicted_voltages' in self.results and 'actual_voltages' in self.results:
            plot_voltage_prediction(
                self.results['actual_voltages'].flatten(),
                self.results['predicted_voltages'].flatten(),
                save_path=os.path.join(plots_dir, 'voltage_prediction.png')
            )
        
        # Plot PSO convergence
        if 'pso_history' in self.results:
            self.pso_optimizer.plot_convergence(
                save_path=os.path.join(plots_dir, 'pso_convergence.png')
            )
    
    def save_models(self):
        """Save trained models."""
        self.logger.info("Saving trained models")
        
        models_dir = os.path.join(self.experiment_dir, 'models')
        
        # Save neural network
        if self.neural_network is not None:
            nn_path = os.path.join(models_dir, 'neural_network.npz')
            self.neural_network.save_model(nn_path)
        
        # Save PSO results
        if self.pso_optimizer is not None:
            pso_path = os.path.join(models_dir, 'pso_results.npz')
            self.pso_optimizer.save_results(pso_path)
    
    def generate_report(self):
        """Generate experiment report."""
        self.logger.info("Generating experiment report")
        
        report_path = os.path.join(self.experiment_dir, 'experiment_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Battery Passport System - Experiment Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment: {EXPERIMENT_CONFIG['name']}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Version: {EXPERIMENT_CONFIG['version']}\n\n")
            
            # Data information
            f.write("Data Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(self.dataset)}\n")
            f.write(f"Training samples: {len(self.train_data)}\n")
            f.write(f"Validation samples: {len(self.val_data)}\n")
            f.write(f"Test samples: {len(self.test_data)}\n\n")
            
            # Neural Network Results
            if 'nn_test_metrics' in self.results:
                f.write("Neural Network Results:\n")
                f.write("-" * 25 + "\n")
                metrics = self.results['nn_test_metrics']
                f.write(f"Test RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"Test MAE: {metrics['mae']:.6f}\n")
                f.write(f"Test MAPE: {metrics['mape']:.2f}%\n")
                f.write(f"Test R²: {metrics['r2']:.6f}\n\n")
            
            # PSO Results
            if 'pso_best_score' in self.results:
                f.write("PSO Optimization Results:\n")
                f.write("-" * 26 + "\n")
                f.write(f"Best RMSE: {self.results['pso_best_score']:.6f}\n\n")
                
                # Parameter comparison
                if 'pso_best_params' in self.results:
                    param_table = format_parameter_table(
                        self.results['pso_best_params'],
                        target_values=np.array(list(BATTERY_CONFIG['target_values'].values()))
                    )
                    f.write(param_table + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write("-" * 13 + "\n")
            f.write(f"Neural Network Architecture: {NN_CONFIG['architecture']}\n")
            f.write(f"Training Configuration: {NN_CONFIG['training']}\n")
            f.write(f"PSO Configuration: {PSO_CONFIG}\n")
        
        self.logger.info(f"Report saved to {report_path}")
    
    def run_complete_experiment(self):
        """Run the complete experiment pipeline."""
        self.logger.info("Starting complete experiment")
        
        try:
            # Load and preprocess data
            self.load_data()
            
            # Train neural network
            self.train_neural_network()
            
            # Evaluate neural network
            self.evaluate_neural_network()
            
            # Run PSO optimization for comparison
            self.run_pso_optimization()
            
            # Generate visualizations
            self.generate_plots()
            
            # Save models
            if EXPERIMENT_CONFIG['save_models']:
                self.save_models()
            
            # Save results
            if EXPERIMENT_CONFIG['save_results']:
                results_path = os.path.join(self.experiment_dir, 'results.pkl')
                save_results(self.results, results_path)
            
            # Generate report
            self.generate_report()
            
            self.logger.info("Experiment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Battery Parameter Estimation Training')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to the dataset CSV file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--no-pso', action='store_true',
                       help='Skip PSO optimization')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with reduced parameters for quick testing')
    
    args = parser.parse_args()
    
    # Modify configuration for quick testing
    if args.quick_test:
        NN_CONFIG['training']['epochs'] = 10
        PSO_CONFIG['swarm']['max_iterations'] = 50
        PSO_CONFIG['swarm']['n_particles'] = 20
    
    # Initialize estimator
    estimator = BatteryParameterEstimator()
    
    try:
        # Load data
        estimator.load_data(args.data_path)
        
        # Train neural network
        estimator.train_neural_network()
        estimator.evaluate_neural_network()
        
        # Run PSO if not disabled
        if not args.no_pso:
            estimator.run_pso_optimization()
        
        # Generate outputs
        estimator.generate_plots()
        estimator.save_models()
        estimator.generate_report()
        
        # Save results
        results_path = os.path.join(estimator.experiment_dir, 'results.pkl')
        save_results(estimator.results, results_path)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved in: {estimator.experiment_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()