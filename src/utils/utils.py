"""
Utility functions for Battery Passport System
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging
import time


def setup_logging(config: Dict) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config (dict): Logging configuration
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.get('log_dir', 'logs'), exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('battery_passport')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Create formatter
    formatter = logging.Formatter(config.get('format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if config.get('file_logging', True):
        log_file = os.path.join(config.get('log_dir', 'logs'), 
                               f"battery_passport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Ensure arrays are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Avoid division by zero for MAPE
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history with losses
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(history['train_losses'], label='Training Loss', alpha=0.8)
    if 'val_losses' in history:
        axes[0].plot(history['val_losses'], label='Validation Loss', alpha=0.8)
    axes[0].set_title('Training History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot learning rate (if available)
    if 'learning_rates' in history:
        axes[1].plot(history['learning_rates'])
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_parameter_comparison(predicted_params: np.ndarray, 
                            target_params: np.ndarray = None,
                            param_names: List[str] = None,
                            save_path: str = None):
    """
    Plot comparison of predicted vs target parameters.
    
    Args:
        predicted_params (np.ndarray): Predicted parameters
        target_params (np.ndarray): Target parameters (optional)
        param_names (list): Parameter names
        save_path (str): Path to save the plot
    """
    if param_names is None:
        param_names = ['C1', 'C2', 'R0', 'R1', 'R2', 'gamma1', 'M0', 'M']
    
    # Take mean of predictions if multiple samples
    if predicted_params.ndim > 1:
        pred_mean = np.mean(predicted_params, axis=0)
        pred_std = np.std(predicted_params, axis=0)
    else:
        pred_mean = predicted_params
        pred_std = np.zeros_like(pred_mean)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(param_names))
    
    # Plot predicted parameters
    bars = ax.bar(x, pred_mean, yerr=pred_std, capsize=5, 
                  alpha=0.7, label='Predicted', color='skyblue')
    
    # Plot target parameters if provided
    if target_params is not None:
        ax.scatter(x, target_params, color='red', s=100, 
                  label='Target', marker='D', zorder=5)
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Parameter Values')
    ax.set_title('Battery Parameter Estimation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale if values span multiple orders of magnitude
    if np.max(pred_mean) / np.min(pred_mean[pred_mean > 0]) > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter comparison plot saved to {save_path}")
    
    plt.show()


def plot_voltage_prediction(actual_voltages: np.ndarray, 
                          predicted_voltages: np.ndarray,
                          time_points: np.ndarray = None,
                          save_path: str = None):
    """
    Plot voltage prediction results.
    
    Args:
        actual_voltages (np.ndarray): Actual voltage measurements
        predicted_voltages (np.ndarray): Predicted voltages
        time_points (np.ndarray): Time points (optional)
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if time_points is None:
        time_points = np.arange(len(actual_voltages))
    
    # Time series plot
    axes[0, 0].plot(time_points, actual_voltages, label='Actual', alpha=0.8)
    axes[0, 0].plot(time_points, predicted_voltages, label='Predicted', alpha=0.8)
    axes[0, 0].set_title('Voltage Prediction Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0, 1].scatter(actual_voltages, predicted_voltages, alpha=0.6)
    min_val = min(np.min(actual_voltages), np.min(predicted_voltages))
    max_val = max(np.max(actual_voltages), np.max(predicted_voltages))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Actual Voltage (V)')
    axes[0, 1].set_ylabel('Predicted Voltage (V)')
    axes[0, 1].set_title('Actual vs Predicted Voltage')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error plot
    error = predicted_voltages - actual_voltages
    axes[1, 0].plot(time_points, error)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_title('Prediction Error Over Time')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Error (V)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1, 1].hist(error, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].set_xlabel('Error (V)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Voltage prediction plot saved to {save_path}")
    
    plt.show()


def save_results(results: Dict, filepath: str):
    """
    Save results to file.
    
    Args:
        results (dict): Results dictionary
        filepath (str): Path to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Determine file format from extension
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif ext.lower() == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        # Default to pickle
        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from file.
    
    Args:
        filepath (str): Path to results file
        
    Returns:
        dict: Loaded results
    """
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        for key, value in results.items():
            if isinstance(value, list) and key in ['parameters', 'voltages', 'history']:
                results[key] = np.array(value)
    
    elif ext.lower() == '.pkl':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return results


def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create directory for experiment results.
    
    Args:
        base_dir (str): Base directory for experiments
        experiment_name (str): Name of the experiment
        
    Returns:
        str: Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'plots', 'logs', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    # If using other libraries, set their seeds too
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config (dict): Configuration to validate
        required_keys (list): Required keys
        
    Returns:
        bool: True if valid
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"Missing required configuration keys: {missing_keys}")
        return False
    
    return True


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.description} completed in {elapsed:.2f} seconds")


def format_parameter_table(parameters: np.ndarray, 
                          param_names: List[str] = None,
                          target_values: np.ndarray = None) -> str:
    """
    Format parameters as a readable table.
    
    Args:
        parameters (np.ndarray): Parameter values
        param_names (list): Parameter names
        target_values (np.ndarray): Target values (optional)
        
    Returns:
        str: Formatted table string
    """
    if param_names is None:
        param_names = ['C1', 'C2', 'R0', 'R1', 'R2', 'gamma1', 'M0', 'M']
    
    lines = []
    lines.append("Battery Parameters:")
    lines.append("-" * 60)
    
    if target_values is not None:
        lines.append(f"{'Parameter':<10} {'Target':<15} {'Estimated':<15} {'Error':<15}")
        lines.append("-" * 60)
        for name, target, estimated in zip(param_names, target_values, parameters):
            error = abs(estimated - target)
            lines.append(f"{name:<10} {target:<15.6f} {estimated:<15.6f} {error:<15.6f}")
    else:
        lines.append(f"{'Parameter':<10} {'Value':<15}")
        lines.append("-" * 30)
        for name, value in zip(param_names, parameters):
            lines.append(f"{name:<10} {value:<15.6f}")
    
    return "\n".join(lines)