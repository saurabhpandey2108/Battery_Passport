"""
Deep Neural Network implementation for Battery Parameter Estimation

This module contains a custom implementation of a deep neural network
optimized for predicting battery electrochemical parameters from
sensor measurements.
"""

import numpy as np
import time
from scipy.interpolate import interp1d


class DeepNeuralNetwork:
    """
    Deep Neural Network for battery parameter estimation.
    
    This network predicts 8 electrochemical parameters (C1, C2, R0, R1, R2, gamma1, M0, M)
    from battery sensor measurements (voltage, current, and derived features).
    """
    
    def __init__(self, sizes, activation='relu', dropout_rate=0.2):
        """
        Initialize the neural network.
        
        Args:
            sizes (list): Network architecture [input_size, hidden1, hidden2, ..., output_size]
            activation (str): Activation function ('relu' or 'sigmoid')
            dropout_rate (float): Dropout rate for regularization
        """
        assert len(sizes) >= 3, "Network must have at least 3 layers (input, hidden, output)."
        self.sizes = sizes
        self.dropout_rate = dropout_rate
        
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not supported, please use 'relu' or 'sigmoid' instead.")
        
        self.params = self.initialize()
        self.cache = {}
        self.adam_opt = self.initialize_adam_optimizer()
        self.t = 1

    def relu(self, x, derivative=False):
        """ReLU activation function."""
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        """Sigmoid activation function."""
        if derivative:
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def initialize(self):
        """Initialize network parameters using He initialization."""
        params = {}
        for i in range(1, len(self.sizes)):
            # He initialization for ReLU
            scale = np.sqrt(2.0 / self.sizes[i-1])
            params[f"W{i}"] = np.random.randn(self.sizes[i], self.sizes[i-1]) * scale
            params[f"b{i}"] = np.zeros((self.sizes[i], 1))
        return params

    def initialize_adam_optimizer(self):
        """Initialize Adam optimizer parameters."""
        return {
            "m": {key: np.zeros_like(value) for key, value in self.params.items()},
            "v": {key: np.zeros_like(value) for key, value in self.params.items()}
        }

    def feed_forward(self, x, training=True):
        """
        Forward propagation through the network.
        
        Args:
            x (np.array): Input features
            training (bool): Whether in training mode (affects dropout)
            
        Returns:
            np.array: Network output (predicted parameters)
        """
        self.cache["A0"] = x.T

        for i in range(1, len(self.sizes) - 1):
            self.cache[f"Z{i}"] = np.matmul(self.params[f"W{i}"], self.cache[f"A{i-1}"]) + self.params[f"b{i}"]
            self.cache[f"A{i}"] = self.activation(self.cache[f"Z{i}"])
            
            # Apply dropout during training
            if training and i < len(self.sizes) - 2:
                mask = np.random.binomial(1, 1-self.dropout_rate, size=self.cache[f"A{i}"].shape) / (1-self.dropout_rate)
                self.cache[f"A{i}"] *= mask

        # Output layer (no activation for regression)
        final_layer = len(self.sizes) - 2
        self.cache[f"Z{final_layer + 1}"] = np.matmul(self.params[f"W{final_layer + 1}"], 
                                                     self.cache[f"A{final_layer}"]) + self.params[f"b{final_layer + 1}"]
        self.cache[f"A{final_layer + 1}"] = self.cache[f"Z{final_layer + 1}"].T
        return self.cache[f"A{final_layer + 1}"]

    def back_propagate(self, voltage_error):
        """
        Backpropagation algorithm.
        
        Args:
            voltage_error (np.array): Error between predicted and actual voltages
            
        Returns:
            dict: Gradients for all parameters
        """
        current_batch_size = voltage_error.shape[0]
        grads = {}

        # Calculate gradients based on voltage error
        # For physics-informed neural networks, we need to propagate the error 
        # through the battery model to the network outputs (parameters)
        # First reshape the error to match the output layer
        dZ = np.zeros((voltage_error.shape[0], self.sizes[-1]))
        for i in range(self.sizes[-1]):
            dZ[:, i] = voltage_error.flatten()
        
        dZ = dZ.T  # Transpose for correct dimensions
        last_layer = len(self.sizes) - 2
        
        # Output layer gradients
        grads[f"W{last_layer + 1}"] = (1./current_batch_size) * np.matmul(dZ, self.cache[f"A{last_layer}"].T)
        grads[f"b{last_layer + 1}"] = (1./current_batch_size) * np.sum(dZ, axis=1, keepdims=True)

        # Backpropagate through hidden layers
        for i in range(last_layer, 0, -1):
            dA = np.matmul(self.params[f"W{i + 1}"].T, dZ)
            dZ = dA * self.activation(self.cache[f"Z{i}"], derivative=True)
            grads[f"W{i}"] = (1./current_batch_size) * np.matmul(dZ, self.cache[f"A{i - 1}"].T)
            grads[f"b{i}"] = (1./current_batch_size) * np.sum(dZ, axis=1, keepdims=True)

        self.grads = grads
        return self.grads

    def optimize(self, l_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer update step."""
        for key in self.params:
            self.adam_opt['m'][key] = beta1 * self.adam_opt['m'][key] + (1 - beta1) * self.grads[key]
            self.adam_opt['v'][key] = beta2 * self.adam_opt['v'][key] + (1 - beta2) * (self.grads[key] ** 2)

            m_hat = self.adam_opt['m'][key] / (1 - beta1 ** self.t)
            v_hat = self.adam_opt['v'][key] / (1 - beta2 ** self.t)

            self.params[key] -= l_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        self.t += 1

    def rmse_loss(self, y, output):
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((y - output) ** 2))

    def train(self, dataset, epochs=50, batch_size=32, l_rate=0.0001, 
              validation_split=0.2, early_stopping_patience=10):
        """
        Train the neural network.
        
        Args:
            dataset (np.array): Training dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            l_rate (float): Learning rate
            validation_split (float): Fraction of data for validation
            early_stopping_patience (int): Patience for early stopping
            
        Returns:
            tuple: Training history (losses, actual_voltages, predicted_voltages)
        """
        # Split data into training and validation
        val_size = int(len(dataset) * validation_split)
        np.random.shuffle(dataset)
        val_data = dataset[:val_size]
        train_data = dataset[val_size:]
        
        num_batches = len(train_data) // batch_size
        if num_batches == 0:
            raise ValueError("Batch size is too large for dataset.")

        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        print(f"Batch size: {batch_size}, Batches per epoch: {num_batches}")

        for epoch in range(epochs):
            start_time = time.time()
            epoch_train_loss = 0
            
            # Shuffle training data each epoch
            np.random.shuffle(train_data)
            
            # Training loop
            for batch_idx in range(num_batches):
                batch = train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                
                # Prepare batch data
                x = batch[:, 1:]  # Features (excluding first column)
                actual_voltages = batch[:, 2].reshape(-1, 1)  # Target voltages
                
                # Forward pass
                predicted_params = self.feed_forward(x, training=True)
                
                # Calculate predicted voltages using battery model
                predicted_voltages = self._calculate_voltages(predicted_params, x)
                
                # Calculate voltage error
                voltage_error = actual_voltages - predicted_voltages.reshape(-1, 1)
                
                # Backpropagation and optimization
                self.back_propagate(voltage_error)
                self.optimize(l_rate=l_rate)
                
                # Accumulate loss
                batch_loss = self.rmse_loss(actual_voltages, predicted_voltages.reshape(-1, 1))
                epoch_train_loss += batch_loss
            
            # Calculate average training loss
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = self._validate(val_data)
            val_losses.append(val_loss)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - "
                  f"Train RMSE: {avg_train_loss:.6f} - Val RMSE: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_params = {key: value.copy() for key, value in self.params.items()}
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                # Restore best parameters
                self.params = self.best_params
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

    def _validate(self, val_data):
        """Perform validation."""
        total_loss = 0
        num_samples = len(val_data)
        
        for i in range(0, num_samples, 32):  # Process in small batches
            batch = val_data[i:i+32]
            x = batch[:, 1:]
            actual_voltages = batch[:, 2].reshape(-1, 1)
            
            # Forward pass (no training mode)
            predicted_params = self.feed_forward(x, training=False)
            predicted_voltages = self._calculate_voltages(predicted_params, x)
            
            batch_loss = self.rmse_loss(actual_voltages, predicted_voltages.reshape(-1, 1))
            total_loss += batch_loss * len(batch)
        
        return total_loss / num_samples

    def _calculate_voltages(self, predicted_params, x):
        """Calculate voltages using the battery model."""
        from ..battery_model.battery_physics import BatteryModel
        
        battery_model = BatteryModel()
        predicted_voltages = []
        
        for idx in range(len(x)):
            try:
                C1, C2, R0, R1, R2, gamma1, M0, M = predicted_params[idx]
                i = x[idx][1]  # Current value
                v, _, _, _, _, _ = battery_model.calculate_voltage(C1, C2, R0, R1, R2, gamma1, M0, M, i)
                predicted_voltages.append(v)
            except Exception as e:
                # Default voltage if calculation fails
                predicted_voltages.append(3.0)
        
        return np.array(predicted_voltages)

    def predict(self, x):
        """Make predictions on new data."""
        return self.feed_forward(x, training=False)

    def save_model(self, filepath):
        """Save model parameters."""
        np.savez(filepath, **self.params)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters."""
        loaded = np.load(filepath)
        self.params = {key: loaded[key] for key in loaded.keys()}
        print(f"Model loaded from {filepath}")