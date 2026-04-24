"""
Test script for neural network training only
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.neural_network.deep_neural_network import DeepNeuralNetwork
from src.data_processing.data_loader import load_battery_data

def test_neural_network():
    """Test neural network training."""
    print("Loading battery data...")
    # Load your battery data
    data_path = os.path.join("data", "processed", "battery_data.csv")
    data_path = os.path.abspath(data_path)
    print(f"Loading data from: {data_path}")
    
    dataset = load_battery_data(data_path)
    
    if dataset is None:
        print("Failed to load data")
        return
    
    print(f"Dataset shape: {dataset.shape}")
    
    # Use a smaller dataset for quick demo
    demo_data = dataset[:50]  # First 50 samples for quick demo
    
    # Define network architecture
    input_size = demo_data.shape[1] - 1  # All columns except the target
    output_size = 8  # Number of battery parameters
    hidden_layers = [32, 32]  # Smaller network for testing
    sizes = [input_size] + hidden_layers + [output_size]
    
    print(f"Network architecture: input={input_size}, hidden={hidden_layers}, output={output_size}")
    
    # Initialize and train the neural network
    dnn = DeepNeuralNetwork(sizes, activation='relu')
    
    # Use a smaller dataset for quick demo
    demo_data = dataset[:100]  # First 100 samples for quick demo
    
    print("Training neural network...")
    history = dnn.train(
        demo_data,
        epochs=3,  # Reduced for demo
        batch_size=10,
        l_rate=0.0001
    )
    
    print("Training completed!")
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")
    if 'val_losses' in history and history['val_losses']:
        print(f"Final validation loss: {history['val_losses'][-1]:.6f}")

if __name__ == "__main__":
    print("Battery Passport System - Neural Network Test")
    print("="*50)
    
    try:
        test_neural_network()
        print("\nNeural network test completed successfully!")
    except Exception as e:
        print(f"Error during neural network test: {e}")
        import traceback
        traceback.print_exc()