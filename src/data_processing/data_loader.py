"""
Data Processing Module for Battery Passport System

This module handles data loading, preprocessing, and feature engineering
for battery sensor data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class BatteryDataProcessor:
    """
    Data processor for battery sensor measurements.
    
    Handles loading, cleaning, and feature engineering of battery data
    including voltage, current, and derived features.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize data processor.
        
        Args:
            window_size (int): Window size for rolling calculations
        """
        self.window_size = window_size
        self.feature_columns = None
        self.target_column = 'Terminal_voltage'
    
    def load_raw_data(self, filepaths: list) -> pd.DataFrame:
        """
        Load and combine raw battery data from multiple Excel files.
        
        Args:
            filepaths (list): List of file paths to load
            
        Returns:
            pd.DataFrame: Combined raw data
        """
        dataframes = []
        
        for filepath in filepaths:
            try:
                df = pd.read_excel(filepath)
                print(f"Loaded {len(df)} samples from {filepath}")
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"Combined dataset shape: {combined_df.shape}")
            return combined_df
        else:
            raise ValueError("No data files could be loaded")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw sensor data.
        
        Args:
            df (pd.DataFrame): Raw data with Time, Current_sense, Terminal_voltage
            
        Returns:
            pd.DataFrame: Data with additional engineered features
        """
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Moving averages
        processed_df['Avg_Current'] = df['Current_sense'].rolling(
            window=self.window_size, min_periods=1).mean()
        processed_df['Avg_Voltage'] = df['Terminal_voltage'].rolling(
            window=self.window_size, min_periods=1).mean()
        
        # Max & Min values
        processed_df['Max_Current'] = df['Current_sense'].rolling(
            window=self.window_size, min_periods=1).max()
        processed_df['Min_Current'] = df['Current_sense'].rolling(
            window=self.window_size, min_periods=1).min()
        processed_df['Max_Voltage'] = df['Terminal_voltage'].rolling(
            window=self.window_size, min_periods=1).max()
        processed_df['Min_Voltage'] = df['Terminal_voltage'].rolling(
            window=self.window_size, min_periods=1).min()
        
        # Peak-to-Peak (Range) values
        processed_df['Peak_Current'] = processed_df['Max_Current'] - processed_df['Min_Current']
        processed_df['Peak_Voltage'] = processed_df['Max_Voltage'] - processed_df['Min_Voltage']
        
        # RMS calculations
        processed_df['RMS_Current'] = np.sqrt(
            (df['Current_sense']**2).rolling(window=self.window_size, min_periods=1).mean())
        processed_df['RMS_Voltage'] = np.sqrt(
            (df['Terminal_voltage']**2).rolling(window=self.window_size, min_periods=1).mean())
        
        # First derivatives (rate of change)
        processed_df['dV/dt'] = df['Terminal_voltage'].diff()
        processed_df['dI/dt'] = df['Current_sense'].diff()
        
        # Fill NaN values from diff() operation
        processed_df['dV/dt'] = processed_df['dV/dt'].fillna(0)
        processed_df['dI/dt'] = processed_df['dI/dt'].fillna(0)
        
        # Drop rows with NaN values from rolling operations
        processed_df = processed_df.dropna()
        
        print(f"Engineered features. Final shape: {processed_df.shape}")
        return processed_df
    
    def normalize_features(self, df: pd.DataFrame, 
                          fit_transform: bool = True) -> pd.DataFrame:
        """
        Normalize features for neural network training.
        
        Args:
            df (pd.DataFrame): Data to normalize
            fit_transform (bool): Whether to fit normalizer or use existing
            
        Returns:
            pd.DataFrame: Normalized data
        """
        # Identify feature columns (exclude Time and Terminal_voltage)
        feature_cols = [col for col in df.columns 
                       if col not in ['Time', 'Terminal_voltage']]
        
        if fit_transform:
            # Store normalization parameters
            self.feature_means = df[feature_cols].mean()
            self.feature_stds = df[feature_cols].std()
            self.voltage_mean = df['Terminal_voltage'].mean()
            self.voltage_std = df['Terminal_voltage'].std()
        
        # Normalize features
        normalized_df = df.copy()
        normalized_df[feature_cols] = (df[feature_cols] - self.feature_means) / self.feature_stds
        
        # Keep voltage unnormalized for physics-informed loss calculation
        # normalized_df['Terminal_voltage'] = (df['Terminal_voltage'] - self.voltage_mean) / self.voltage_std
        
        return normalized_df
    
    def prepare_dataset(self, df: pd.DataFrame, 
                       test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare final dataset for training.
        
        Args:
            df (pd.DataFrame): Processed and normalized data
            test_size (float): Fraction of data for testing
            
        Returns:
            tuple: (train_data, test_data) as numpy arrays
        """
        # Convert to numpy array
        data_array = df.to_numpy()
        
        # Shuffle data
        np.random.shuffle(data_array)
        
        # Split into train/test
        split_idx = int(len(data_array) * (1 - test_size))
        train_data = data_array[:split_idx]
        test_data = data_array[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        return train_data, test_data
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Create sequences for time-series prediction (if needed).
        
        Args:
            data (np.ndarray): Input data
            sequence_length (int): Length of each sequence
            
        Returns:
            np.ndarray: Sequenced data
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """Save processed data to CSV."""
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data from CSV."""
        df = pd.read_csv(filepath)
        print(f"Loaded processed data: {df.shape}")
        return df
    
    def get_feature_info(self, df: pd.DataFrame) -> dict:
        """Get information about features in the dataset."""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'feature_columns': [col for col in df.columns 
                              if col not in ['Time', 'Terminal_voltage']],
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict()
        }
        return info


def load_battery_data(filepath: str) -> np.ndarray:
    """
    Convenience function to load battery data.
    
    Args:
        filepath (str): Path to the processed CSV file
        
    Returns:
        np.ndarray: Loaded data as numpy array
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset: {df.shape}")
        return df.to_numpy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def process_raw_battery_data(raw_data_paths: list, 
                           output_path: str,
                           window_size: int = 5) -> pd.DataFrame:
    """
    Complete pipeline to process raw battery data.
    
    Args:
        raw_data_paths (list): List of paths to raw Excel files
        output_path (str): Path to save processed data
        window_size (int): Window size for feature engineering
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    processor = BatteryDataProcessor(window_size=window_size)
    
    # Load raw data
    raw_df = processor.load_raw_data(raw_data_paths)
    
    # Engineer features
    processed_df = processor.engineer_features(raw_df)
    
    # Normalize features
    normalized_df = processor.normalize_features(processed_df)
    
    # Save processed data
    processor.save_processed_data(normalized_df, output_path)
    
    return normalized_df


if __name__ == "__main__":
    # Example usage
    raw_files = [
        "data/raw/DPW10.xlsx",
        "data/raw/DPW20.xlsx", 
        "data/raw/DPW30.xlsx",
        "data/raw/DPW40.xlsx",
        "data/raw/DPW50.xlsx",
        "data/raw/DPW60.xlsx",
        "data/raw/DPW70.xlsx",
        "data/raw/DPW80.xlsx",
        "data/raw/DPW90.xlsx"
    ]
    
    processed_data = process_raw_battery_data(
        raw_files, 
        "data/processed/battery_data.csv",
        window_size=5
    )