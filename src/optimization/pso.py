"""
Particle Swarm Optimization for Battery Parameter Estimation

This module implements PSO algorithm specifically designed for optimizing
battery electrochemical parameters.
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from ..battery_model.battery_physics import BatteryModel


class Particle:
    """Individual particle in the PSO swarm."""
    
    def __init__(self, dim, bounds):
        """
        Initialize a particle.
        
        Args:
            dim (int): Dimension of the search space (number of parameters)
            bounds (list): List of (min, max) tuples for each parameter
        """
        self.position = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.best_position = np.zeros(dim)
        self.best_score = float('inf')
        
        # Initialize random position within bounds
        for i in range(dim):
            self.position[i] = random.uniform(bounds[i][0], bounds[i][1])
        self.best_position = self.position.copy()


class BatteryPSO:
    """
    Particle Swarm Optimization for battery parameter estimation.
    
    This class implements PSO specifically designed for optimizing the 8
    electrochemical parameters of the battery model.
    """
    
    def __init__(self, n_particles=50, w=0.729, c1=1.49445, c2=1.49445):
        """
        Initialize the PSO optimizer.
        
        Args:
            n_particles (int): Number of particles in the swarm
            w (float): Inertia weight
            c1 (float): Cognitive weight (personal best)
            c2 (float): Social weight (global best)
        """
        self.n_particles = n_particles
        self.dim = 8  # 8 battery parameters
        
        # Parameter bounds: [C1, C2, R0, R1, R2, gamma1, M0, M]
        self.bounds = [
            (3000, 450000),    # C1 (F)
            (3000, 450000),    # C2 (F)
            (0.1, 10),         # R0 (Ω)
            (0.1, 10),         # R1 (Ω)
            (0.1, 10),         # R2 (Ω)
            (8, 1200),         # gamma1
            (0.1, 5),          # M0
            (0.1, 8)           # M
        ]
        
        # PSO parameters
        self.w = w          # Inertia weight
        self.c1 = c1        # Cognitive weight
        self.c2 = c2        # Social weight
        
        # Initialize particles
        self.particles = [Particle(self.dim, self.bounds) for _ in range(n_particles)]
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        
        # Optimization history
        self.history = []
        
        # Initialize battery model
        self.battery_model = BatteryModel()

    def fitness(self, params, dataset):
        """
        Calculate fitness (RMSE) for a given set of parameters.
        
        Args:
            params (np.array): Battery parameters [C1, C2, R0, R1, R2, gamma1, M0, M]
            dataset (np.array): Dataset with [time, current, voltage, ...]
            
        Returns:
            float: RMSE between predicted and actual voltages
        """
        # Check if parameters are within bounds
        for i, (param, bound) in enumerate(zip(params, self.bounds)):
            if param < bound[0] or param > bound[1]:
                return float('inf')
        
        total_error = 0
        valid_samples = 0
        
        # Initialize state variables for simulation
        ir1, ir2 = 0, 0
        z = 100  # Initial SOC
        h, s = 0, 0
        
        for sample in dataset:
            try:
                i = sample[1]  # Current value
                actual_voltage = sample[2]  # Actual voltage
                
                # Calculate predicted voltage using battery model
                predicted_voltage, ir1, ir2, z, h, s = self.battery_model.calculate_voltage(
                    *params, i, ir1, ir2, z, h, s
                )
                
                if predicted_voltage is not None:
                    error = (predicted_voltage - actual_voltage) ** 2
                    total_error += error
                    valid_samples += 1
                    
            except Exception as e:
                # Skip invalid samples
                continue
        
        if valid_samples == 0:
            return float('inf')
        
        rmse = np.sqrt(total_error / valid_samples)
        return rmse

    def update_particle(self, particle):
        """Update particle velocity and position."""
        for d in range(self.dim):
            r1, r2 = random.random(), random.random()
            
            # Update velocity
            particle.velocity[d] = (
                self.w * particle.velocity[d] +
                self.c1 * r1 * (particle.best_position[d] - particle.position[d]) +
                self.c2 * r2 * (self.global_best_position[d] - particle.position[d])
            )
            
            # Update position
            particle.position[d] = particle.position[d] + particle.velocity[d]
            
            # Ensure position stays within bounds
            particle.position[d] = max(self.bounds[d][0],
                                     min(self.bounds[d][1],
                                         particle.position[d]))

    def optimize(self, dataset, max_iterations=200, tolerance=1e-6, verbose=True):
        """
        Run PSO optimization.
        
        Args:
            dataset (np.array): Training dataset
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            verbose (bool): Print progress
            
        Returns:
            tuple: (best_position, best_score)
        """
        if verbose:
            print("Starting PSO optimization...")
            print(f"Population size: {self.n_particles}")
            print(f"Max iterations: {max_iterations}")
            print("-" * 60)
        
        # Progress tracking
        best_scores = []
        no_improve_count = 0
        prev_best = float('inf')
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # Update each particle
            for particle in self.particles:
                # Calculate fitness
                current_fitness = self.fitness(particle.position, dataset)
                
                # Update personal best
                if current_fitness < particle.best_score:
                    particle.best_score = current_fitness
                    particle.best_position = particle.position.copy()
                    
                    # Update global best
                    if current_fitness < self.global_best_score:
                        self.global_best_score = current_fitness
                        self.global_best_position = particle.position.copy()
                
                # Update particle
                self.update_particle(particle)
            
            # Store best score
            best_scores.append(self.global_best_score)
            
            # Check convergence
            if abs(prev_best - self.global_best_score) < tolerance:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best = self.global_best_score
            
            # Print progress
            if verbose and (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration + 1:3d}/{max_iterations} | "
                      f"Best RMSE: {self.global_best_score:.6f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if no_improve_count >= 20:
                if verbose:
                    print(f"\nEarly stopping at iteration {iteration + 1}")
                    print("No improvement in RMSE for 20 iterations")
                break
        
        self.history = best_scores
        
        if verbose:
            total_time = time.time() - start_time
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            print(f"Best RMSE: {self.global_best_score:.6f}")
            self._print_results()
        
        return self.global_best_position, self.global_best_score

    def _print_results(self):
        """Print optimization results."""
        param_names = ['C1', 'C2', 'R0', 'R1', 'R2', 'gamma1', 'M0', 'M']
        target_values = [38000, 38000, 0.0082, 0.0158, 0.0158, 100, 0.002, 0.05]
        
        print("\nOptimal Parameters:")
        print("-" * 60)
        print(f"{'Parameter':<10} {'Target':<12} {'Optimized':<12} {'Difference':<12}")
        print("-" * 60)
        
        for name, target, optimized in zip(param_names, target_values, self.global_best_position):
            diff = abs(optimized - target)
            print(f"{name:<10} {target:<12.6f} {optimized:<12.6f} {diff:<12.6f}")

    def plot_convergence(self, save_path=None):
        """
        Plot convergence history.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.title('PSO Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Best RMSE')
        plt.grid(True)
        plt.yscale('log')  # Log scale for better visualization
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()

    def get_swarm_diversity(self):
        """Calculate swarm diversity for analysis."""
        positions = np.array([p.position for p in self.particles])
        mean_position = np.mean(positions, axis=0)
        diversity = np.mean([np.linalg.norm(p - mean_position) for p in positions])
        return diversity

    def save_results(self, filepath):
        """Save optimization results."""
        results = {
            'best_position': self.global_best_position,
            'best_score': self.global_best_score,
            'history': self.history,
            'parameters': {
                'n_particles': self.n_particles,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2
            }
        }
        np.savez(filepath, **results)
        print(f"Results saved to {filepath}")


def run_pso_optimization(dataset_path, n_particles=100, max_iterations=200):
    """
    Convenience function to run PSO optimization.
    
    Args:
        dataset_path (str): Path to the dataset
        n_particles (int): Number of particles
        max_iterations (int): Maximum iterations
        
    Returns:
        tuple: (best_params, best_score, pso_instance)
    """
    import pandas as pd
    
    # Load dataset
    try:
        dataset = pd.read_csv(dataset_path).to_numpy()
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None
    
    # Initialize and run PSO
    pso = BatteryPSO(n_particles=n_particles)
    best_params, best_score = pso.optimize(dataset, max_iterations=max_iterations)
    
    return best_params, best_score, pso