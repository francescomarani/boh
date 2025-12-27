import matplotlib.pyplot as plt
import numpy as np

"""
Visualization utilities for Albert robot simulation results.

This module provides plotting functions for analyzing MPC controller
performance and robot trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


class SimulationPlotter:
    """
    Plotter for Albert simulation results.
    
    Provides various plotting methods for trajectory, states, inputs,
    and MPC prediction visualization.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize plotter.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_all(self, history: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive plot with all simulation results.
        
        Parameters
        ----------
        history : Dict
            Simulation history from AlbertSimulation.get_history()
        save_path : str, optional
            Path to save figure. If None, displays interactively.
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create 2x2 grid of subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_trajectory(ax1, history)
        
        # 2. Position vs time (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_position_vs_time(ax2, history)
        
        # 3. Control inputs (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_control_inputs(ax3, history)
        
        # 4. Tracking error (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_tracking_error(ax4, history)
        
        plt.suptitle('Albert Robot MPC Simulation Results', 
                     fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
    
    def _plot_trajectory(self, ax, history: Dict):
        """Plot 2D trajectory with target and MPC predictions."""
        states = history['states']
        target = history['target']
        
        # Plot actual trajectory
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, 
                label='Actual trajectory', zorder=3)
        
        # Plot start and end points
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=12, 
                label='Start', zorder=5)
        ax.plot(states[-1, 0], states[-1, 1], 'bs', markersize=12, 
                label='End', zorder=5)
        
        # Plot target
        ax.plot(target[0], target[1], 'r*', markersize=20, 
                label='Target', zorder=5)
        
        # Plot target tolerance circle
        circle = plt.Circle((target[0], target[1]), 0.1, 
                           color='r', fill=False, linestyle='--', 
                           linewidth=2, label='Target tolerance', zorder=2)
        ax.add_patch(circle)
        
        # Plot MPC predictions (sample every N steps for clarity)
        pred_trajs = history['predicted_trajectories']
        sample_rate = max(1, len(pred_trajs) // 10)
        
        for i in range(0, len(pred_trajs), sample_rate):
            traj = pred_trajs[i]
            ax.plot(traj[0, :], traj[1, :], 'r--', alpha=0.3, 
                   linewidth=1, zorder=1)
        
        # Add MPC prediction label
        if len(pred_trajs) > 0:
            ax.plot([], [], 'r--', alpha=0.5, label='MPC predictions')
        
        ax.set_xlabel('X position [m]', fontsize=12)
        ax.set_ylabel('Y position [m]', fontsize=12)
        ax.set_title('Robot Trajectory', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _plot_position_vs_time(self, ax, history: Dict):
        """Plot x and y positions over time."""
        time = history['time']
        states = history['states']
        target = history['target']
        
        ax.plot(time, states[:, 0], 'b-', linewidth=2, label='X position')
        ax.plot(time, states[:, 1], 'g-', linewidth=2, label='Y position')
        
        # Plot target lines
        ax.axhline(y=target[0], color='b', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='X target')
        ax.axhline(y=target[1], color='g', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='Y target')
        
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Position [m]', fontsize=12)
        ax.set_title('Position vs Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_control_inputs(self, ax, history: Dict):
        """Plot control inputs (velocity and angular velocity)."""
        time = history['time'][:-1]  # One less than states
        inputs = history['inputs']
        
        ax.plot(time, inputs[:, 0], 'b-', linewidth=2, 
               label='Linear velocity [m/s]')
        ax.plot(time, inputs[:, 1], 'r-', linewidth=2, 
               label='Angular velocity [rad/s]')
        
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Control input', fontsize=12)
        ax.set_title('Control Inputs', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_tracking_error(self, ax, history: Dict):
        """Plot tracking error over time."""
        time = history['time']
        states = history['states']
        target = history['target']
        
        # Compute position error
        error = np.linalg.norm(states[:, :2] - target[:2], axis=1)
        
        ax.plot(time, error, 'r-', linewidth=2, label='Position error')
        ax.axhline(y=0.1, color='g', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='Tolerance (0.1 m)')
        
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Error [m]', fontsize=12)
        ax.set_title('Tracking Error', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def plot_trajectory_only(self, history: Dict, save_path: Optional[str] = None):
        """
        Plot only the trajectory (useful for presentations).
        
        Parameters
        ----------
        history : Dict
            Simulation history
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_trajectory(ax, history)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Trajectory plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_control_analysis(self, history: Dict, save_path: Optional[str] = None):
        """
        Detailed control analysis plots.
        
        Parameters
        ----------
        history : Dict
            Simulation history
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        self._plot_control_inputs(axes[0], history)
        self._plot_tracking_error(axes[1], history)
        
        plt.suptitle('Control Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Control analysis plot saved to {save_path}")
        else:
            plt.show()
    
    def print_statistics(self, history: Dict):
        """
        Print simulation statistics.
        
        Parameters
        ----------
        history : Dict
            Simulation history
        """
        states = history['states']
        inputs = history['inputs']
        target = history['target']
        time = history['time']
        
        # Compute statistics
        final_error = np.linalg.norm(states[-1, :2] - target[:2])
        mean_error = np.mean(np.linalg.norm(states[:, :2] - target[:2], axis=1))
        max_velocity = np.max(np.abs(inputs[:, 0]))
        max_angular_vel = np.max(np.abs(inputs[:, 1]))
        
        print("\n" + "=" * 60)
        print("SIMULATION STATISTICS")
        print("=" * 60)
        print(f"Total simulation time:     {time[-1]:.2f} s")
        print(f"Number of steps:           {len(time)}")
        print(f"Average time step:         {np.mean(np.diff(time)):.4f} s")
        print()
        print(f"Final position error:      {final_error:.4f} m")
        print(f"Mean position error:       {mean_error:.4f} m")
        print()
        print(f"Max linear velocity:       {max_velocity:.3f} m/s")
        print(f"Max angular velocity:      {max_angular_vel:.3f} rad/s")
        print()
        print(f"Initial position:          [{states[0, 0]:.2f}, {states[0, 1]:.2f}]")
        print(f"Final position:            [{states[-1, 0]:.2f}, {states[-1, 1]:.2f}]")
        print(f"Target position:           [{target[0]:.2f}, {target[1]:.2f}]")
        print("=" * 60 + "\n")


def quick_plot(history: Dict, save_path: Optional[str] = None):
    """
    Quick plotting function for convenience.
    
    Parameters
    ----------
    history : Dict
        Simulation history from AlbertSimulation
    save_path : str, optional
        Path to save figure
    """
    plotter = SimulationPlotter()
    plotter.plot_all(history, save_path)
    plotter.print_statistics(history)
    

def plot_trajectories(filename, x, u, T):
    f = plt.figure(figsize=(12, 6))

    # Plot position
    ax2 = f.add_subplot(311)
    x1 = x[0, :]
    plt.plot(x1)
    plt.ylabel(r"$p$ $(m)$", fontsize=14)
    plt.yticks([np.min(x1), 0, np.max(x1)])
    plt.ylim([np.min(x1) - 0.1, np.max(x1) + 0.1])
    plt.xlim([0, T])
    plt.grid()

    # Plot velocity
    ax3 = plt.subplot(3, 1, 2)
    x2 = x[1, :]
    plt.plot(x2)
    plt.yticks([np.min(x2), 0, np.max(x2)])
    plt.ylim([np.min(x2) - 0.1, np.max(x2) + 0.1])
    plt.xlim([0, T])
    plt.ylabel(r"$v$ $(m/s)$", fontsize=14)
    plt.grid()

    # Plot acceleration (input)
    ax1 = plt.subplot(3, 1, 3)

    plt.plot(u[0, :])
    plt.ylabel(r"$a$ $(m/s^2)$", fontsize=14)
    plt.yticks([np.min(u), 0, np.max(u)])
    plt.ylim([np.min(u) - 0.1, np.max(u) + 0.1])
    plt.xlabel(r"$t$", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.xlim([0, T])
    plt.show()

def plot_comparison(filename, x_without, x_with, u_without, u_with, T):
    f = plt.figure(figsize=(12, 6))

    # Plot position
    plt.title('Comparison of MPC methods')
    ax2 = f.add_subplot(311)
    plt.plot(x_without[0, :], label='Without Terminal Set')
    plt.plot(x_with[0, :], label='With Terminal Set')
    ax2.legend()
    x1 = x_without[0, :]
    plt.ylabel(r"$p$ $(m)$", fontsize=14)
    plt.yticks([np.min(x1), 0, np.max(x1)])
    plt.ylim([np.min(x1) - 0.1, np.max(x1) + 0.1])
    plt.xlim([0, T])
    plt.grid()

    # Plot velocity
    ax3 = plt.subplot(3, 1, 2)
    x2 = x_without[1, :]
    plt.plot(x_without[1, :], label='Without Terminal Set')
    plt.plot(x_with[1, :], label='With Terminal Set')
    ax3.legend()
    plt.yticks([np.min(x2), 0, np.max(x2)])
    plt.ylim([np.min(x2) - 0.1, np.max(x2) + 0.1])
    plt.xlim([0, T])
    plt.ylabel(r"$v$ $(m/s)$", fontsize=14)
    plt.grid()

    # Plot acceleration (input)
    ax1 = plt.subplot(3, 1, 3)
    plt.plot(u_without[0, :], label='Without Terminal Set')
    plt.plot(u_with[0, :], label='With Terminal Set')
    ax1.legend()
    plt.ylabel(r"$a$ $(m/s^2)$", fontsize=14)
    plt.yticks([np.min(u_without), 0, np.max(u_without)])
    plt.ylim([np.min(u_without) - 0.1, np.max(u_without) + 0.1])
    plt.xlabel(r"$t$", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.xlim([0, T])
    # plt.savefig(filename, bbox_inches='tight')
    plt.show()
