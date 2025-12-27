"""
Differential drive dynamics model for mobile robot.

This module implements the kinematic model for a differential drive robot
with support for different facing directions.
"""

import numpy as np
import casadi as cs
from typing import Literal


class DifferentialDriveDynamics:
    """
    Differential drive robot dynamics model.
    
    Implements continuous and discrete dynamics for a differential drive robot
    with configurable facing direction in the robot's local frame.
    
    State: [x, y, theta]
    Input: [v, omega] where v is linear velocity, omega is angular velocity
    """
    
    VALID_DIRECTIONS = ['x', 'y', '-x', '-y']
    
    def __init__(
        self,
        dt: float,
        facing_direction: Literal['x', 'y', '-x', '-y'] = '-y'
    ):
        """
        Initialize differential drive dynamics.
        
        Parameters
        ----------
        dt : float
            Time discretization step
        facing_direction : str
            Direction robot faces in its local frame:
            - 'x': robot front points in +x direction
            - 'y': robot front points in +y direction
            - '-x': robot front points in -x direction  
            - '-y': robot front points in -y direction (Albert default)
        
        Raises
        ------
        ValueError
            If facing_direction is not one of the valid options
        """
        if facing_direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid facing_direction: {facing_direction}. "
                f"Must be one of {self.VALID_DIRECTIONS}"
            )
        
        self.dt = dt
        self.facing_direction = facing_direction
        
        # State and input dimensions
        self.state_dim = 3  # [x, y, theta]
        self.input_dim = 2  # [v, omega]
        
        # State limits [x, y, theta]
        self.x_min = np.array([-10.0, -10.0, -2*np.pi])
        self.x_max = np.array([100.0, 100.0, 2*np.pi])
        
        # Input limits [v, omega]
        self.u_min = np.array([-1.0, -2.0])
        self.u_max = np.array([1.0, 2.0])
    
    def continuous_dynamics(self, x: cs.MX, u: cs.MX) -> cs.MX:
        """
        Continuous-time dynamics: ẋ = f(x, u)
        
        Parameters
        ----------
        x : cs.MX
            State [x, y, theta] (CasADi symbolic)
        u : cs.MX
            Input [v, omega] (CasADi symbolic)
        
        Returns
        -------
        cs.MX
            State derivative [ẋ, ẏ, θ̇]
        """
        # Extract states
        theta = x[2]
        
        # Extract inputs
        v = u[0]      # Linear velocity
        omega = u[1]  # Angular velocity
        
        # Compute state derivatives based on facing direction
        if self.facing_direction == 'x':
            # Robot facing +x: forward motion increases x
            x_dot = v * cs.cos(theta)
            y_dot = v * cs.sin(theta)
        
        elif self.facing_direction == '-x':
            # Robot facing -x: forward motion decreases x
            x_dot = -v * cs.cos(theta)
            y_dot = -v * cs.sin(theta)
        
        elif self.facing_direction == 'y':
            # Robot facing +y: forward motion increases y
            x_dot = -v * cs.sin(theta)
            y_dot = v * cs.cos(theta)
        
        elif self.facing_direction == '-y':
            # Robot facing -y: forward motion decreases y
            x_dot = v * cs.sin(theta)
            y_dot = -v * cs.cos(theta)
        
        # Angular velocity directly controls theta_dot
        theta_dot = omega
        
        return cs.vertcat(x_dot, y_dot, theta_dot)
    
    def discrete_dynamics(self, x: cs.MX, u: cs.MX) -> cs.MX:
        """
        Discrete-time dynamics using Euler integration.
        
        x[k+1] = x[k] + dt * f(x[k], u[k])
        
        Parameters
        ----------
        x : cs.MX
            State at time k (CasADi symbolic)
        u : cs.MX
            Input at time k (CasADi symbolic)
        
        Returns
        -------
        cs.MX
            State at time k+1
        """
        return x + self.dt * self.continuous_dynamics(x, u)
    
    def simulate_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Simulate one step using numpy arrays (for simulation loop).
        
        This is used outside the optimization for actual simulation.
        
        Parameters
        ----------
        x : np.ndarray
            Current state [x, y, theta] (3,)
        u : np.ndarray
            Current input [v, omega] (2,)
        
        Returns
        -------
        np.ndarray
            Next state [x, y, theta] (3,)
        """
        # Initialize next state
        x_next = np.zeros(3)
        
        # Extract current state
        theta = x[2]
        
        # Extract inputs
        v = u[0]
        omega = u[1]
        
        # Compute next state based on facing direction
        if self.facing_direction == 'x':
            x_next[0] = x[0] + self.dt * v * np.cos(theta)
            x_next[1] = x[1] + self.dt * v * np.sin(theta)
        
        elif self.facing_direction == '-x':
            x_next[0] = x[0] - self.dt * v * np.cos(theta)
            x_next[1] = x[1] - self.dt * v * np.sin(theta)
        
        elif self.facing_direction == 'y':
            x_next[0] = x[0] - self.dt * v * np.sin(theta)
            x_next[1] = x[1] + self.dt * v * np.cos(theta)
        
        elif self.facing_direction == '-y':
            x_next[0] = x[0] + self.dt * v * np.sin(theta)
            x_next[1] = x[1] - self.dt * v * np.cos(theta)
        
        # Update orientation
        x_next[2] = x[2] + self.dt * omega
        
        return x_next
    
    def set_state_limits(self, x_min: np.ndarray, x_max: np.ndarray):
        """
        Update state limits.
        
        Parameters
        ----------
        x_min : np.ndarray
            Minimum state values [x_min, y_min, theta_min]
        x_max : np.ndarray
            Maximum state values [x_max, y_max, theta_max]
        """
        assert len(x_min) == 3 and len(x_max) == 3
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
    
    def set_input_limits(self, u_min: np.ndarray, u_max: np.ndarray):
        """
        Update input limits.
        
        Parameters
        ----------
        u_min : np.ndarray
            Minimum input values [v_min, omega_min]
        u_max : np.ndarray
            Maximum input values [v_max, omega_max]
        """
        assert len(u_min) == 2 and len(u_max) == 2
        self.u_min = np.array(u_min)
        self.u_max = np.array(u_max)
    
    def __repr__(self) -> str:
        """String representation of the dynamics model."""
        return (
            f"DifferentialDriveDynamics(\n"
            f"  dt={self.dt},\n"
            f"  facing_direction='{self.facing_direction}',\n"
            f"  state_dim={self.state_dim},\n"
            f"  input_dim={self.input_dim},\n"
            f"  state_limits=[{self.x_min}, {self.x_max}],\n"
            f"  input_limits=[{self.u_min}, {self.u_max}]\n"
            f")"
        )