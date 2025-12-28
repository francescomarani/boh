"""
Albert Robot Simulation with MPC Controller

This module provides the main simulation class for running Albert robot
in the bar environment with MPC-based navigation.
"""

import numpy as np
from typing import List, Tuple, Optional
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from bar_env import BarEnvironment
from mpc_planning import BaseMPC
from models import DifferentialDriveDynamics


class AlbertSimulation:
    """
    Simulation wrapper for Albert robot with MPC controller in bar environment.
    
    Handles:
    - Environment setup and reset
    - MPC controller initialization
    - Simulation loop execution
    - Data collection for visualization
    """
    
    def __init__(
        self,
        robot_urdf: str,
        initial_pose: np.ndarray,
        target_pose: np.ndarray,
        dt: float = 0.01,
        mpc_horizon: int = 20,
        mpc_weights: Optional[Tuple[float, float]] = None,
        facing_direction: str = '-y',
        render: bool = True,
        max_iter: int = 10,
        warm_start: bool = True
    ):
        """
        Initialize Albert simulation.
        
        Parameters
        ----------
        robot_urdf : str
            Path to robot URDF file
        initial_pose : np.ndarray
            Initial robot pose [x, y, theta]
        target_pose : np.ndarray
            Target pose [x, y, theta]
        dt : float
            Simulation time step
        mpc_horizon : int
            MPC prediction horizon
        mpc_weights : Tuple[float, float], optional
            MPC weights (wx, wu). Defaults to (10.0, 0.1)
        facing_direction : str
            Robot facing direction ('x', 'y', '-x', '-y')
        render : bool
            Enable rendering
        max_iter : int
            Maximum solver iterations for MPC
        warm_start : bool
            Enable warm start for MPC solver
        """
        # Store configuration
        self.dt = dt
        self.initial_pose = initial_pose
        self.target_pose = target_pose
        self.render = render
        
        # MPC weights
        if mpc_weights is None:
            self.wx, self.wu = 10.0, 0.1
        else:
            self.wx, self.wu = mpc_weights
        
        # Initialize environment
        print("Initializing bar environment...")
        robots = [
            GenericDiffDriveRobot(
                urdf="albert.urdf",
                mode="vel",
                actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
                castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
                wheel_radius = 0.08,
                wheel_distance = 0.494,
                spawn_rotation = 0,
                facing_direction = '-y',
            ),
        ]
        self.env = BarEnvironment(
            robots=robots,
            render=render,
            dt=dt
        )
        
        # Initialize dynamics model
        print("Initializing dynamics model...")
        self.dynamics = DifferentialDriveDynamics(
            dt=dt,
            facing_direction=facing_direction
        )
        
        # Initialize MPC controller
        print("Initializing MPC controller...")
        self.mpc = BaseMPC(
            dynamics=self.dynamics,
            N=mpc_horizon,
            x_target=target_pose,
            wx=self.wx,
            wu=self.wu,
            SOLVER_MAX_ITER=max_iter,
            DO_WARM_START=warm_start
        )
        
        # Data storage for plotting
        self.history = {
            'time': [],
            'states': [],
            'inputs': [],
            'predicted_trajectories': []
        }
        
        print("✓ Albert simulation initialized successfully!")
        
        # Print environment action dimension
        self.total_action_dim = self.env.n()
        print(f"Environment action dimension: {self.total_action_dim}")
    
    def reset(self) -> np.ndarray:
        """
        Reset simulation to initial conditions.
        
        Returns
        -------
        np.ndarray
            Initial observation
        """
        # Reset environment

        obs,_ = self.env.reset(
            pos=np.array([self.initial_pose[0], self.initial_pose[1], self.initial_pose[2], 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
        )
        print(f"Initial observation: {obs}")
        # Clear history
        self.history = {
            'time': [],
            'states': [],
            'inputs': [],
            'predicted_trajectories': []
        }
        
        return obs
    
    def run(
        self,
        max_steps: int = 1000,
        tolerance: float = 0.1,
        verbose: bool = True
    ) -> dict:
        """
        Run simulation until target is reached or max steps exceeded.
        
        Parameters
        ----------
        max_steps : int
            Maximum simulation steps
        tolerance : float
            Distance tolerance for reaching target
        verbose : bool
            Print progress information
        
        Returns
        -------
        dict
            Simulation history containing time, states, inputs, and trajectories
        """
        # Reset simulation
        obs = self.reset()
        
        # Extract initial state
        current_state = self._extract_state(obs)
        
        # Simulation loop
        print(f"\nStarting simulation (max_steps={max_steps})...")
        print(f"Initial pose: [{current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}]")
        print(f"Target pose: [{self.target_pose[0]:.2f}, {self.target_pose[1]:.2f}, {self.target_pose[2]:.2f}]")
        print("-" * 60)
        
        for step in range(max_steps):
            # Store current state
            self.history['time'].append(step * self.dt)
            self.history['states'].append(current_state.copy())
            
            # Compute distance to target
            distance_to_target = np.linalg.norm(
                current_state[:2] - self.target_pose[:2]
            )
            
            # Check if target reached
            if distance_to_target < tolerance:
                if verbose:
                    print(f"\n✓ Target reached at step {step}!")
                    print(f"  Final position: [{current_state[0]:.2f}, {current_state[1]:.2f}]")
                    print(f"  Distance to target: {distance_to_target:.4f} m")
                break
            
            # Solve MPC
            try:
                u_opt, x_next_pred, x_traj, theta_next = self.mpc.solve(current_state)
            except Exception as e:
                print(f"\n✗ MPC solver failed at step {step}: {e}")
                break
            
            # Store control input and predicted trajectory
            self.history['inputs'].append(u_opt.copy())
            self.history['predicted_trajectories'].append(x_traj.copy())
            
            # Apply control to environment
            action = np.zeros(self.total_action_dim)
            action[0] = u_opt[0]  # Linear velocity
            action[1] = u_opt[1]  # Angular velocity
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Check for termination
            if terminated or truncated:
                if verbose:
                    print(f"\n✗ Simulation terminated at step {step}")
                    print(f"  Reason: {info}")
                break
            
            # Update current state
            current_state = self._extract_state(obs)
            
            # Print progress
            if verbose and step % 100 == 0:
                print(f"Step {step:4d} | Distance: {distance_to_target:.3f} m | "
                      f"Pos: [{current_state[0]:.2f}, {current_state[1]:.2f}]")
                # DEBUG: Stampa ogni 10 steps
            if step % 10 == 0:
                print(f"\n🔍 DEBUG Step {step}:")
                print(f"  Current state: {current_state}")
                print(f"  MPC input u_opt: {u_opt}")
                print(f"  Action sent to env: {action}")
                print(f"  Observation keys: {list(obs.keys())}")
                if 'robot_0' in obs:
                    print(f"  Robot obs keys: {list(obs['robot_0'].keys())}")
                    if 'joint_state' in obs['robot_0']:
                        print(f"  Joint positions: {obs['robot_0']['joint_state']['position']}")
                        print(f"  Joint velocities: {obs['robot_0']['joint_state']['velocity']}")
        
        # Final statistics
        if verbose:
            print("-" * 60)
            print(f"Simulation completed:")
            print(f"  Total steps: {len(self.history['time'])}")
            print(f"  Total time: {self.history['time'][-1]:.2f} s")
            print(f"  Final distance to target: {distance_to_target:.4f} m")
        
        return self.get_history()
    
    def _extract_state(self, obs: dict) -> np.ndarray:
        """
        Extract [x, y, theta] state from observation.
        
        Parameters
        ----------
        obs : dict
            Environment observation
        
        Returns
        -------
        np.ndarray
            State vector [x, y, theta]
        """
        robot_obs = obs['robot_0']
        
        # Extract position
        if 'joint_state' in robot_obs:
            position = robot_obs['joint_state']['position'][:2]
        else:
            position = np.zeros(2)
        
        # Extract orientation (theta)
        if 'joint_state' in robot_obs and len(robot_obs['joint_state']['position']) > 2:
            theta = robot_obs['joint_state']['position'][2]
        else:
            theta = 0.0
        
        return np.array([position[0], position[1], theta])
    
    def get_history(self) -> dict:
        """
        Get simulation history.
        
        Returns
        -------
        dict
            Dictionary containing:
            - time: List of time stamps
            - states: List of state vectors
            - inputs: List of control inputs
            - predicted_trajectories: List of predicted trajectories
        """
        return {
            'time': np.array(self.history['time']),
            'states': np.array(self.history['states']),
            'inputs': np.array(self.history['inputs']),
            'predicted_trajectories': self.history['predicted_trajectories'],
            'target': self.target_pose
        }
    
    def close(self):
        """Close environment and cleanup resources."""
        if hasattr(self, 'env'):
            self.env.close()
        print("✓ Simulation closed")