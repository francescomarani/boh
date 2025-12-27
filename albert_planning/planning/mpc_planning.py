"""
Model Predictive Control (MPC) for differential drive robot.

This module implements a basic MPC controller using CasADi optimization
for trajectory tracking with a differential drive robot.
"""

import numpy as np
import casadi as cs
from typing import Tuple
from models import DifferentialDriveDynamics


class BaseMPC:
    """
    MPC Controller for differential drive robot.
    
    Solves a finite-horizon optimal control problem to track a target position
    while respecting state and input constraints.
    """
    
    def __init__(
        self,
        dynamics: DifferentialDriveDynamics,
        N: int,
        x_target: np.ndarray,
        wx: float,
        wu: float,
        SOLVER_MAX_ITER: int = 10,
        DO_WARM_START: bool = True
    ):
        """
        Initialize MPC controller.
        
        Parameters
        ----------
        dynamics : DifferentialDriveDynamics
            Differential drive dynamics model
        N : int
            Prediction horizon (number of time steps)
        x_target : np.ndarray
            Target state [x_des, y_des, theta_des]
        wx : float
            State tracking weight (penalizes position error)
        wu : float
            Input weight (penalizes control effort)
        SOLVER_MAX_ITER : int
            Maximum iterations for solver (after initial solve)
        DO_WARM_START : bool
            Enable warm start for faster solving
        """
        self.dynamics = dynamics
        self.N = N
        self.x_target = x_target
        self.wx = wx
        self.wu = wu
        self.max_iter = SOLVER_MAX_ITER
        self.warm_start = DO_WARM_START
        
        # Create the optimization problem (done once at initialization)
        self.opti, self.X_var, self.U_var, self.p_x_init = (
            self._create_optimization_problem()
        )
    
    def _create_optimization_problem(self) -> Tuple:
        """
        Create CasADi optimization problem structure.
        
        This method sets up:
        - Decision variables (states and inputs)
        - Cost function (tracking + input)
        - Constraints (dynamics, bounds)
        - Solver configuration
        
        Returns
        -------
        Tuple
            (opti, X_var, U_var, p_x_init) containing:
            - opti: CasADi Opti object
            - X_var: List of state variables
            - U_var: List of input variables
            - p_x_init: Parameter for initial condition
        """
        print("Creating MPC optimization problem...")
        
        opti = cs.Opti()
        
        nx = self.dynamics.state_dim
        nu = self.dynamics.input_dim
        N = self.N
        
        # Decision variables for states and inputs
        X = []  # States: X[0], X[1], ..., X[N]
        U = []  # Inputs: U[0], U[1], ..., U[N-1]
        
        # Create state variables
        for k in range(N + 1):
            X.append(opti.variable(nx))
        
        # Create input variables with constraints
        for k in range(N):
            U.append(opti.variable(nu))
            # Input box constraints
            opti.subject_to(opti.bounded(
                self.dynamics.u_min, U[-1], self.dynamics.u_max
            ))
        
        # Parameter for initial condition (set at each solve)
        p_x_init = opti.parameter(nx)
        opti.subject_to(X[0] == p_x_init)
        
        # Build cost function and dynamics constraints
        cost = 0
        
        for k in range(N):
            # Tracking cost (position only: [x, y])
            pos_error = X[k][:2] - self.x_target[:2]
            cost += self.wx * (pos_error.T @ pos_error)
            
            # Input cost (penalize control effort)
            cost += self.wu * (U[k].T @ U[k])
            
            # Dynamics constraint: X[k+1] = f(X[k], U[k])
            opti.subject_to(
                X[k + 1] == self.dynamics.discrete_dynamics(X[k], U[k])
            )
            
            # State box constraints (position only)
            opti.subject_to(opti.bounded(
                self.dynamics.x_min[0], X[k][0], self.dynamics.x_max[0]
            ))
            opti.subject_to(opti.bounded(
                self.dynamics.x_min[1], X[k][1], self.dynamics.x_max[1]
            ))
        
        # Terminal cost (higher weight for final state)
        terminal_pos_error = X[-1][:2] - self.x_target[:2]
        cost += 10.0 * (terminal_pos_error.T @ terminal_pos_error)
        
        # Set objective
        opti.minimize(cost)
        
        # Configure IPOPT solver
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000,           # High limit for initial solve
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,       # Accept "good enough" solutions
        }
        opti.solver("ipopt", opts)
        
        # Solve once to convergence (initialization)
        print("Solving initial MPC problem to convergence...")
        opti.set_value(p_x_init, np.zeros(nx))
        
        try:
            opti.solve()
            print("✓ Initial MPC problem solved successfully!")
        except Exception as e:
            print(f"✗ Warning: Initial solve failed: {e}")
            print("  Continuing anyway - solver may take longer on first real solve.")
        
        # Reconfigure solver for faster subsequent solves
        opts["ipopt.max_iter"] = self.max_iter
        opti.solver("ipopt", opts)
        
        return opti, X, U, p_x_init
    
    def solve(self, x_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solve MPC problem for current state.
        
        Parameters
        ----------
        x_init : np.ndarray
            Current state [x, y, theta] (3,)
        
        Returns
        -------
        Tuple containing:
            u_opt : np.ndarray
                Optimal input [v, omega] (2,)
            x_next : np.ndarray
                Predicted next state (3,)
            x_traj : np.ndarray
                Predicted state trajectory (3, N+1)
            theta_next : float
                Predicted next orientation
        
        Raises
        ------
        ValueError
            If x_init doesn't have exactly 3 elements
        """
        # Validate input
        if x_init.shape[0] != 3:
            raise ValueError(
                f"Expected x_init to have 3 elements [x, y, theta], "
                f"got {x_init.shape[0]}"
            )
        
        try:
            # Set initial condition parameter
            self.opti.set_value(self.p_x_init, x_init)
            
            # Solve optimization problem
            sol = self.opti.solve()
            
            # Warm start for next iteration (shifting strategy)
            if self.warm_start:
                self._apply_warm_start(sol)
            
            # Extract optimal control and trajectory
            u_opt = sol.value(self.U_var[0])
            x_next = sol.value(self.X_var[1])
            
            # Extract full predicted trajectory
            x_traj = np.zeros((3, self.N + 1))
            for k in range(self.N + 1):
                x_traj[:, k] = sol.value(self.X_var[k])
            
            return u_opt, x_next, x_traj, x_next[2]
        
        except Exception as e:
            print(f"✗ MPC solver failed: {e}")
            return self._safe_fallback(x_init)
    
    def _apply_warm_start(self, sol):
        """
        Apply warm start by shifting solution for next iteration.
        
        Strategy: Use previous solution shifted by one time step as
        initial guess for next solve.
        
        Parameters
        ----------
        sol : CasADi solution object
            Solution from previous solve
        """
        # Shift states: X[k] initial guess = previous solution X[k+1]
        for k in range(self.N):
            self.opti.set_initial(
                self.X_var[k],
                sol.value(self.X_var[k + 1])
            )
        
        # Shift inputs: U[k] initial guess = previous solution U[k+1]
        for k in range(self.N - 1):
            self.opti.set_initial(
                self.U_var[k],
                sol.value(self.U_var[k + 1])
            )
        
        # Keep last state and input the same
        self.opti.set_initial(self.X_var[-1], sol.value(self.X_var[-1]))
        self.opti.set_initial(self.U_var[-1], sol.value(self.U_var[-1]))
    
    def _safe_fallback(self, x_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Return safe default values when solver fails.
        
        Parameters
        ----------
        x_init : np.ndarray
            Current state
        
        Returns
        -------
        Tuple
            Safe default values (zero input, current state maintained)
        """
        # Zero input (stop the robot)
        u_safe = np.zeros(2)
        
        # Maintain current state
        x_traj = np.tile(x_init.reshape(-1, 1), (1, self.N + 1))
        
        # Ensure correct state dimension
        if len(x_init) == 3:
            return u_safe, x_init, x_traj, x_init[2]
        else:
            # Emergency fallback with zeros
            x_init_safe = np.zeros(3)
            x_traj_safe = np.zeros((3, self.N + 1))
            return u_safe, x_init_safe, x_traj_safe, 0.0
    
    def update_target(self, x_target: np.ndarray):
        """
        Update target state.
        
        Note: This requires rebuilding the optimization problem.
        For dynamic target updates, consider using a parameter instead.
        
        Parameters
        ----------
        x_target : np.ndarray
            New target state [x, y, theta]
        """
        self.x_target = x_target
        print("Warning: Target updated. Consider rebuilding MPC for best performance.")
    
    def __repr__(self) -> str:
        """String representation of MPC controller."""
        return (
            f"BaseMPC(\n"
            f"  horizon={self.N},\n"
            f"  target={self.x_target},\n"
            f"  weights=(wx={self.wx}, wu={self.wu}),\n"
            f"  max_iter={self.max_iter},\n"
            f"  warm_start={self.warm_start}\n"
            f")"
        )