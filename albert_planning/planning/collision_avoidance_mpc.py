"""
Collision Avoidance MPC Controller

This module extends the BaseMPC controller with collision avoidance capabilities.
It supports both hard constraints (distance constraints) and soft constraints
(barrier functions in the cost).

Obstacle types supported:
- Circle/Cylinder: ||p - p_obs|| >= r_obs + r_robot + margin
- Box: Uses bounding circle approximation or half-plane constraints
"""
import numpy as np
import casadi as cs
from typing import Optional, Tuple
from mpc_planning import BaseMPC


class CollisionAvoidanceMPC(BaseMPC):
    """
    MPC Controller with collision avoidance constraints.
    
    This extends BaseMPC to add obstacle avoidance using obstacles
    already registered in the environment.
    
    Parameters
    ----------
    dynamics : DifferentialDriveDynamics
        Robot dynamics model
    N : int
        Prediction horizon
    x_target : np.ndarray
        Target state [x_des, y_des, theta_des]
    wx : float
        State tracking weight
    wu : float
        Input weight
    env : UrdfEnv or BarEnvironment
        Environment containing obstacles (accessed via env.get_obstacles())
    robot_radius : float
        Robot bounding radius for collision checking
    safety_margin : float
        Additional safety margin beyond robot radius
    use_soft_constraints : bool
        If True, use barrier functions instead of hard constraints
    soft_constraint_weight : float
        Weight for soft constraint penalty
    constraint_every_n_steps : int
        Apply constraints every N steps (reduces computation)
    SOLVER_MAX_ITER : int
        Maximum solver iterations for subsequent solves
    DO_WARM_START : bool
        Enable warm start
    """
    
    def __init__(
        self,
        dynamics,
        N: int,
        x_target: np.ndarray,
        wx: float,
        wu: float,
        env,  # BarEnvironment or UrdfEnv with obstacles
        robot_radius: float = 0.35,
        safety_margin: float = 0.1,
        use_soft_constraints: bool = False,
        soft_constraint_weight: float = 100.0,
        constraint_every_n_steps: int = 1,
        SOLVER_MAX_ITER: int = 10,
        DO_WARM_START: bool = True
    ):
        """Initialize Collision Avoidance MPC."""
        # Store environment reference
        self.env = env
        
        # Store collision avoidance parameters
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.use_soft_constraints = use_soft_constraints
        self.soft_constraint_weight = soft_constraint_weight
        self.constraint_every_n_steps = constraint_every_n_steps
        
        # Call parent constructor (creates optimization problem with CA)
        super().__init__(dynamics, N, x_target, wx, wu, SOLVER_MAX_ITER, DO_WARM_START)
        
        # Print CA configuration
        num_obstacles = len(self.env.get_obstacles())
        print(f"\n✓ Collision Avoidance MPC initialized:")
        print(f"  Robot radius:      {robot_radius} m")
        print(f"  Safety margin:     {safety_margin} m")
        print(f"  Total clearance:   {robot_radius + safety_margin} m")
        print(f"  Obstacles tracked: {num_obstacles}")
        print(f"  Constraint type:   {'Soft' if use_soft_constraints else 'Hard'}")
        if use_soft_constraints:
            print(f"  Soft weight:       {soft_constraint_weight}")
    
    def _create_optimization_problem(self):
        """
        Create CasADi optimization problem with collision avoidance.
        
        Overrides parent method to add obstacle constraints from environment.
        """
        opti = cs.Opti()
        
        nx = self.dynamics.state_dim
        nu = self.dynamics.input_dim
        N = self.N
        
        # Create decision variables
        X = []
        U = []
        
        for k in range(N + 1):
            X.append(opti.variable(nx))
        
        for k in range(N):
            U.append(opti.variable(nu))
            # Input constraints
            opti.subject_to(opti.bounded(
                self.dynamics.u_min, U[-1], self.dynamics.u_max
            ))
        
        # Parameter for initial condition
        p_x_init = opti.parameter(nx)
        opti.subject_to(X[0] == p_x_init)
        
        # Build cost function
        cost = 0
        
        for k in range(N):
            # Tracking cost (position only)
            pos_error = X[k][:2] - self.x_target[:2]
            cost += self.wx * (pos_error.T @ pos_error)
            
            # Input cost
            cost += self.wu * (U[k].T @ U[k])
            
            # Dynamics constraint
            opti.subject_to(
                X[k + 1] == self.dynamics.discrete_dynamics(X[k], U[k])
            )
            
            # State box constraints
            opti.subject_to(opti.bounded(
                self.dynamics.x_min[0], X[k][0], self.dynamics.x_max[0]
            ))
            opti.subject_to(opti.bounded(
                self.dynamics.x_min[1], X[k][1], self.dynamics.x_max[1]
            ))
        
        # Terminal cost
        terminal_pos_error = X[-1][:2] - self.x_target[:2]
        cost += 10.0 * (terminal_pos_error.T @ terminal_pos_error)
        
        # =====================================================================
        # COLLISION AVOIDANCE - Use obstacles from environment
        # =====================================================================
        obstacles = self.env.get_obstacles()
        
        if len(obstacles) > 0:
            total_clearance = self.robot_radius + self.safety_margin
            
            for k in range(N + 1):
                # Apply constraints only every N steps to reduce computation
                if k % self.constraint_every_n_steps != 0:
                    continue
                
                for obst_id, obst in obstacles.items():
                    # Skip walls (handled by state bounds)
                    obst_name = str(obst.name()).lower()
                    if 'wall' in obst_name:
                        continue
                    
                    # Robot position at timestep k
                    robot_pos = X[k][:2]  # [x, y]
                    
                    # Obstacle position (from mpscenes obstacle)
                    obs_pos_full = obst.position()  # Returns [x, y, z]
                    obs_pos = cs.vertcat(obs_pos_full[0], obs_pos_full[1])  # Only [x, y]
                    
                    # Get obstacle size for minimum distance
                    obs_size = obst.size()  # Returns obstacle dimensions
                    
                    # Compute bounding radius based on obstacle type
                    if obst.type() == 'cylinder':
                        # For cylinder: radius is directly available
                        obs_radius = obs_size[0] if len(obs_size) > 0 else 0.5
                    elif obst.type() == 'box':
                        # For box: use half diagonal as bounding radius
                        # obs_size is [length, width, height]
                        obs_radius = cs.sqrt(obs_size[0]**2 + obs_size[1]**2) / 2
                    else:
                        # Default: assume sphere/circle
                        obs_radius = obs_size[0] if len(obs_size) > 0 else 0.5
                    
                    # Distance from robot to obstacle center
                    dx = robot_pos[0] - obs_pos[0]
                    dy = robot_pos[1] - obs_pos[1]
                    dist_squared = dx**2 + dy**2
                    
                    # Minimum distance required
                    min_dist = obs_radius + total_clearance
                    min_dist_squared = min_dist**2
                    
                    if self.use_soft_constraints:
                        # Soft constraint: barrier function in cost
                        epsilon = 0.01  # Numerical stability
                        penalty = self.soft_constraint_weight / (
                            dist_squared - min_dist_squared + epsilon
                        )
                        cost += penalty
                    else:
                        # Hard constraint: dist_squared >= min_dist_squared
                        opti.subject_to(dist_squared >= min_dist_squared)
        
        # Minimize cost
        opti.minimize(cost)
        
        # Solver configuration
        print("Creating Collision Avoidance MPC optimization problem...")
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 3000,            # High for initial solve
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 10,
            "ipopt.mu_strategy": "adaptive",   # Better for CA problems
        }
        opti.solver("ipopt", opts)
        
        # Initial solve
        print("Solving initial collision avoidance MPC problem...")
        opti.set_value(p_x_init, np.zeros(nx))
        
        # Set good initial guess
        for k in range(N + 1):
            opti.set_initial(X[k], np.zeros(nx))
        for k in range(N):
            opti.set_initial(U[k], np.zeros(nu))
        
        try:
            opti.solve()
            print("✓ Initial CA-MPC problem solved successfully!")
        except Exception as e:
            print(f"⚠ Warning: Initial solve failed: {e}")
            print("  Continuing with default initialization...")
        
        # Reconfigure for faster subsequent solves
        opts["ipopt.max_iter"] = max(self.max_iter, 50)
        opti.solver("ipopt", opts)
        
        return opti, X, U, p_x_init
    
    def get_min_obstacle_distance(self, x: np.ndarray) -> Tuple[float, str]:
        """
        Compute minimum distance from robot to any obstacle.
        
        Parameters
        ----------
        x : np.ndarray
            Robot state [x, y, theta]
        
        Returns
        -------
        Tuple[float, str]
            (min_distance, obstacle_name)
        """
        obstacles = self.env.get_obstacles()
        
        if len(obstacles) == 0:
            return float('inf'), ""
        
        robot_pos = x[:2]
        min_dist = float('inf')
        min_obs_name = ""
        
        for obst_id, obst in obstacles.items():
            # Skip walls
            obst_name = str(obst.name())
            if 'wall' in obst_name.lower():
                continue
            
            # Get obstacle position and size
            obs_pos = obst.position()[:2]
            obs_size = obst.size()
            
            # Compute bounding radius
            if obst.type() == 'cylinder':
                obs_radius = obs_size[0]
            elif obst.type() == 'box':
                obs_radius = np.sqrt(obs_size[0]**2 + obs_size[1]**2) / 2
            else:
                obs_radius = obs_size[0]
            
            # Distance from robot to obstacle surface
            dist = np.linalg.norm(robot_pos - obs_pos) - obs_radius
            
            if dist < min_dist:
                min_dist = dist
                min_obs_name = obst_name
        
        return min_dist, min_obs_name
    
    def update_obstacles(self):
        """
        Refresh obstacles from environment and rebuild optimization problem.
        
        Call this if obstacles change dynamically during simulation.
        """
        self.opti, self.X_var, self.U_var, self.p_x_init = (
            self._create_optimization_problem()
        )
        num_obstacles = len(self.env.get_obstacles())
        print(f"✓ CA-MPC updated with {num_obstacles} obstacles")