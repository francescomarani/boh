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
from typing import List, Dict, Optional, Tuple
from mpc_planning import BaseMPC


class Obstacle:
    """
    Obstacle representation for MPC collision avoidance.

    Attributes:
        name: Obstacle identifier
        type: 'circle' or 'box'
        position: [x, y] center position
        radius: Radius for circle obstacles
        size: [length, width] for box obstacles
    """

    def __init__(self, name: str, obs_type: str, position: np.ndarray,
                 radius: float = None, size: np.ndarray = None):
        self.name = name
        self.type = obs_type
        self.position = np.array(position[:2])  # Only x, y
        self.radius = radius
        self.size = np.array(size) if size is not None else None

        # For box obstacles, compute bounding circle radius
        if self.type == 'box' and self.size is not None:
            # Bounding circle: half diagonal
            self.bounding_radius = np.sqrt(self.size[0]**2 + self.size[1]**2) / 2
        else:
            self.bounding_radius = self.radius

    def __repr__(self):
        return f"Obstacle({self.name}, {self.type}, pos={self.position}, r={self.bounding_radius:.2f})"


class CollisionAvoidanceMPC(BaseMPC):
    """
    MPC Controller with collision avoidance constraints.

    This extends BaseMPC to add:
    - Circular obstacle avoidance constraints
    - Box obstacle avoidance (using bounding circles)
    - Optional soft constraints (barrier functions)
    - Configurable safety margins

    Args:
        dynamics: DifferentialDriveDynamics object
        N: Prediction horizon
        x_target: Target state [x_des, y_des, theta_des]
        wx: State tracking weight
        wu: Input weight
        obstacles: List of Obstacle objects
        robot_radius: Robot bounding radius for collision checking
        safety_margin: Additional safety margin
        use_soft_constraints: If True, add barrier functions to cost instead of hard constraints
        soft_constraint_weight: Weight for soft constraint penalty
        constraint_every_n_steps: Apply constraints every N steps (reduces computation)
        SOLVER_MAX_ITER: Max iterations for subsequent solves
        DO_WARM_START: Enable warm start
    """

    def __init__(self, dynamics, N: int, x_target: np.ndarray,
                 wx: float, wu: float,
                 obstacles: List[Obstacle] = None,
                 robot_radius: float = 0.35,
                 safety_margin: float = 0.1,
                 use_soft_constraints: bool = False,
                 soft_constraint_weight: float = 100.0,
                 constraint_every_n_steps: int = 1,
                 SOLVER_MAX_ITER: int = 10,
                 DO_WARM_START: bool = True):

        # Store collision avoidance parameters before calling parent __init__
        self.obstacles = obstacles if obstacles is not None else []
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.use_soft_constraints = use_soft_constraints
        self.soft_constraint_weight = soft_constraint_weight
        self.constraint_every_n_steps = constraint_every_n_steps

        # Call parent constructor (this creates the optimization problem)
        super().__init__(dynamics, N, x_target, wx, wu, SOLVER_MAX_ITER, DO_WARM_START)

        print(f"\nCollision Avoidance MPC initialized:")
        print(f"  Robot radius: {robot_radius}m")
        print(f"  Safety margin: {safety_margin}m")
        print(f"  Total clearance: {robot_radius + safety_margin}m")
        print(f"  Number of obstacles: {len(self.obstacles)}")
        print(f"  Soft constraints: {use_soft_constraints}")
        if use_soft_constraints:
            print(f"  Soft constraint weight: {soft_constraint_weight}")

    def _create_optimization_problem(self):
        """
        Create CasADi optimization problem with collision avoidance constraints.

        Overrides parent method to add obstacle constraints.
        """
        opti = cs.Opti()

        nx = self.dynamics.state_dim
        nu = self.dynamics.input_dim
        N = self.N

        # Create all decision variables for state and control
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

        # Parameter for initial condition (will be set at each MPC call)
        p_x_init = opti.parameter(nx)
        opti.subject_to(X[0] == p_x_init)

        # Add cost function and dynamics constraints
        cost = 0

        for k in range(N):
            # Tracking cost (only position [x, y])
            pos_error = X[k][:2] - self.x_target[:2]
            cost += self.wx * pos_error.T @ pos_error

            # Input cost
            cost += self.wu * U[k].T @ U[k]

            # Dynamics constraint: X[k+1] = X[k] + dt * f(X[k], U[k])
            opti.subject_to(
                X[k + 1] == self.dynamics.discrete_dynamics(X[k], U[k])
            )

            # State constraints (optional, on position)
            opti.subject_to(opti.bounded(
                self.dynamics.x_min[0], X[k][0], self.dynamics.x_max[0]
            ))
            opti.subject_to(opti.bounded(
                self.dynamics.x_min[1], X[k][1], self.dynamics.x_max[1]
            ))

        # Terminal cost
        terminal_pos_error = X[-1][:2] - self.x_target[:2]
        cost += 10.0 * terminal_pos_error.T @ terminal_pos_error

        # =====================================================================
        # COLLISION AVOIDANCE CONSTRAINTS
        # =====================================================================
        if len(self.obstacles) > 0:
            total_clearance = self.robot_radius + self.safety_margin

            # Add constraints for each timestep and each obstacle
            for k in range(N + 1):
                # Only add constraints every N steps to reduce computation
                if k % self.constraint_every_n_steps != 0:
                    continue

                for obs in self.obstacles:
                    # Robot position at timestep k
                    robot_pos = X[k][:2]  # [x, y]
                    obs_pos = obs.position  # [x, y]

                    if obs.type == 'circle':
                        # Circular obstacle: use Euclidean distance
                        dx = robot_pos[0] - obs_pos[0]
                        dy = robot_pos[1] - obs_pos[1]
                        dist_squared = dx**2 + dy**2
                        dist = cs.sqrt(dist_squared + 1e-6)  # Smooth sqrt
                        min_dist = obs.radius + total_clearance

                        # HYBRID: Always add hard constraint with reduced clearance
                        hard_clearance = obs.radius + self.robot_radius + 0.05
                        opti.subject_to(dist_squared >= hard_clearance**2)

                        if self.use_soft_constraints:
                            # Exponential penalty: gives "vision" of obstacles
                            # Fast decay (5.0) = only affects nearby obstacles
                            # Low weight = avoids strong local minima
                            decay_rate = 5.0  # Fast decay - only nearby obstacles matter
                            penalty = self.soft_constraint_weight * cs.exp(
                                -decay_rate * (dist - min_dist)
                            )
                            cost += penalty

                    else:  # Box obstacle
                        # Proper signed distance to axis-aligned box
                        half_x = obs.size[0] / 2.0
                        half_y = obs.size[1] / 2.0

                        # Distance from robot to box edges
                        dx = cs.fabs(robot_pos[0] - obs_pos[0]) - half_x
                        dy = cs.fabs(robot_pos[1] - obs_pos[1]) - half_y

                        # Signed distance to box
                        dist_outside_sq = cs.fmax(dx, 0)**2 + cs.fmax(dy, 0)**2
                        dist_outside = cs.sqrt(dist_outside_sq + 1e-6)
                        dist_inside = cs.fmin(cs.fmax(dx, dy), 0)
                        signed_dist = dist_outside + dist_inside

                        # HYBRID: Always add hard constraint with reduced clearance
                        hard_clearance = self.robot_radius + 0.05
                        opti.subject_to(dist_outside_sq >= hard_clearance**2)

                        if self.use_soft_constraints:
                            # Exponential penalty for boxes
                            # Fast decay = only nearby obstacles matter
                            decay_rate = 5.0
                            penalty = self.soft_constraint_weight * cs.exp(
                                -decay_rate * (signed_dist - total_clearance)
                            )
                            cost += penalty

        # Minimize cost
        opti.minimize(cost)

        # Create the optimization solver with high max_iter for first solve
        print("Creating Collision Avoidance MPC optimization problem...")
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 2000,  # More iterations due to collision constraints
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,
        }
        opti.solver("ipopt", opts)

        # Solve the problem to convergence the first time
        print("Solving initial collision avoidance MPC problem...")
        opti.set_value(p_x_init, np.zeros(nx))

        # Set good initial guess to help convergence with obstacles
        for k in range(N + 1):
            opti.set_initial(X[k], np.zeros(nx))
        for k in range(N):
            opti.set_initial(U[k], np.zeros(nu))

        try:
            opti.solve()
            print("Initial collision avoidance MPC problem solved!")
        except Exception as e:
            print(f"Warning: Initial solve failed: {e}")
            print("Continuing with default initialization...")

        # Set solver options for subsequent solves
        # Use higher max_iter for collision avoidance (more complex problem)
        # With many obstacles, need more iterations to converge
        opts["ipopt.max_iter"] = max(self.max_iter, 200)
        opti.solver("ipopt", opts)

        return opti, X, U, p_x_init

    def update_obstacles(self, obstacles: List[Obstacle]) -> None:
        """
        Update obstacle list and recreate the optimization problem.

        This should be called when obstacles change (e.g., dynamic obstacles).

        Args:
            obstacles: New list of Obstacle objects
        """
        self.obstacles = obstacles
        self.opti, self.X_var, self.U_var, self.p_x_init = self._create_optimization_problem()
        print(f"Updated MPC with {len(obstacles)} obstacles")

    def get_min_obstacle_distance(self, x: np.ndarray) -> Tuple[float, str]:
        """
        Compute minimum distance from robot position to any obstacle.

        Args:
            x: Robot state [x, y, theta]

        Returns:
            Tuple of (min_distance, obstacle_name)
        """
        if len(self.obstacles) == 0:
            return float('inf'), ""

        robot_pos = x[:2]
        min_dist = float('inf')
        min_obs_name = ""

        for obs in self.obstacles:
            if obs.type == 'circle':
                dist = np.linalg.norm(robot_pos - obs.position) - obs.radius
            else:  # Box
                # Signed distance to axis-aligned box
                half_x = obs.size[0] / 2.0
                half_y = obs.size[1] / 2.0
                dx = abs(robot_pos[0] - obs.position[0]) - half_x
                dy = abs(robot_pos[1] - obs.position[1]) - half_y
                # Distance outside box
                dist_outside = np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                # Distance inside box (negative)
                dist_inside = min(max(dx, dy), 0)
                dist = dist_outside + dist_inside

            if dist < min_dist:
                min_dist = dist
                min_obs_name = obs.name

        return min_dist, min_obs_name


def obstacles_from_dict_list(obstacle_dicts: List[dict]) -> List[Obstacle]:
    """
    Convert a list of obstacle dictionaries to Obstacle objects.

    This is used to convert the output of BarEnvironment.get_mpc_obstacles()
    to Obstacle objects for the MPC controller.

    Args:
        obstacle_dicts: List of dicts with keys 'name', 'type', 'position',
                       and 'radius' (for circles) or 'size' (for boxes)

    Returns:
        List of Obstacle objects
    """
    obstacles = []

    for obs_dict in obstacle_dicts:
        obs_type = obs_dict.get('type', 'circle')
        obstacle = Obstacle(
            name=obs_dict['name'],
            obs_type=obs_type,
            position=np.array(obs_dict['position']),
            radius=obs_dict.get('radius') if obs_type == 'circle' else None,
            size=np.array(obs_dict['size']) if obs_type == 'box' else None
        )
        obstacles.append(obstacle)

    print(f"Converted {len(obstacles)} obstacles for MPC")
    return obstacles
