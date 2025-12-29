"""
Discrete Action Controller for Differential Drive Robot

Simple "rotate-then-translate" controller with three discrete actions:
1. Rotate left (in place)
2. Rotate right (in place)
3. Go straight

This avoids the complexity of continuous MPC optimization and produces
more predictable behavior for waypoint following.
"""

import numpy as np
from typing import Tuple, List, Optional


class DiscreteController:
    """
    Discrete action controller for waypoint following.

    Actions:
        - ROTATE_LEFT:  (0, +omega_max) - rotate counter-clockwise
        - ROTATE_RIGHT: (0, -omega_max) - rotate clockwise
        - GO_STRAIGHT:  (v_max, 0)      - move forward
    """

    def __init__(self,
                 v_max: float = 0.5,
                 omega_max: float = 1.0,
                 heading_threshold: float = 0.15,  # ~8.5 degrees
                 obstacles: List[dict] = None,
                 robot_radius: float = 0.35,
                 safety_margin: float = 0.15):
        """
        Args:
            v_max: Maximum forward velocity
            omega_max: Maximum angular velocity
            heading_threshold: Angle error below which robot goes straight (radians)
            obstacles: List of obstacles for collision checking
            robot_radius: Robot radius for collision checking
            safety_margin: Safety margin for obstacles
        """
        self.v_max = v_max
        self.omega_max = omega_max
        self.heading_threshold = heading_threshold

        self.obstacles = obstacles or []
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.min_clearance = robot_radius + safety_margin

        # Current waypoint
        self.target = np.array([0., 0.])

    def update_target(self, target: np.ndarray):
        """Update the current target waypoint."""
        self.target = np.array(target[:2])

    def get_heading_error(self, x: np.ndarray) -> float:
        """
        Calculate heading error to target.

        Returns angle in [-pi, pi] that robot needs to turn.
        Positive = need to turn left, Negative = need to turn right.
        """
        dx = self.target[0] - x[0]
        dy = self.target[1] - x[1]

        # Desired heading
        desired_heading = np.arctan2(dy, dx)

        # Current heading
        current_heading = x[2]

        # Heading error (wrapped to [-pi, pi])
        error = desired_heading - current_heading
        error = np.arctan2(np.sin(error), np.cos(error))

        return error

    def get_min_obstacle_distance(self, x: np.ndarray) -> Tuple[float, str]:
        """Get distance to nearest obstacle."""
        min_dist = float('inf')
        min_name = "none"

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                dist = np.linalg.norm(x[:2] - np.array(obs['position'][:2])) - obs.get('radius', 0.3)
            else:  # box
                cx, cy = obs['position'][:2]
                size = obs.get('size', [0.5, 0.5])
                half_x, half_y = size[0]/2, size[1]/2

                # Distance to box edges
                dx = max(abs(x[0] - cx) - half_x, 0)
                dy = max(abs(x[1] - cy) - half_y, 0)
                dist = np.sqrt(dx**2 + dy**2)

            if dist < min_dist:
                min_dist = dist
                min_name = obs.get('name', 'obstacle')

        return min_dist, min_name

    def check_forward_clear(self, x: np.ndarray, look_ahead: float = 0.5) -> bool:
        """Check if path ahead is clear of obstacles."""
        # Position ahead in current heading direction
        ahead_x = x[0] + look_ahead * np.cos(x[2])
        ahead_y = x[1] + look_ahead * np.sin(x[2])
        ahead = np.array([ahead_x, ahead_y, x[2]])

        min_dist, _ = self.get_min_obstacle_distance(ahead)
        return min_dist > self.min_clearance

    def solve(self, x: np.ndarray) -> Tuple[np.ndarray, None, None, None]:
        """
        Get control action based on current state.

        Returns (u, None, None, None) to match MPC interface.
        u = [v, omega]
        """
        # Get heading error
        heading_error = self.get_heading_error(x)

        # Check distance to target
        dist_to_target = np.linalg.norm(self.target - x[:2])

        # Check if forward path is clear
        forward_clear = self.check_forward_clear(x)

        # Decision logic
        if dist_to_target < 0.1:
            # Very close to target - stop
            u = np.array([0.0, 0.0])
        elif abs(heading_error) > self.heading_threshold:
            # Need to rotate first
            if heading_error > 0:
                # Turn left
                u = np.array([0.0, self.omega_max])
            else:
                # Turn right
                u = np.array([0.0, -self.omega_max])
        elif forward_clear:
            # Aligned and clear - go straight
            # Reduce speed when close to target
            speed = min(self.v_max, dist_to_target)
            u = np.array([speed, 0.0])
        else:
            # Obstacle ahead - rotate to find clear path
            # Prefer rotating in direction of heading error
            if heading_error >= 0:
                u = np.array([0.0, self.omega_max])
            else:
                u = np.array([0.0, -self.omega_max])

        # Return in MPC-compatible format
        return u, None, None, None


class DiscreteControllerWithAvoidance(DiscreteController):
    """
    Extended discrete controller with reactive obstacle avoidance.

    When an obstacle is too close, the robot will:
    1. Stop forward motion
    2. Rotate away from the obstacle
    3. Use hysteresis to prevent oscillation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avoidance_distance = self.min_clearance + 0.3  # Start avoiding at this distance
        self.clear_distance = self.min_clearance + 0.5  # Must be this far to exit avoidance

        # State for hysteresis
        self.avoiding = False
        self.avoidance_direction = 0  # +1 = rotating left, -1 = rotating right
        self.avoidance_steps = 0
        self.min_avoidance_steps = 10  # Minimum steps to commit to avoidance

    def solve(self, x: np.ndarray) -> Tuple[np.ndarray, None, None, None]:
        """Get control with obstacle avoidance and hysteresis."""
        # Check nearest obstacle
        min_dist, obs_name = self.get_min_obstacle_distance(x)

        # Check if we should exit avoidance mode
        if self.avoiding:
            self.avoidance_steps += 1

            # Only exit if we've done minimum steps AND we're far enough from obstacles
            if self.avoidance_steps >= self.min_avoidance_steps and min_dist > self.clear_distance:
                self.avoiding = False
                self.avoidance_direction = 0
                self.avoidance_steps = 0
            else:
                # Continue avoidance in committed direction
                return np.array([0.0, self.avoidance_direction * self.omega_max]), None, None, None

        # Check if we should enter avoidance mode
        if min_dist < self.avoidance_distance:
            # Check if forward path is blocked
            if not self.check_forward_clear(x, look_ahead=0.4):
                # Enter avoidance mode
                self.avoiding = True
                self.avoidance_steps = 0

                # Decide direction: rotate towards target if possible, otherwise away from obstacle
                heading_error = self.get_heading_error(x)
                obstacle_angle = self._get_obstacle_direction(x, obs_name)

                if obstacle_angle is not None:
                    # Rotate away from obstacle
                    if obstacle_angle > 0:
                        self.avoidance_direction = -1  # Obstacle on left, rotate right
                    else:
                        self.avoidance_direction = 1   # Obstacle on right, rotate left
                else:
                    # No clear obstacle direction, rotate towards target
                    self.avoidance_direction = 1 if heading_error > 0 else -1

                return np.array([0.0, self.avoidance_direction * self.omega_max]), None, None, None

        # Normal behavior - rotate then translate
        return super().solve(x)

    def _get_obstacle_direction(self, x: np.ndarray, obs_name: str) -> Optional[float]:
        """Get angle to obstacle relative to robot heading."""
        for obs in self.obstacles:
            if obs.get('name', '') == obs_name:
                obs_pos = np.array(obs['position'][:2])
                dx = obs_pos[0] - x[0]
                dy = obs_pos[1] - x[1]

                # Angle to obstacle in world frame
                angle_to_obs = np.arctan2(dy, dx)

                # Relative to robot heading
                relative_angle = angle_to_obs - x[2]
                relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

                return relative_angle

        return None
