"""
A* Path Planner in Configuration Space

This module implements a grid-based A* path planner that:
1. Creates a 2D occupancy grid of the environment
2. Grows obstacles by robot radius + safety margin (C-space)
3. Finds collision-free path using A*
4. Returns waypoints for MPC to follow
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for the occupancy grid."""
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -10.0
    y_max: float = 10.0
    resolution: float = 0.1  # meters per cell


class ConfigurationSpacePlanner:
    """
    A* path planner in configuration space.

    The planner creates an occupancy grid where obstacles are "grown"
    by the robot radius + safety margin, allowing the robot to be
    treated as a point.
    """

    def __init__(self,
                 obstacles: List[dict],
                 robot_radius: float = 0.35,
                 safety_margin: float = 0.15,
                 grid_config: GridConfig = None):
        """
        Initialize the planner.

        Args:
            obstacles: List of obstacle dicts with 'type', 'position', 'radius'/'size'
            robot_radius: Robot bounding radius
            safety_margin: Additional safety margin
            grid_config: Grid configuration (uses defaults if None)
        """
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.clearance = robot_radius + safety_margin

        self.config = grid_config or GridConfig()

        # Create occupancy grid
        self._create_grid()
        self._mark_obstacles()

    def _create_grid(self):
        """Create the occupancy grid."""
        self.nx = int((self.config.x_max - self.config.x_min) / self.config.resolution)
        self.ny = int((self.config.y_max - self.config.y_min) / self.config.resolution)

        # 0 = free, 1 = occupied
        self.grid = np.zeros((self.nx, self.ny), dtype=np.uint8)

        print(f"Created grid: {self.nx}x{self.ny} cells, resolution={self.config.resolution}m")

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        ix = int((x - self.config.x_min) / self.config.resolution)
        iy = int((y - self.config.y_min) / self.config.resolution)
        return ix, iy

    def _grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.config.x_min + (ix + 0.5) * self.config.resolution
        y = self.config.y_min + (iy + 0.5) * self.config.resolution
        return x, y

    def _mark_obstacles(self):
        """Mark obstacles in the grid (grown by clearance)."""
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                self._mark_circle(obs)
            elif obs['type'] == 'box':
                self._mark_box(obs)

        occupied = np.sum(self.grid)
        total = self.nx * self.ny
        print(f"Marked {occupied} occupied cells ({100*occupied/total:.1f}% of grid)")

    def _mark_circle(self, obs: dict):
        """Mark a circular obstacle (grown by clearance)."""
        cx, cy = obs['position'][:2]
        radius = obs.get('radius', 0.3) + self.clearance

        # Find bounding box in grid coordinates
        ix_min, iy_min = self._world_to_grid(cx - radius, cy - radius)
        ix_max, iy_max = self._world_to_grid(cx + radius, cy + radius)

        # Clamp to grid bounds
        ix_min = max(0, ix_min)
        iy_min = max(0, iy_min)
        ix_max = min(self.nx - 1, ix_max)
        iy_max = min(self.ny - 1, iy_max)

        # Mark cells within radius
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                wx, wy = self._grid_to_world(ix, iy)
                dist = np.sqrt((wx - cx)**2 + (wy - cy)**2)
                if dist <= radius:
                    self.grid[ix, iy] = 1

    def _mark_box(self, obs: dict):
        """Mark a box obstacle (grown by clearance)."""
        cx, cy = obs['position'][:2]
        size = obs.get('size', [0.5, 0.5])

        # Grow the box by clearance
        half_x = size[0] / 2 + self.clearance
        half_y = size[1] / 2 + self.clearance

        # Find bounding box in grid coordinates
        ix_min, iy_min = self._world_to_grid(cx - half_x, cy - half_y)
        ix_max, iy_max = self._world_to_grid(cx + half_x, cy + half_y)

        # Clamp to grid bounds
        ix_min = max(0, ix_min)
        iy_min = max(0, iy_min)
        ix_max = min(self.nx - 1, ix_max)
        iy_max = min(self.ny - 1, iy_max)

        # Mark all cells in the box
        self.grid[ix_min:ix_max+1, iy_min:iy_max+1] = 1

    def is_valid(self, ix: int, iy: int) -> bool:
        """Check if a grid cell is valid and free."""
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return False
        return self.grid[ix, iy] == 0

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, ix: int, iy: int) -> List[Tuple[int, int, float]]:
        """Get valid neighbors of a cell with costs."""
        neighbors = []

        # 8-connected grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = ix + dx, iy + dy
                if self.is_valid(nx, ny):
                    # Diagonal moves cost sqrt(2), cardinal moves cost 1
                    cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1.0
                    neighbors.append((nx, ny, cost))

        return neighbors

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             simplify: bool = True) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using A*.

        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            simplify: If True, simplify the path to key waypoints

        Returns:
            List of waypoints [(x, y), ...] or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start[0], start[1])
        goal_grid = self._world_to_grid(goal[0], goal[1])

        # Check if start and goal are valid
        if not self.is_valid(*start_grid):
            print(f"Warning: Start position {start} is in obstacle!")
            return None
        if not self.is_valid(*goal_grid):
            print(f"Warning: Goal position {goal} is in obstacle!")
            return None

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                print(f"A* found path with {len(path)} cells")

                # Convert to world coordinates
                world_path = [self._grid_to_world(ix, iy) for ix, iy in path]

                if simplify:
                    world_path = self._simplify_path(world_path)
                    print(f"Simplified to {len(world_path)} waypoints")

                return world_path

            for nx, ny, cost in self.get_neighbors(*current):
                neighbor = (nx, ny)
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("A* failed to find a path!")
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from A* search."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _simplify_path(self, path: List[Tuple[float, float]],
                       min_distance: float = 0.5,
                       max_distance: float = 1.5) -> List[Tuple[float, float]]:
        """
        Simplify path by removing intermediate points.

        Uses line-of-sight checks to skip unnecessary waypoints.
        Adds intermediate points if segments are too long for MPC to track.

        Args:
            path: Full path from A*
            min_distance: Minimum distance between waypoints
            max_distance: Maximum distance between waypoints (add intermediates if exceeded)
        """
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to skip as many points as possible
            farthest_visible = current_idx + 1

            for i in range(current_idx + 2, len(path)):
                if self._line_of_sight(path[current_idx], path[i]):
                    farthest_visible = i

            # Only add waypoint if it's far enough from previous
            if len(simplified) > 0:
                dist = np.sqrt((path[farthest_visible][0] - simplified[-1][0])**2 +
                              (path[farthest_visible][1] - simplified[-1][1])**2)
                if dist >= min_distance or farthest_visible == len(path) - 1:
                    simplified.append(path[farthest_visible])
            else:
                simplified.append(path[farthest_visible])

            current_idx = farthest_visible

        # Post-process: add intermediate points for long segments
        final_path = [simplified[0]]
        for i in range(1, len(simplified)):
            p1 = simplified[i-1]
            p2 = simplified[i]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if dist > max_distance:
                # Add intermediate waypoints
                n_segments = int(np.ceil(dist / max_distance))
                for j in range(1, n_segments):
                    t = j / n_segments
                    intermediate = (
                        p1[0] + t * (p2[0] - p1[0]),
                        p1[1] + t * (p2[1] - p1[1])
                    )
                    final_path.append(intermediate)

            final_path.append(p2)

        return final_path

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if there's a clear line of sight between two points."""
        # Bresenham-like line check in world coordinates
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        steps = int(dist / (self.config.resolution * 0.5))  # Check every half-cell

        if steps == 0:
            return True

        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])

            ix, iy = self._world_to_grid(x, y)
            if not self.is_valid(ix, iy):
                return False

        return True

    def visualize(self, path: List[Tuple[float, float]] = None,
                  start: Tuple[float, float] = None,
                  goal: Tuple[float, float] = None,
                  save_path: str = 'cspace_plan.png'):
        """Visualize the configuration space and path."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot occupancy grid
        extent = [self.config.x_min, self.config.x_max,
                  self.config.y_min, self.config.y_max]
        ax.imshow(self.grid.T, origin='lower', extent=extent,
                  cmap='Greys', alpha=0.7)

        # Plot path
        if path is not None:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
            ax.plot(path_x, path_y, 'bo', markersize=8)

        # Plot start and goal
        if start is not None:
            ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
        if goal is not None:
            ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Configuration Space Path Planning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()


def plan_path_for_mpc(obstacles: List[dict],
                      start: Tuple[float, float],
                      goal: Tuple[float, float],
                      robot_radius: float = 0.35,
                      safety_margin: float = 0.15) -> Optional[List[List[float]]]:
    """
    Convenience function to plan a path for MPC.

    Args:
        obstacles: List of obstacle dicts from environment
        start: Start position (x, y)
        goal: Goal position (x, y)
        robot_radius: Robot radius
        safety_margin: Safety margin

    Returns:
        List of waypoints [[x, y], ...] suitable for MPC, or None if no path
    """
    planner = ConfigurationSpacePlanner(
        obstacles=obstacles,
        robot_radius=robot_radius,
        safety_margin=safety_margin
    )

    path = planner.plan(start, goal, simplify=True)

    if path is None:
        return None

    # Visualize
    planner.visualize(path, start, goal)

    # Convert to list format (excluding start and goal)
    # MPC will use these as intermediate waypoints
    waypoints = [[p[0], p[1]] for p in path[1:-1]]

    return waypoints


if __name__ == "__main__":
    # Test with example obstacles
    obstacles = [
        {'name': 'bar', 'type': 'box', 'position': [3.0, 0.0], 'size': [0.6, 5.0]},
        {'name': 'cabinet_0', 'type': 'box', 'position': [4.65, 0.0], 'size': [0.6, 0.5]},
        {'name': 'cabinet_1', 'type': 'box', 'position': [4.65, 1.0], 'size': [0.6, 0.5]},
        {'name': 'cabinet_-1', 'type': 'box', 'position': [4.65, -1.0], 'size': [0.6, 0.5]},
        {'name': 'barstool_0', 'type': 'circle', 'position': [2.3, 0.0], 'radius': 0.25},
        {'name': 'barstool_1', 'type': 'circle', 'position': [2.3, 1.0], 'radius': 0.25},
        {'name': 'barstool_2', 'type': 'circle', 'position': [2.3, 2.0], 'radius': 0.25},
    ]

    start = (0.0, 0.0)
    goal = (4.0, 0.0)

    waypoints = plan_path_for_mpc(obstacles, start, goal)

    if waypoints:
        print(f"\nWaypoints for MPC:")
        for i, wp in enumerate(waypoints):
            print(f"  {i}: [{wp[0]:.2f}, {wp[1]:.2f}]")
