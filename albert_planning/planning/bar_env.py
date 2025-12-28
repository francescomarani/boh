import numpy as np
import pybullet as p
import os
from typing import List, Optional
from urdfenvs.urdf_common.generic_robot import GenericRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.urdf_obstacle import UrdfObstacle

class BarEnvironment(UrdfEnv):
    """
    Bar environment with integrated scenario setup.
    Extends UrdfEnv to include bar-specific furniture and layout.
    
    This environment includes:
    - Room walls (as tracked obstacles)
    - Bar counter (as tracked obstacle)
    - Barstools, tables, chairs, cabinets (as PyBullet bodies)
    """
    
    # --- Room and Bar Constants ---
    BAR_POS = [3.0, 0.0, 0.4]
    BAR_SIZE = [0.6, 5, 0.8]
    ROOM_SIZE_HORIZ = 20.0 
    ROOM_SIZE_VERT = 10.0 
    WALL_HEIGHT = 1.0
    WALL_THICKNESS = 0.1
    
    def __init__(
        self,
        robots: List[type(GenericRobot)],
        render: bool = False,
        enforce_real_time: Optional[bool] = None,
        dt: float = 0.01,
        num_sub_steps: int = 20,
        observation_checking: bool = True,
        # Bar-specific parameters
        floor_urdf: str = "urdfenvs/floor/floor.urdf",
        bar_cabinet_urdf: str = "urdfenvs/bar_cabinet/bar_cabinet.urdf",
        barstool_urdf: str = "urdfenvs/barstool/barstool.urdf",
        chair_urdf_prefix: str = "urdfenvs/chair/chair_table",
        table_urdf_prefix: str = "urdfenvs/round_table/round_table",
        auto_setup_scene: bool = True,
        furniture_as_obstacles: bool = False  # Add furniture as MPC obstacles
    ) -> None:
        """
        Initialize Bar Environment.
        
        Parameters
        ----------
        robots : List[GenericRobot]
            List of robots to simulate
        render : bool
            Whether to render the simulation
        dt : float
            Time step for physics engine
        num_sub_steps : int
            Number of physics sub-steps per dt
        observation_checking : bool
            Whether to validate observations
        bar_cabinet_urdf : str
            Path to bar cabinet URDF file
        barstool_urdf : str
            Path to barstool URDF file
        chair_urdf_prefix : str
            Prefix for chair URDF files (will append _{counter}.urdf)
        table_urdf_prefix : str
            Prefix for table URDF files (will append _{counter}.urdf)
        furniture_as_obstacles : bool
            If True, add simplified furniture as BoxObstacle for MPC collision avoidance.
            If False, furniture is only visual (loaded as PyBullet bodies).
            Default: False (visual only)
        """
        # Initialize parent UrdfEnv
        super().__init__(
            robots=robots,
            render=render,
            enforce_real_time=enforce_real_time,
            dt=dt,
            num_sub_steps=num_sub_steps,
            observation_checking=observation_checking
        )
        
        # Store URDF paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.floor_urdf = os.path.join(base_dir, floor_urdf)    
        self.bar_cabinet_urdf = os.path.join(base_dir, bar_cabinet_urdf)
        self.barstool_urdf = os.path.join(base_dir, barstool_urdf)
        self.chair_urdf_prefix = os.path.join(base_dir, chair_urdf_prefix)
        self.table_urdf_prefix = os.path.join(base_dir, table_urdf_prefix)
        self.furniture_as_obstacles = furniture_as_obstacles
        
        # Counters for unique URDFs
        self.chair_counter = 1
        self.table_counter = 1
        
        # Track furniture body IDs for potential removal/reset
        self.furniture_bodies = {
            'floor': [],
            'barstools': [],
            'cabinets': [],
            'tables': [],
            'chairs': []
        }
        
        # Setup scene if requested
        if auto_setup_scene:
            self.setup_bar_scene()
    
    def setup_bar_scene(self) -> None:
        """
        Setup the complete bar scene with walls, bar, and furniture.
        This method can be called manually if auto_setup_scene=False.
        Can also be used to reload the scene after modifications.
        """
        print("=" * 50)
        print("Setting up Bar Environment...")
        print("=" * 50)
        
        self._load_floor()
        self._add_walls_and_bar()
        
        # Add furniture as obstacles if requested (for MPC)
        if self.furniture_as_obstacles:
            self._add_furniture_obstacles()
        
        # Always load visual furniture
        self._load_furniture()
        
        print("=" * 50)
        print("Bar Environment setup complete!")
        print("=" * 50)
    
    def _load_floor(self) -> None:
        """Load the floor URDF into the environment."""
        try:
            floor_id = p.loadURDF(
                self.floor_urdf, 
                [0, 0, 0], 
                p.getQuaternionFromEuler([0, 0, 0]), 
                useFixedBase=True
            )
            self.furniture_bodies['floor'].append(floor_id)
            print("✓ Loaded floor")
        except Exception as e:
            print(f"✗ Error loading floor: {e}")

            
    def _add_walls_and_bar(self) -> None:
        """
        Add walls and bar counter as tracked obstacles using mpscenes BoxObstacle.
        These will be registered in the environment's obstacle dictionary
        for collision detection and are suitable for MPC planning.
        """
        wall_offset_horiz = self.ROOM_SIZE_HORIZ / 2.0
        wall_offset_vert = self.ROOM_SIZE_VERT / 2.0
        wall_size_horiz = [self.ROOM_SIZE_VERT, self.WALL_THICKNESS, self.WALL_HEIGHT]
        wall_size_vert = [self.WALL_THICKNESS, self.ROOM_SIZE_HORIZ, self.WALL_HEIGHT]
        
        # Bar counter obstacle
        bar_dict = {
            'type': 'box',
            'geometry': {
                'position': self.BAR_POS,
                'length': self.BAR_SIZE[0],
                'width': self.BAR_SIZE[1],
                'height': self.BAR_SIZE[2],
            },
            'rgba': [0.6, 0.4, 0.2, 1.0],
            'movable': False,
        }
        
        # Wall obstacle dictionaries
        wall_dicts = [
            # Top wall
            {
                'type': 'box',
                'geometry': {
                    'position': [0, wall_offset_horiz, self.WALL_HEIGHT/2],
                    'length': wall_size_horiz[0],
                    'width': wall_size_horiz[1],
                    'height': wall_size_horiz[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
            # Bottom wall
            {
                'type': 'box',
                'geometry': {
                    'position': [0, -wall_offset_horiz, self.WALL_HEIGHT/2],
                    'length': wall_size_horiz[0],
                    'width': wall_size_horiz[1],
                    'height': wall_size_horiz[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
            # Right wall
            {
                'type': 'box',
                'geometry': {
                    'position': [wall_offset_vert, 0, self.WALL_HEIGHT/2],
                    'length': wall_size_vert[0],
                    'width': wall_size_vert[1],
                    'height': wall_size_vert[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
            # Left wall
            {
                'type': 'box',
                'geometry': {
                    'position': [-wall_offset_vert, 0, self.WALL_HEIGHT/2],
                    'length': wall_size_vert[0],
                    'width': wall_size_vert[1],
                    'height': wall_size_vert[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
        ]
        
        # Create obstacles list
        obstacles = []
        obstacles.append(BoxObstacle(name="bar_table", content_dict=bar_dict))
        
        for i, wall_dict in enumerate(wall_dicts):
            wall_names = ["wall_top", "wall_bottom", "wall_right", "wall_left"]
            obstacles.append(BoxObstacle(name=wall_names[i], content_dict=wall_dict))
        
        # Add all obstacles to environment
        for obstacle in obstacles:
            self.add_obstacle(obstacle)
        
        print(f"✓ Added {len(obstacles)} structural obstacles (walls + bar)")
    
    def _add_furniture_obstacles(self) -> None:
        """
        Add simplified furniture as box obstacles for MPC collision avoidance.
        
        This creates bounding box approximations of furniture that the MPC
        controller can use for collision-free trajectory planning.
        
        Note: Visual furniture is still loaded separately for rendering.
        """
        from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
        
        furniture_obstacles = []
        
        # === BARSTOOLS (as small boxes) ===
        barstool_positions = [0.0, 1.0, -1.0, 2.0, -2.0]
        for y in barstool_positions:
            stool_dict = {
                'type': 'box',
                'geometry': {
                    'position': [2.3, y, 0.3],
                    'width': 0.4,
                    'height': 0.6,
                    'length': 0.4,
                },
                'rgba': [0.4, 0.3, 0.2, 0.3],  # Semi-transparent
                'movable': False,
            }
            furniture_obstacles.append(
                BoxObstacle(name=f"obstacle_barstool_{y}", content_dict=stool_dict)
            )
        
        # === TABLES (as cylinders for round tables) ===
        table_positions = [
            [-2.0, 1.0, 0.0],
            [-2.0, -3.0, 0.0],
            [-2.0, 6.0, 0.0],
            [3.0, 7.0, 0.0],
            [3.0, -5.0, 0.0]
        ]
        
        for i, pos in enumerate(table_positions):
            # Use cylinder for round tables (more accurate)
            table_dict = {
                'type': 'cylinder',
                'geometry': {
                    'position': [pos[0], pos[1], 0.4],
                    'radius': 0.5,   # Table radius
                    'height': 0.8,   # Table height
                },
                'rgba': [0.5, 0.4, 0.3, 0.3],
                'movable': False,
            }
            furniture_obstacles.append(
                CylinderObstacle(name=f"obstacle_table_{i}", content_dict=table_dict)
            )
        
        # === CHAIRS (simplified as small boxes) ===
        # Define chairs around each table
        chair_offset = 0.7  # Distance from table center
        chair_configs = [
            ([0.0, chair_offset], 0.0),      # Top
            ([0.0, -chair_offset], 180.0),   # Bottom
            ([-chair_offset, 0.0], 90.0),    # Left
            ([chair_offset, 0.0], -90.0)     # Right
        ]
        
        chair_counter = 0
        for table_idx, table_pos in enumerate(table_positions):
            for chair_idx, (offset, angle) in enumerate(chair_configs):
                chair_dict = {
                    'type': 'box',
                    'geometry': {
                        'position': [table_pos[0] + offset[0], 
                                   table_pos[1] + offset[1], 
                                   0.25],
                        'width': 0.4,
                        'height': 0.5,
                        'length': 0.4,
                    },
                    'rgba': [0.3, 0.25, 0.2, 0.3],
                    'movable': False,
                }
                furniture_obstacles.append(
                    BoxObstacle(name=f"obstacle_chair_{chair_counter}", 
                              content_dict=chair_dict)
                )
                chair_counter += 1
        
        # === CABINETS ===
        cabinet_positions = [0.0, 1.0, -1.0]
        for y in cabinet_positions:
            cabinet_dict = {
                'type': 'box',
                'geometry': {
                    'position': [4.65, y, 0.5],
                    'width': 0.5,
                    'height': 1.0,
                    'length': 0.6,
                },
                'rgba': [0.3, 0.25, 0.2, 0.3],
                'movable': False,
            }
            furniture_obstacles.append(
                BoxObstacle(name=f"obstacle_cabinet_{y}", content_dict=cabinet_dict)
            )
        
        # Add all furniture obstacles to environment
        for obstacle in furniture_obstacles:
            self.add_obstacle(obstacle)
        
        print(f"✓ Added {len(furniture_obstacles)} furniture obstacles for MPC")
        print(f"  - {len(barstool_positions)} barstools")
        print(f"  - {len(table_positions)} tables")
        print(f"  - {chair_counter} chairs")
        print(f"  - {len(cabinet_positions)} cabinets")
    
    def _load_furniture(self) -> None:
        """
        Load all furniture (barstools, cabinets, tables, chairs).
        These are loaded directly as PyBullet bodies since they're
        purely visual/physics objects without need for obstacle tracking.
        """
        self._load_barstools()
        self._load_cabinets()
        self._load_table_groups()
    
    def _load_barstools(self) -> None:
        """Load barstools along the bar counter."""
        try:
            stool_positions = [0.0, 1.0, -1.0, 2.0, -2.0]
            for y in stool_positions:
                body_id = p.loadURDF(
                    self.barstool_urdf, 
                    [2.3, y, 0.0], 
                    p.getQuaternionFromEuler([1.57, 0, 0]), 
                    useFixedBase=True
                )
                self.furniture_bodies['barstools'].append(body_id)
            
            print(f"✓ Loaded {len(stool_positions)} barstools")
        except Exception as e:
            print(f"✗ Error loading barstools: {e}")
    
    def _load_cabinets(self) -> None:
        """Load storage cabinets behind the bar."""
        try:
            cabinet_positions = [0.0, 1.0, -1.0]
            for y in cabinet_positions:
                body_id = p.loadURDF(
                    self.bar_cabinet_urdf, 
                    [4.65, y, 0.0], 
                    p.getQuaternionFromEuler([1.57, 0, -1.57]), 
                    useFixedBase=True
                )
                self.furniture_bodies['cabinets'].append(body_id)
            
            print(f"✓ Loaded {len(cabinet_positions)} cabinets")
        except Exception as e:
            print(f"✗ Error loading cabinets: {e}")
    
    def _load_table_groups(self) -> None:
        """
        Load all table groups with surrounding chairs.
        Each table has up to 4 chairs (top, bottom, left, right).
        """
        # Define all table positions in the bar
        table_configs = [
            {'pos': [-2.0, 1.0, 0.0], 'skip_chairs': []},
            {'pos': [-2.0, -3.0, 0.0], 'skip_chairs': []},
            {'pos': [-2.0, 6.0, 0.0], 'skip_chairs': []},
            {'pos': [3.0, 7.0, 0.0], 'skip_chairs': []},
            {'pos': [3.0, -5.0, 0.0], 'skip_chairs': []}
        ]
        
        for config in table_configs:
            self._load_table_group(**config)
        
        print(f"✓ Loaded {len(table_configs)} table groups")
    
    def _load_table_group(self, pos: List[float], skip_chairs: List[int] = None) -> None:
        """
        Load a table with surrounding chairs.
        
        Parameters
        ----------
        pos : List[float]
            Table position [x, y, z]
        skip_chairs : List[int], optional
            Indices of chairs to skip loading:
            0 = top, 1 = bottom, 2 = left, 3 = right
        """
        if skip_chairs is None:
            skip_chairs = []
        
        try:
            # Load table
            table_urdf = f"{self.table_urdf_prefix}_{self.table_counter}.urdf"
            self.table_counter += 1
            
            table_id = p.loadURDF(
                table_urdf, 
                pos, 
                p.getQuaternionFromEuler([1.57, 0, 0]), 
                useFixedBase=True
            )
            self.furniture_bodies['tables'].append(table_id)
            
            # Chair configurations: (offset_from_table, euler_angles)
            chair_configs = [
                ([0.0, 0.5, 0.0], [1.57, 0, 0]),           # 0: Top
                ([0.0, -0.5, 0.0], [1.57, 0, 3.14159]),    # 1: Bottom
                ([-0.5, 0.0, 0.0], [1.57, 0, 1.57]),       # 2: Left
                ([0.5, 0.0, 0.0], [1.57, 0, -1.57])        # 3: Right
            ]
            
            tx, ty, tz = pos
            
            # Load chairs
            for i, (offset, euler) in enumerate(chair_configs):
                chair_urdf = f"{self.chair_urdf_prefix}_{self.chair_counter}.urdf"
                self.chair_counter += 1
                
                if i not in skip_chairs:
                    cx, cy, cz = offset
                    chair_id = p.loadURDF(
                        chair_urdf,
                        [tx + cx, ty + cy, tz + cz],
                        p.getQuaternionFromEuler(euler),
                        useFixedBase=True
                    )
                    self.furniture_bodies['chairs'].append(chair_id)
        
        except Exception as e:
            print(f"✗ Error loading table group at {pos}: {e}")
    
    def clear_furniture(self) -> None:
        """
        Remove all furniture from the scene.
        Useful for resetting or reconfiguring the environment.
        """
        for category, body_ids in self.furniture_bodies.items():
            for body_id in body_ids:
                try:
                    p.removeBody(body_id)
                except:
                    pass
            body_ids.clear()
        
        # Reset counters
        self.chair_counter = 1
        self.table_counter = 1
        
        print("✓ Cleared all furniture")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        mount_positions: Optional[np.ndarray] = None,
        mount_orientations: Optional[np.ndarray] = None,
        reload_scene: bool = False
    ) -> tuple:
        """
        Reset the environment.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional reset options
        pos : np.ndarray, optional
            Initial robot joint positions
        vel : np.ndarray, optional
            Initial robot joint velocities
        mount_positions : np.ndarray, optional
            Robot mounting positions
        mount_orientations : np.ndarray, optional
            Robot mounting orientations
        reload_scene : bool
            If True, clear and reload all furniture (useful if scene was modified)
        
        Returns
        -------
        tuple
            (observation, info) - Initial observation and info dict
        """
        # Reset parent environment
        result = super().reset(seed, options, pos, vel, mount_positions, mount_orientations)
        
        # Optionally reload the entire scene
        if reload_scene:
            self.clear_furniture()
            self.setup_bar_scene()
        
        return result
    
    def get_furniture_count(self) -> dict:
        """
        Get count of furniture items in the scene.
        
        Returns
        -------
        dict
            Dictionary with counts for each furniture category
        """
        return {
            category: len(body_ids) 
            for category, body_ids in self.furniture_bodies.items()
        }