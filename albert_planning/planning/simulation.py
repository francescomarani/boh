import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from bar_env import BarEnvironment
import visualization
from models import DifferentialDriveDynamics
from mpc_planning import BaseMPC
from collision_avoidance_mpc import (
    CollisionAvoidanceMPC,
    Obstacle,
    obstacles_from_dict_list
)
from tqdm import tqdm

class AlbertSimulation:
    def __init__(self, dt=0.01, Base_N=20, Arm_N=10,
                 T=50, x_init=np.array([0., 1., 0.]),
                 x_target=np.array([5., 0., 0.]),
                 # Collision avoidance parameters
                 enable_collision_avoidance=True,
                 robot_radius=0.35,
                 safety_margin=0.15,
                 use_soft_constraints=False,
                 soft_constraint_weight=100.0):
        self.dt = dt  # Simulation timestep

        # MPC PARAMETERS
        self.Base_N = Base_N  # Base MPC time horizon
        self.Arm_N = Arm_N    # Arm MPC time horizon
        STATE_WEIGHT = 50.0
        INPUT_WEIGHT = 20
        SOLVER_MAX_ITER = 30
        DO_WARM_START = True  # Warm start flag

        self.T = T  # Total simulation time steps
        self.plot_trajectories = False  # Plot trajectories flag
        self.x_init = x_init  # Initial state
        self.x_target = x_target  # Target state

        # Collision avoidance parameters
        self.enable_collision_avoidance = enable_collision_avoidance
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.use_soft_constraints = use_soft_constraints
        self.soft_constraint_weight = soft_constraint_weight

        # Vehicle dynamics model
        self.base_model = DifferentialDriveDynamics(dt)

        # MPC controller will be initialized in run_albert() after environment is created
        # This allows us to extract obstacles from the environment
        self.base_mpc = None

        # PyBullet robot ID (will be set in run_albert)
        self.robot_id = None
        self.base_joint_indices = None
    
    def run_albert(self, render=False, goal=True, obstacles=True):
        """
        Initialize and run the Albert robot simulation

        Args:
            render: Enable rendering
            goal: Include goal visualization
            obstacles: Include obstacles

        Returns:
            history: Simulation history
        """
        robots = [
            GenericDiffDriveRobot(
                urdf="albert.urdf",
                mode="vel",
                actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
                castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
                wheel_radius=0.08,
                wheel_distance=0.494,
                spawn_rotation=0,
                facing_direction='-y',
            ),
        ]
        env: BarEnvironment = BarEnvironment(
            dt=self.dt, robots=robots, render=render,
            furniture_as_obstacles=self.enable_collision_avoidance
        )

        # Reset environment with initial configuration
        ob = env.reset(
            pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
        )
        print(f"Initial observation: {ob}")

        self.env = env

        # Initialize MPC controller
        STATE_WEIGHT = 50.0
        INPUT_WEIGHT = 20
        SOLVER_MAX_ITER = 30
        DO_WARM_START = True

        if self.enable_collision_avoidance:
            print("\n" + "=" * 60)
            print("COLLISION AVOIDANCE ENABLED")
            print("=" * 60)

            # Get obstacles from environment (single source of truth)
            obstacle_dicts = env.get_mpc_obstacles()
            obstacle_list = obstacles_from_dict_list(obstacle_dicts)

            # Create collision-aware MPC
            self.base_mpc = CollisionAvoidanceMPC(
                dynamics=self.base_model,
                N=self.Base_N,
                x_target=self.x_target,
                wx=STATE_WEIGHT,
                wu=INPUT_WEIGHT,
                obstacles=obstacle_list,
                robot_radius=self.robot_radius,
                safety_margin=self.safety_margin,
                use_soft_constraints=self.use_soft_constraints,
                soft_constraint_weight=self.soft_constraint_weight,
                constraint_every_n_steps=1,  # Check every step for safety
                SOLVER_MAX_ITER=SOLVER_MAX_ITER,
                DO_WARM_START=DO_WARM_START
            )
            self.obstacles = obstacle_list
        else:
            print("\n" + "=" * 60)
            print("COLLISION AVOIDANCE DISABLED (using standard MPC)")
            print("=" * 60)

            # Create standard MPC without collision avoidance
            self.base_mpc = BaseMPC(
                self.base_model, self.Base_N, self.x_target,
                STATE_WEIGHT, INPUT_WEIGHT,
                SOLVER_MAX_ITER, DO_WARM_START
            )
            self.obstacles = []

        history = self.simulate(ob[0])

        env.close()
        return history
    
    def simulate(self, ob):
        """
        Run the MPC simulation loop
        
        Returns:
            tuple: (history, x_real, u_real, x_all) - observation history and state/control trajectories
        """
        # Initialize output arrays
        T = self.T
        x_real = np.zeros((3, T+1))  # [x, y, theta]
        x_all = np.zeros((3, self.Base_N+1, T+1))  # MPC predictions
        u_real = np.zeros((2, T))  # [v, omega]
        
        # Get initial state from environment using PyBullet API
        x_real[:,0]= ob['robot_0']["joint_state"]["position"][0:3]
        
        # IMPORTANT: Determine the total action dimension from environment
        # The environment expects action for base + arm + gripper
        print(f"\nEnvironment info:")
        print(f"  Action space dimension: {self.env.n()}")
        total_action_dim = self.env.n()
        
        print(f"\nStarting MPC simulation:")
        print(f"  Initial state: x={x_real[0,0]:.3f}, y={x_real[1,0]:.3f}, θ={x_real[2,0]:.3f}")
        print(f"  Target state:  x={self.x_target[0]:.3f}, y={self.x_target[1]:.3f}, θ={self.x_target[2]:.3f}")
        print(f"  Horizon: {self.Base_N}, Max iterations: {self.base_mpc.max_iter}\n")
        
        theta_all = np.zeros((T))
        history = []
        
        for t in tqdm(range(0, T), desc='Simulating'):
            # Current state
            x_current = x_real[:, t]
            
            # Solve MPC to get optimal control for BASE ONLY
            u_base, x_pred, x_all_out, theta_out = self.base_mpc.solve(x_current)
            
            # Create full action vector: base control + zero arm control
            # u_base is [v, omega] with shape (2,)
            # Full action is [v, omega, arm_joint1, arm_joint2, ..., gripper1, gripper2]
            action = np.zeros(total_action_dim)
            action[0] = u_base[0]  # Linear velocity
            action[1] = u_base[1]  # Angular velocity
            # action[2:] remain zero (arm and gripper don't move)
            
            # Apply control to REAL robot in PyBullet
            try:
                ob, reward, done, truncated, info = self.env.step(action)
            except ValueError:
                # If step() returns only 4 values instead of 5
                ob, reward, done, info = self.env.step(action)
                truncated = False
            
            # Extract REAL state using PyBullet API (not MPC prediction!)
            x_real[:, t+1] = ob['robot_0']["joint_state"]["position"][0:3]
            
            # Save for visualization (only base control)
            x_all[:, :, t] = x_all_out
            u_real[:, t] = u_base  # Save only base control
            theta_all[t] = theta_out
            history.append(ob)
            
            # Check convergence
            distance = np.linalg.norm(x_real[0:2, t+1] - self.x_target[0:2])
            if distance < 0.2:  # 20cm threshold
                print(f"\n✓ Goal reached at step {t}!")
                print(f"  Final position: x={x_real[0,t+1]:.3f}, y={x_real[1,t+1]:.3f}")
                print(f"  Final distance: {distance:.3f}m")
                # Truncate arrays
                x_real = x_real[:, :t+2]
                u_real = u_real[:, :t+1]
                x_all = x_all[:, :, :t+1]
                theta_all = theta_all[:t]
                break
            
            # Print progress every 10 steps
            if t % 10 == 0:
                progress_msg = (f"\nStep {t}: pos=({x_real[0,t]:.2f}, {x_real[1,t]:.2f}), "
                               f"θ={x_real[2,t]:.2f}, dist={distance:.2f}m, "
                               f"u=({u_base[0]:.3f}, {u_base[1]:.3f})")

                # Show obstacle distance if collision avoidance is enabled
                if hasattr(self.base_mpc, 'get_min_obstacle_distance'):
                    min_obs_dist, obs_name = self.base_mpc.get_min_obstacle_distance(x_current)
                    if min_obs_dist < float('inf'):
                        clearance = min_obs_dist - self.robot_radius
                        progress_msg += f", obs_dist={clearance:.2f}m ({obs_name})"

                print(progress_msg)
        
        # Plot trajectories using existing visualization code
        if self.plot_trajectories:
            visualization.plot_trajectories('mpc_control.eps', x_real, u_real, x_real.shape[1]-1)
        
        print(f"\nSimulation complete!")
        print(f"  Total steps: {x_real.shape[1]-1}")
        print(f"  Final distance to goal: {np.linalg.norm(x_real[0:2,-1] - self.x_target[0:2]):.3f}m")
        
        return history, x_real, u_real, x_all


def plot_results(x_real, u_real, x_target, dt, save_path='albert_mpc_results.png',
                 obstacles=None, robot_radius=0.35):
    """
    Create comprehensive visualization of MPC results

    Args:
        x_real: State trajectory (3, T+1) [x, y, theta]
        u_real: Control input trajectory (2, T) [v, omega]
        x_target: Target state (3,) [x, y, theta]
        dt: Timestep
        save_path: Path to save figure
        obstacles: List of Obstacle objects to visualize
        robot_radius: Robot radius for visualization
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 2D Trajectory with orientation arrows
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    # Plot obstacles first (behind trajectory)
    if obstacles is not None and len(obstacles) > 0:
        for obs in obstacles:
            if obs.type == 'circle':
                circle = plt.Circle(obs.position, obs.radius,
                                    color='red', alpha=0.3, zorder=1)
                ax1.add_patch(circle)
            elif obs.type == 'box':
                # Draw rectangle centered at position
                half_size = obs.size / 2
                rect = plt.Rectangle(
                    (obs.position[0] - half_size[0], obs.position[1] - half_size[1]),
                    obs.size[0], obs.size[1],
                    color='red', alpha=0.3, zorder=1
                )
                ax1.add_patch(rect)

    ax1.plot(x_real[0, :], x_real[1, :], 'b-', linewidth=2.5, label='Robot trajectory')
    ax1.plot(x_real[0, 0], x_real[1, 0], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(x_target[0], x_target[1], 'r*', markersize=20, label='Goal', zorder=5)
    
    # Draw orientation arrows every N steps
    arrow_step = max(1, len(x_real[0]) // 15)
    for i in range(0, x_real.shape[1], arrow_step):
        dx = 0.15 * np.cos(x_real[2, i])
        dy = 0.15 * np.sin(x_real[2, i])
        ax1.arrow(x_real[0, i], x_real[1, i], dx, dy, 
                 head_width=0.1, head_length=0.08, fc='blue', ec='blue', alpha=0.6)
    
    # Draw goal circle
    circle = plt.Circle((x_target[0], x_target[1]), 0.2, color='red', fill=False, 
                        linestyle='--', linewidth=2, label='Goal region (20cm)')
    ax1.add_patch(circle)
    
    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_title('Robot Trajectory in 2D Space', fontsize=14, fontweight='bold')
    
    # 2. States vs Time
    ax2 = fig.add_subplot(gs[0, 2])
    time = np.arange(x_real.shape[1]) * dt
    ax2.plot(time, x_real[0, :], 'b-', linewidth=2, label='x')
    ax2.plot(time, x_real[1, :], 'g-', linewidth=2, label='y')
    ax2.axhline(y=x_target[0], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axhline(y=x_target[1], color='g', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.set_xlabel('Time [s]', fontsize=10)
    ax2.set_ylabel('Position [m]', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Position vs Time', fontsize=12, fontweight='bold')
    
    # 3. Orientation vs Time
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(time, np.rad2deg(x_real[2, :]), 'r-', linewidth=2)
    ax3.axhline(y=np.rad2deg(x_target[2]), color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.set_xlabel('Time [s]', fontsize=10)
    ax3.set_ylabel('Orientation [deg]', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Orientation vs Time', fontsize=12, fontweight='bold')
    
    # 4. Control Inputs
    ax4 = fig.add_subplot(gs[2, 0])
    time_u = np.arange(u_real.shape[1]) * dt
    ax4.plot(time_u, u_real[0, :], 'b-', linewidth=2, label='v (linear vel)')
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, linewidth=1, label='v max')
    ax4.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax4.set_xlabel('Time [s]', fontsize=10)
    ax4.set_ylabel('Linear velocity [m/s]', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Linear Velocity Control', fontsize=12, fontweight='bold')
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_u, u_real[1, :], 'g-', linewidth=2, label='ω (angular vel)')
    ax5.axhline(y=2.0, color='r', linestyle='--', alpha=0.3, linewidth=1, label='ω max')
    ax5.axhline(y=-2.0, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax5.set_xlabel('Time [s]', fontsize=10)
    ax5.set_ylabel('Angular velocity [rad/s]', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Angular Velocity Control', fontsize=12, fontweight='bold')
    
    # 5. Distance to Goal
    ax6 = fig.add_subplot(gs[2, 2])
    distance = np.sqrt((x_real[0, :] - x_target[0])**2 + (x_real[1, :] - x_target[1])**2)
    ax6.plot(time, distance, 'purple', linewidth=2.5)
    ax6.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Goal threshold (20cm)')
    ax6.fill_between(time, 0, 0.2, alpha=0.2, color='green', label='Goal region')
    ax6.set_xlabel('Time [s]', fontsize=10)
    ax6.set_ylabel('Distance [m]', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Distance to Goal', fontsize=12, fontweight='bold')
    ax6.set_yscale('log')
    
    # Add overall title
    fig.suptitle('Albert Mobile Base - MPC Control Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Results plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"

    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)

        # =====================================================================
        # COLLISION AVOIDANCE MPC SIMULATION
        # =====================================================================
        # Set enable_collision_avoidance=True to enable obstacle avoidance
        # Set enable_collision_avoidance=False for standard MPC (no obstacles)

        # Target: navigate to end of bar (accessible without going through barstools)
        # Bar is at [3.0, 0.0] with size [0.6, 5.0] (extends y: -2.5 to +2.5)
        # Barstools are at x=2.3, y in [-2, -1, 0, 1, 2]
        # This target is reachable by going around barstools via y > 2.5
        x_target = np.array([3.0, 3.5, 0.])  # End of bar, past the barstools

        # Create simulation with collision avoidance
        sim = AlbertSimulation(
            dt=0.05,  # 50ms timestep
            Base_N=50,  # Longer horizon to plan around obstacles
            T=800,  # Max timesteps
            x_init=np.array([0., 0., 0.]),
            x_target=x_target,
            # Collision avoidance parameters
            enable_collision_avoidance=True,  # Set to False to disable
            robot_radius=0.35,  # Albert robot radius (approximate)
            safety_margin=0.15,  # Safety margin
            use_soft_constraints=True,  # Soft + hard constraints for obstacle "vision"
            soft_constraint_weight=10.0  # Low weight + fast decay avoids local minima
        )

        # Run simulation
        history, x_real, u_real, x_all = sim.run_albert(render=True)

        # Plot comprehensive results with obstacles
        plot_results(x_real, u_real, sim.x_target, sim.dt,
                     save_path='albert_mpc_collision_avoidance_results.png',
                     obstacles=sim.obstacles if hasattr(sim, 'obstacles') else None,
                     robot_radius=sim.robot_radius)