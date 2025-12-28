import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from bar_env import BarEnvironment
import visualization
from models import DifferentialDriveDynamics
from mpc_planning import BaseMPC, ArmMPC
from tqdm import tqdm

class AlbertSimulation:
    def __init__(self, dt=0.01, Base_N=20, Arm_N=10,
                 T=50, x_init=np.array([0., 1., 0.]),
                 x_target=np.array([5., 0., 0.])):
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
        
        # Vehicle dynamics model
        self.base_model = DifferentialDriveDynamics(dt)
        
        # Base MPC controller
        self.base_mpc = BaseMPC(self.base_model, Base_N, self.x_target,
                                STATE_WEIGHT, INPUT_WEIGHT, 
                                SOLVER_MAX_ITER, DO_WARM_START)
        
        # Arm MPC controller placeholder (initialized when available)
        self.arm_mpc = None
        self.arm_wq = 200.0
        self.arm_wu = 2.0
        self.arm_max_iter = 30

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
                wheel_radius = 0.08,
                wheel_distance = 0.494,
                spawn_rotation = 0,
                facing_direction = '-y',
            ),
        ]
        env: BarEnvironment = BarEnvironment(
            dt=self.dt, robots=robots, render=render
        )
        
        # Reset environment with initial configuration
        ob = env.reset(
            pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
        )
        print(f"Initial observation: {ob}")
        
        self.env = env
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
                print(f"\nStep {t}: pos=({x_real[0,t]:.2f}, {x_real[1,t]:.2f}), "
                      f"θ={x_real[2,t]:.2f}, dist={distance:.2f}m, "
                      f"u=({u_base[0]:.3f}, {u_base[1]:.3f})")
        
        # Plot trajectories using existing visualization code
        if self.plot_trajectories:
            visualization.plot_trajectories('mpc_control.eps', x_real, u_real, x_real.shape[1]-1)
        
        print(f"\nSimulation complete!")
        print(f"  Total steps: {x_real.shape[1]-1}")
        print(f"  Final distance to goal: {np.linalg.norm(x_real[0:2,-1] - self.x_target[0:2]):.3f}m")
        
        return history, x_real, u_real, x_all

    def simulate_arm(self, ob, q_goal=None, max_steps=200, tol=1e-2, verbose=True):
        """
        Run joint-space Arm MPC to move arm joints while keeping base stationary.

        Args:
            ob: Initial observation (dict) as returned by env.reset()[0]
            q_goal: Optional numpy array with target joint positions for arm joints
            max_steps: Maximum number of simulation steps for arm controller
            tol: Tolerance on joint-space L2 error to stop
            verbose: Print progress

        Returns:
            history: list of observations during arm execution
            q_arm_hist: (n_joints, steps) array of joint trajectories
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call run_albert or create env first.")

        total_action_dim = self.env.n()
        robot = self.env._robots[0]

        # Determine arm joint indices (exclude wheels / castors)
        exclude = set()
        exclude.update(getattr(robot, "_actuated_wheels", []))
        exclude.update(getattr(robot, "_castor_wheels", []))
        arm_idxs = [i for i, n in enumerate(robot._joint_names) if n not in exclude]

        # Read current full joint positions
        q_full = ob['robot_0']["joint_state"]["position"]
        q_arm = np.array([q_full[i] for i in arm_idxs])

        # Default goal: small offsets from current
        if q_goal is None:
            offsets = np.zeros(len(arm_idxs))
            if len(offsets) >= 1:
                offsets[0] = 0.6
            if len(offsets) >= 2:
                offsets[1] = -0.3
            q_goal = q_arm + offsets

        # Clip to robot limits when available
        try:
            q_min = np.array([robot._limit_pos_j[0, idx + 1] for idx in arm_idxs])
            q_max = np.array([robot._limit_pos_j[1, idx + 1] for idx in arm_idxs])
            q_goal = np.clip(q_goal, q_min, q_max)
        except Exception:
            pass

        # Create ArmMPC if not already created
        if self.arm_mpc is None:
            self.arm_mpc = ArmMPC(robot, arm_idxs, self.Arm_N, self.dt, q_goal,
                                    wq=self.arm_wq, wu=self.arm_wu,
                                    SOLVER_MAX_ITER=self.arm_max_iter,
                                    DO_WARM_START=True)
            if verbose:
                print(f"✓ ArmMPC created for {len(arm_idxs)} joints")
        else:
            # update target if ArmMPC exists
            self.arm_mpc.q_target = np.asarray(q_goal).reshape(len(arm_idxs))

        history = []
        q_arm_hist = np.zeros((len(arm_idxs), max_steps + 1))
        q_arm_hist[:, 0] = q_arm

        for step in range(max_steps):
            u_arm, q_next, _ = self.arm_mpc.solve(q_arm)

            # Build full action: zeros for base, place arm velocities in appropriate indices
            action = np.zeros(total_action_dim)
            for i_local, flat_idx in enumerate(self.arm_mpc.arm_idxs):
                action[flat_idx] = float(u_arm[i_local])

            try:
                ob, reward, done, truncated, info = self.env.step(action)
            except ValueError:
                ob, reward, done, info = self.env.step(action)
                truncated = False

            # Read new arm state
            q_full = ob['robot_0']["joint_state"]["position"]
            q_arm = np.array([q_full[i] for i in self.arm_mpc.arm_idxs])
            q_arm_hist[:, step + 1] = q_arm
            history.append(ob)

            err_norm = np.linalg.norm(q_arm - self.arm_mpc.q_target)
            if verbose and step % 10 == 0:
                print(f"  Arm step {step}, err={err_norm:.4f}")
            if err_norm < tol:
                if verbose:
                    print(f"  ✓ Arm goal reached in {step} steps (err={err_norm:.4f})")
                q_arm_hist = q_arm_hist[:, :step + 2]
                break

        return history, q_arm_hist

    def run_albert_arm(self, render=False, q_goal=None):
        """
        Initialize environment and run the Arm MPC test (base remains stationary).
        """
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
        env: BarEnvironment = BarEnvironment(
            dt=self.dt, robots=robots, render=render
        )

        # Reset environment with initial configuration
        ob = env.reset(
            pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
        )
        print(f"Initial observation: {ob}")

        self.env = env
        history, q_arm_hist = self.simulate_arm(ob[0], q_goal=q_goal)

        env.close()
        return history, q_arm_hist


def plot_results(x_real, u_real, x_target, dt, save_path='albert_mpc_results.png'):
    """
    Create comprehensive visualization of MPC results
    
    Args:
        x_real: State trajectory (3, T+1) [x, y, theta]
        u_real: Control input trajectory (2, T) [v, omega]
        x_target: Target state (3,) [x, y, theta]
        dt: Timestep
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 2D Trajectory with orientation arrows
    ax1 = fig.add_subplot(gs[0:2, 0:2])
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
        
        # Create simulation with reasonable target
        sim = AlbertSimulation(
            dt=0.05,  # 50ms timestep
            Base_N=70,
            Arm_N=50,
            T=800,  # More timesteps to reach goal
            x_init=np.array([0., 0., 0.]),
            x_target=np.array([20., -10., 2.])  # Target 
        )
        
        # Run simulation
        sim.run_albert_arm(render=True)
        
        # Plot comprehensive results
        # plot_results(x_real, u_real, sim.x_target, sim.dt, 
        #              save_path='albert_mpc_results.png')