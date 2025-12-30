import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
import time
import math
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.scenario.bar_class_scenario import BarClassScenario

# albert.py and prm_planner.py need to be in the same folder
from prm_planner import PRMPlanner 

def get_robot_pos(env):
    """Get current position [x, y, theta]"""
    pos, quat = p.getBasePositionAndOrientation(env._robots[0]._robot)
    euler = p.getEulerFromQuaternion(quat)
    # Apply the offset so 0 radians matches the robot's face
    theta_offset = np.pi / 2
    corrected_theta = euler[2] + theta_offset
    # Normalize to [-pi, pi]
    corrected_theta = (corrected_theta + np.pi) % (2 * np.pi) - np.pi

    return np.array([pos[0], pos[1], corrected_theta]) 

def drive_to_point(current_pose, target_pos):
    """Simple P-Controller to drive robot to x,y"""
    x, y, theta = current_pose
    tx, ty = target_pos
    
    dx = tx - x
    dy = ty - y
    dist = math.sqrt(dx**2 + dy**2)
    desired_angle = math.atan2(dy, dx)
    
    # Calculate angle error (Shortest path)
    angle_diff = (desired_angle - theta + np.pi) % (2 * np.pi) - np.pi     # Normalize to [-pi, pi]
    
    # PID Gains
    kp_lin = 1.5 # Forward speed gain
    kp_ang = 5.0 # Turn speed gain
    
    if abs(angle_diff) > 0.3: # If facing wrong way, turn first
        lin_vel = 0.0
        ang_vel = kp_ang * angle_diff
    else: # If facing target, drive forward
        lin_vel = kp_lin * dist
        ang_vel = kp_ang * angle_diff
        
    # Cap velocities for Albert
    lin_vel = np.clip(lin_vel, -1.0, 1.0)
    ang_vel = np.clip(ang_vel, -2.0, 2.0)
    
    lin_vel = -lin_vel # We invert the linear velocity so it drives the other way

    # Convert to Wheel Velocities
    L = 0.494 # Wheel distance
    v_right = lin_vel + (ang_vel * L / 2)
    v_left = lin_vel - (ang_vel * L / 2)
    
    return v_right, v_left


def run_albert(n_steps=10000, render=False, goal=True, obstacles=True):
    # 1. Define Robot
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

    # 2. Init Environment
    print("Loading Scenario...")
    my_scenario = BarClassScenario()
    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render, scenario=my_scenario)

    # 3. Reset
    start_pos = [-4.0, 8, 0.2]
    goal_pos = (0.0, 0.0) # Near the bar
    env.reset()
    robot_id = env._robots[0]._robot
    # Hidding robot underground so that planner doesn't hit it
    p.resetBasePositionAndOrientation(robot_id, [0, 0, -10], [0,0,0,1])
    
    # 4. PRM PLANNER EXECUTION 
    print("Initializing Planner...")
    # Bounds: [x_min, x_max, y_min, y_max] (Based on your room size)
    prm = PRMPlanner(bounds=[-4.5, 4.5, -9.5, 9.5], robot_radius=0.6)
    prm.build_roadmap(n_samples=1000, connect_dist=5.0)  # Build Map 
    
    # Uncomment to see the whole roadmap connections in blue
    prm.draw_roadmap() # drawing roadmap

    # Find Path
    print(f"Planning path from {start_pos[:2]} to {goal_pos}...")
    path = prm.find_path((start_pos[0], start_pos[1]), goal_pos)

    # Bring robot back to start
    print(f"Teleporting robot to start...")
    p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, 0]))

    if not path:
        print("Failed to find path! Robot is stuck.")
        env.close()
        return
    # Draw path in green
    prm.draw_path(path)


    # 5. EXECUTION LOOP
    current_idx = 0
    action = np.zeros(env.n())
    
    print("Following path...")
    try:
        while True:
            robot_pose = get_robot_pos(env)

            # --- DEBUG: Show Robot Forward Direction ---
            # Draws a BLUE LINE indicating where the robot thinks "Forward" is.
            cx, cy, ctheta = robot_pose
            fx = cx + math.cos(ctheta) * 1.0
            fy = cy + math.sin(ctheta) * 1.0
            p.addUserDebugLine([cx, cy, 0.5], [fx, fy, 0.5], [0, 0, 1], lifeTime=0.1, lineWidth=3)
            # ------------------------------------------


            # Check if we have path points left
            if current_idx < len(path):
                target = path[current_idx]
                dist = math.dist(robot_pose[:2], target)
                # Visual Debug: Draw line to current target
                p.addUserDebugLine([robot_pose[0], robot_pose[1], 0.5], [target[0], target[1], 0.5], [1, 0, 0], lifeTime=0.1, lineWidth=2)
                
                # If we are close to the waypoint, go to next one
                if dist < 0.3:
                    current_idx += 1
                    print(f"Reached waypoint {current_idx}/{len(path)}")
                    continue
                # Calculate wheel velocities
                v_right, v_left = drive_to_point(robot_pose, target)
                # If the robot turns RIGHT when it should go LEFT, swap these two lines:
                action[0] = v_right
                action[1] = v_left
            else:
                # Reached Goal
                action = np.zeros(env.n()) # Stop
                print("Goal Reached!")
            ob, *_ = env.step(action)
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("Stopping...")
    
    env.close()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_albert(render=True)





