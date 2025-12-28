import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def _name_to_index_map(robot):
    """Return a mapping joint_name -> flat-action-index for the given robot.

    The environment uses a flat action vector (length = env.n()). Each robot
    contributes a block inside that vector; the order is the robot's
    `robot._joint_names` list. Use this mapping to set velocities by joint
    name (much safer than hardcoding numeric indices).
    """
    return {name: i for i, name in enumerate(robot._joint_names)}


def _arm_joint_indices(robot):
    """Return the indices (in the flat action vector) that correspond to
    the robot arm (i.e., exclude actuated base wheels / castors when present).
    """
    # the concrete robot stores these as internal attributes with leading underscore
    exclude = set()
    exclude.update(getattr(robot, "_actuated_wheels", []))
    exclude.update(getattr(robot, "_castor_wheels", []))
    return [i for i, name in enumerate(robot._joint_names) if name not in exclude]


def run_albert(n_steps=1000, render=False):
    """Simplified runner for the Albert example.

    - Keeps the base (wheels) fixed at zero velocity.
    - Demonstrates how to command arm joint velocities **by joint name**.

    Control note: to move a joint set action[name_to_index[joint_name]] = velocity.
    The example below moves two arm joints while keeping the base stationary.
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

    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)
    robot = env._robots[0]

    # Build helper maps
    name_to_idx = _name_to_index_map(robot)
    arm_idxs = _arm_joint_indices(robot)

    print("joint names:", robot._joint_names)
    print("flat action index -> joint name:", list(enumerate(robot._joint_names)))
    print("arm indices (excluded base wheels):", arm_idxs)

    # action vector is flat over all robot DOFs
    action = np.zeros(env.n())

    # Ensure base is stopped (indices corresponding to actuated wheels)
    for wheel in getattr(robot, "actuated_wheels", []):
        if wheel in name_to_idx:
            action[name_to_idx[wheel]] = 0.0

    # Example: set two arm joint velocities by name (safe, robust)
    # adjust these names to move different arm joints
    example_moves = {"mmrobot_joint4": -0.1, "mmrobot_joint5": 0.05}
    for jname, vel in example_moves.items():
        if jname in name_to_idx:
            action[name_to_idx[jname]] = vel
        else:
            print(f"Warning: joint '{jname}' not found in robot; available: {robot._joint_names}")

    ob = env.reset(pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))
    print(f"Initial observation : {ob}")

    history = []
    steps_executed = 0
    for _ in range(n_steps):
        # re-enforce base 0 velocity each step to be defensive
        for wheel in getattr(robot, "actuated_wheels", []):
            if wheel in name_to_idx:
                action[name_to_idx[wheel]] = 0.0

        ob, *_ = env.step(action)
        history.append(ob)

    env.close()
    return history


def run_albert_torque_sequence(render=False, base_torque=1.0, torque_increment=0.5, duration_per_joint=1.0):
    """Run a sequence that applies torque to each arm joint in turn.

    Behavior:
      - Creates the robot in torque control mode (ControlMode.torque)
      - Disables default velocity controllers so torque commands take effect
      - For each arm joint (excluding base wheels) applies a constant torque
        for `duration_per_joint` seconds; torque magnitude increases by
        `torque_increment` for each subsequent joint (so later joints move "faster").

    Note: torque values must be kept within joint effort limits; increase
    `base_torque` conservatively and inspect joint motion.
    """
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="tor",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        ),
    ]


    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)

    # Reset the environment first; this creates the pybullet body (_robot)
    env.reset(pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))
    robot = env._robots[0]

    # helper maps (computed after reset so robot has its joint and id info)
    name_to_idx = _name_to_index_map(robot)
    arm_idxs = _arm_joint_indices(robot)

    print("Running torque-sequence on arm joints:", [robot._joint_names[i] for i in arm_idxs])

    # disable velocity control so torques are directly applied (robot._robot exists now)
    robot.disable_velocity_control()

    dt = env.dt if hasattr(env, "dt") else 0.01
    steps_per = max(1, int(duration_per_joint / dt))

    # iterate arm joints and apply torques sequentially
    for i, idx in enumerate(arm_idxs):
        torque = base_torque * (1.0 + torque_increment * i)
        torques = np.zeros(robot.n())
        if idx < robot.n():
            torques[idx] = torque
        else:
            print(f"Skipping joint index {idx} (out of torque-array range)")
            continue
        print(f"Applying torque {torque:.3f} to joint {robot._joint_names[idx]} (index {idx}) for {duration_per_joint}s")
        for _ in range(steps_per):
            env.step(torques)

        # clear torques and give a short settling step
        env.step(np.zeros(robot.n()))

    env.close()


def run_albert_endpoint_mpc_controller(goal_xyz_local=(0.0, 0.0, 0.6),
                                      render=False,
                                      horizon=4,
                                      reg=1e-2,
                                      max_speed=0.5,
                                      max_steps=2000,
                                      tol_pos=0.02,
                                      verbose=True):
    """Cartesian endpoint MPC controller (single-shot linearized MPC).

    Short description:
      - Single-shooting MPC: assume dq constant over horizon H and optimize
        to minimize the predicted Cartesian error at the horizon.
      - Uses regularized least-squares closed-form solution and enforces
        velocity/joint limits by projection (clipping).

    See inline comments for algorithmic details. Use `verbose=True` to get
    diagnostic prints (current error, perturbed goal when necessary, fallback
    messages).
    """

    import pybullet as p

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

    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)
    env.reset(pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))
    robot = env._robots[0]

    arm_idxs = _arm_joint_indices(robot)
    if len(arm_idxs) == 0:
        raise RuntimeError("No arm joints found")
    end_flat_idx = arm_idxs[-1]
    end_link_index = robot._robot_joints[end_flat_idx]

    # compute target in world frame
    link_state = p.getLinkState(robot._robot, 0, computeLinkVelocity=0)
    base_pos = np.array([link_state[0][0], link_state[0][1], link_state[0][2]])
    goal_world = base_pos + np.array(goal_xyz_local)
    if verbose:
        print("Endpoint controller target (world):", goal_world)

    # If the goal is extremely close to the current end-effector position,
    # perturb slightly to force a visible motion in interactive runs.
    ee_state0 = p.getLinkState(robot._robot, end_link_index, computeLinkVelocity=0)
    ee_pos0 = np.array(ee_state0[0])
    if np.linalg.norm(goal_world - ee_pos0) < tol_pos:
        perturb = np.array([0.03, 0.0, 0.0])  # 3 cm small nudge along x
        goal_world = goal_world + perturb
        if verbose:
            print(f"Goal was within tol_pos; perturbed goal by {perturb} to {goal_world} to force movement")

    dt = env.dt if hasattr(env, "dt") else 0.01

    arm_cols = arm_idxs
    n_arm = len(arm_cols)
    # integrator state for Cartesian integral term (anti-windup applied)
    ee_int = np.zeros(3)
    max_int = 0.05

    for step in range(max_steps):
        ee_state = p.getLinkState(robot._robot, end_link_index, computeLinkVelocity=1)
        ee_pos = np.array(ee_state[0])

        err = goal_world - ee_pos
        dist = np.linalg.norm(err)
        if verbose:
            print(f"Step {step}: ee_pos={ee_pos}, err_norm={dist:.4f}")
        if dist < tol_pos:
            print(f"Endpoint reached in {step} steps, pos error {dist:.4f} m")
            break

        # Read joint positions in PyBullet's joint-index order (0..numJoints-1).
        # PyBullet's calculateJacobian expects position/velocity lists of length
        # equal to the number of DOFs returned by getNumJoints(). The robot's
        # internal mapping `robot._robot_joints` maps flat-action indices ->
        # PyBullet joint indices; we use both representations below.
        n_joints_pb = p.getNumJoints(robot._robot)
        joint_positions_pb = []
        for j in range(n_joints_pb):
            try:
                pos_j, vel_j, _, _ = p.getJointState(robot._robot, j)
            except Exception:
                pos_j = 0.0
            joint_positions_pb.append(pos_j)

        # Build a flat-ordered q array (length = robot.n()) where index = flat index
        # and value = position read from PyBullet at the corresponding pb index.
        q = np.zeros(robot.n())
        for flat_idx, pb_idx in enumerate(robot._robot_joints):
            try:
                q[flat_idx] = joint_positions_pb[pb_idx]
            except Exception:
                q[flat_idx] = 0.0

        if verbose:
            pb_arm_cols = [robot._robot_joints[i] for i in arm_cols]
            print(f"pybullet joints: {n_joints_pb}, flat->pb mapping length: {len(robot._robot_joints)}")
            print(f"arm flat indices: {arm_cols}, arm pb indices: {pb_arm_cols}")

        try:
            # Compute Jacobian and select arm columns (3 x n_arm)
            # Use the PyBullet-ordered joint positions vector for calculateJacobian
            jac_t, jac_r = p.calculateJacobian(
                robot._robot,
                end_link_index,
                [0, 0, 0],
                joint_positions_pb,
                [0.0] * n_joints_pb,
                [0.0] * n_joints_pb,
            )
            J_full = np.array(jac_t)  # shape: 3 x n_joints_pb
            # Map arm flat indices -> corresponding pybullet joint indices and select
            pb_arm_cols = [robot._robot_joints[i] for i in arm_cols]
            J = J_full[:, pb_arm_cols]  # 3 x n_arm

            # Single-shooting: assume constant dq over horizon H
            H = max(1, int(horizon))
            scale = dt * H

            # Read end-effector linear velocity (world frame)
            v_ee = np.zeros(3)
            if len(ee_state) > 6 and ee_state[6] is not None:
                v_ee = np.array(ee_state[6])

            # Cartesian PID+I controller to improve final-time convergence and
            # compensate for unmodeled dynamics (simple uncertainty handling).
            Kp = 1.2
            Ki = 0.6
            Kd = 0.8

            # integrator update with anti-windup
            ee_int += err * dt
            ee_int = np.clip(ee_int, -max_int, max_int)

            v_p = Kp * err / max(1e-6, scale)
            v_i = Ki * ee_int
            v_d = -Kd * v_ee
            v_des = v_p + v_i + v_d

            # final push: boost speed near the target to avoid slow tail behaviour
            final_boost_thresh = max(5 * tol_pos, tol_pos + 1e-6)
            boost = 1.5 if dist < final_boost_thresh else 1.0
            vcap = max_speed * boost
            vnorm = np.linalg.norm(v_des)
            if vnorm > vcap and vnorm > 0:
                v_des = v_des * (vcap / vnorm)

            # Adaptive regularization: more conservative when EE is fast,
            # but relax slightly when close to goal to speed final convergence.
            reg_eff = reg + 1e-2 * min(10.0, np.linalg.norm(v_ee))
            if dist < final_boost_thresh:
                shrink = 0.4 * (1.0 - dist / final_boost_thresh)
                reg_eff = max(1e-8, reg_eff * (1.0 - shrink))

            if verbose:
                print(f"v_p={v_p}, v_i={v_i}, v_d={v_d}, v_des={v_des}, reg_eff={reg_eff:.4e}, boost={boost}")

            # Solve for joint velocities with damped least-squares mapping
            A = (J.T @ J) + reg_eff * np.eye(n_arm)
            b = J.T @ v_des
            dq_arm = np.linalg.pinv(A) @ b

            # Clip per-joint velocities to limits
            for i_local, flat_idx in enumerate(arm_cols):
                try:
                    vmax = abs(robot._limit_vel_j[1, flat_idx])
                except Exception:
                    vmax = float('inf')
                dq_arm[i_local] = float(np.clip(dq_arm[i_local], -vmax, vmax))

            # Enforce joint position limits at horizon (projection)
            q_arm = q[arm_cols]
            q_arm_pred = q_arm + dq_arm * scale
            for i_local, flat_idx in enumerate(arm_cols):
                try:
                    low = robot._limit_pos_j[0, flat_idx + 1]
                    high = robot._limit_pos_j[1, flat_idx + 1]
                    q_arm_pred[i_local] = float(np.clip(q_arm_pred[i_local], low, high))
                    dq_arm[i_local] = (q_arm_pred[i_local] - q_arm[i_local]) / scale
                except Exception:
                    pass

            dq = np.zeros(robot.n())
            for i_local, flat_idx in enumerate(arm_cols):
                dq[flat_idx] = dq_arm[i_local]

            # fill joint command array from dq_arm (flat order)
            dq = np.zeros(robot.n())
            for i_local, flat_idx in enumerate(arm_cols):
                dq[flat_idx] = dq_arm[i_local]

        except Exception as e:
            # Robust fallback: avoid calculateJacobian entirely (it failed) and
            # use inverse kinematics + joint-space PD to produce a safe dq.
            if verbose:
                print("Jacobian/MPC failed:", e)
                print("Falling back to IK-based joint-space PD (no Jacobian calls)")
                print(f"Diagnostics: n_joints_pb={n_joints_pb}, len(joint_positions_pb)={len(joint_positions_pb)}, end_link_index={end_link_index}")
            try:
                # IK solution: returns angles for PyBullet joint ordering (len == n_joints_pb)
                ik_solution = p.calculateInverseKinematics(robot._robot, end_link_index, goal_world)
                dq = np.zeros(robot.n())
                # PD gain for joint space fallback
                kp_js = 2.0
                for i_local, flat_idx in enumerate(arm_cols):
                    pb_idx = robot._robot_joints[flat_idx]
                    try:
                        target_q = ik_solution[pb_idx]
                        cur_q = joint_positions_pb[pb_idx]
                        dq_val = kp_js * (target_q - cur_q)
                        try:
                            vmax = abs(robot._limit_vel_j[1, flat_idx])
                            if vmax < 1e-3:
                                vmax = 0.1
                        except Exception:
                            vmax = 0.1
                        dq[flat_idx] = float(np.clip(dq_val, -vmax, vmax))
                    except Exception:
                        dq[flat_idx] = 0.0
                if verbose:
                    print("IK fallback dq (flat order):", [dq[i] for i in arm_cols])
            except Exception as e2:
                # If IK also fails, zero the command but print diagnostic information
                print("Fallback also failed, zeroing command:", e2)
                if verbose:
                    print("Full diagnostics:")
                    print(f"robot._robot={robot._robot}, end_link_index={end_link_index}")
                    print(f"n_joints_pb={n_joints_pb}, robot._robot_joints={robot._robot_joints}")
                dq = np.zeros(robot.n())

        # cap per-joint velocities again to be defensive
        for i in range(robot.n()):
            try:
                vmax = abs(robot._limit_vel_j[1, i])
                dq[i] = float(np.clip(dq[i], -vmax, vmax))
            except Exception:
                pass

        # construct flat action vector (keep base wheels zero)
        action = np.zeros(env.n())
        name_to_idx = _name_to_index_map(robot)
        for wheel in getattr(robot, "_actuated_wheels", []):
            if wheel in name_to_idx:
                action[name_to_idx[wheel]] = 0.0

        # fill action entries for all joints
        for idx in range(robot.n()):
            action[idx] = dq[idx]
        if verbose:
            arm_action = [action[i] for i in arm_cols]
            print(f"Applying action to arm (first {len(arm_action)} joints):", arm_action)
        env.step(action)

    env.close()

if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert_endpoint_mpc_controller(render=True)
