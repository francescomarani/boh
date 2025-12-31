import numpy as np
import casadi as cs
import pybullet as p
from adam.casadi.computations import KinDynComputations

class DifferentialDriveDynamics:
    def __init__(self, dt, facing_direction='-y'):
        """
        Differential drive dynamics model
        
        Args:
            dt: Time discretization step
            facing_direction: Direction robot faces in its local frame
                'x': robot front points in +x direction
                'y': robot front points in +y direction
                '-x': robot front points in -x direction  
                '-y': robot front points in -y direction (Albert default)
        """
        self.dt = dt
        self.facing_direction = facing_direction
        self.state_dim = 3  # [x, y, theta]
        self.input_dim = 2  # [v, omega]
        
        # Limits on state [x, y, theta]
        self.x_min = np.array([-10., -10., -2*np.pi])
        self.x_max = np.array([100., 100., 2*np.pi])
        
        # Limits on input [v, omega]
        self.u_min = np.array([-1.0, -2.0])
        self.u_max = np.array([1.0, 2.0])
    
    def continuous_dynamics(self, x, u):
        """
        Continuous-time dynamics: ẋ = f(x, u)
        
        Args:
            x: [x, y, theta] (CasADi symbolic)
            u: [v, omega] where v is forward velocity, omega is angular velocity
        
        Returns: 
            [ẋ, ẏ, θ̇]
        """
        # IMPORTANT: Match the robot's facing_direction!
        if self.facing_direction == 'x':
            x_dot = u[0] * cs.cos(x[2])
            y_dot = u[0] * cs.sin(x[2])
        elif self.facing_direction == '-x':
            x_dot = -u[0] * cs.cos(x[2])
            y_dot = -u[0] * cs.sin(x[2])
        elif self.facing_direction == 'y':
            # Robot facing +y: forward motion increases y, not x
            x_dot = -u[0] * cs.sin(x[2])
            y_dot = u[0] * cs.cos(x[2])
        elif self.facing_direction == '-y':
            # Robot facing -y: forward motion decreases y
            x_dot = u[0] * cs.sin(x[2])
            y_dot = -u[0] * cs.cos(x[2])
        else:
            raise ValueError(f"Invalid facing_direction: {self.facing_direction}")
        
        theta_dot = u[1]
        
        return cs.vertcat(x_dot, y_dot, theta_dot)
    
    def discrete_dynamics(self, x, u):
        """
        Discrete-time dynamics using Euler integration
        x[k+1] = x[k] + dt * f(x[k], u[k])
        """
        return x + self.dt * self.continuous_dynamics(x, u)
    
    def simulate_step(self, x, u):
        """
        Simulate one step (for numpy arrays, not CasADi)
        Used in simulation loop
        """
        x_next = np.zeros(3)
        
        if self.facing_direction == 'x':
            x_next[0] = x[0] + self.dt * u[0] * np.cos(x[2])
            x_next[1] = x[1] + self.dt * u[0] * np.sin(x[2])
        elif self.facing_direction == '-x':
            x_next[0] = x[0] - self.dt * u[0] * np.cos(x[2])
            x_next[1] = x[1] - self.dt * u[0] * np.sin(x[2])
        elif self.facing_direction == 'y':
            x_next[0] = x[0] - self.dt * u[0] * np.sin(x[2])
            x_next[1] = x[1] + self.dt * u[0] * np.cos(x[2])
        elif self.facing_direction == '-y':
            x_next[0] = x[0] + self.dt * u[0] * np.sin(x[2])
            x_next[1] = x[1] - self.dt * u[0] * np.cos(x[2])
        
        x_next[2] = x[2] + self.dt * u[1]
        
        return x_next
    

class ManipulatorDynamics:
    """
    Lightweight arm dynamics wrapper that exposes double-integrator state
    dynamics plus inverse-dynamics and forward-kinematics helpers.

    - State: x = [q, qdot] (joint positions/velocities for the selected arm joints)
    - Input: u = qddot (joint accelerations)
    - Continuous dynamics: xdot = [qdot, qddot]
    - Discrete dynamics: q_{k+1} = q_k + dt*qdot_k + 0.5*dt^2*u_k,
                         qdot_{k+1} = qdot_k + dt*u_k

    For inverse dynamics a constant mass matrix / bias is estimated once from
    PyBullet (if available) and then used symbolically in CasADi. If mass/bias
    functions are provided, those override the estimate.
    """

    def __init__(
        self,
        robot,
        arm_indices,
        dt,
        ee_link_index=None,
        mass_matrix_fun=None,
        bias_force_fun=None,
        fk_fun=None,
    ):
        """
        Args:
            robot: DifferentialDriveRobot (or GenericRobot) instance already loaded in PyBullet.
            arm_indices: iterable of action-space indices that correspond to arm joints (e.g. [2,3,...]).
            dt: timestep used by the controller/simulator.
            ee_link_index: optional PyBullet link index for the end-effector.
            mass_matrix_fun: optional callable M(q) -> (n,n) CasADi expression.
            bias_force_fun: optional callable h(q,qdot) -> (n,) CasADi expression.
            fk_fun: optional callable fk(q) -> (3,) CasADi expression for EE position.
        """
        self.robot = robot
        self.arm_indices = list(arm_indices)
        self.dt = float(dt)
        self.n = len(self.arm_indices)
        self.ee_link_index = ee_link_index
        self.ee_link_name = None

        # Limits pulled from the robot when available.
        self.q_min, self.q_max = self._extract_limits("_limit_pos_j", offset=1, default=10.0)
        self.qd_min, self.qd_max = self._extract_limits("_limit_vel_j", offset=1, default=5.0)
        self.tau_min, self.tau_max = self._extract_limits("_limit_tor_j", offset=0, default=20.0)

        # CasADi symbols
        q = cs.SX.sym("q", self.n)
        qd = cs.SX.sym("qd", self.n)
        qdd = cs.SX.sym("qdd", self.n)
        state = cs.vertcat(q, qd)

        # Continuous xdot = [qd; qdd]
        xdot = cs.vertcat(qd, qdd)
        self.f = cs.Function("f", [state, qdd], [xdot])

        # Discrete Euler step
        q_next = q + self.dt * qd
        qd_next = qd + self.dt * qdd
        self.discrete_f = cs.Function("discrete_f", [state, qdd], [cs.vertcat(q_next, qd_next)])

        # Try to build FK and inverse dynamics from URDF via KinDynComputations
        kindyn_fk, kindyn_inv_dyn = self._build_kindyn_functions(fk_fun, mass_matrix_fun, bias_force_fun)

        if kindyn_fk is not None:
            self.fk = kindyn_fk
        else:
            ee = cs.SX.zeros(3)
            self.fk = cs.Function("fk", [q], [ee])

        if kindyn_inv_dyn is not None:
            self.inv_dyn = kindyn_inv_dyn
        else:
            # Mass matrix / bias fallback
            M_expr, h_expr = self._build_mass_bias(mass_matrix_fun, bias_force_fun)
            tau = cs.mtimes(M_expr, qdd) + h_expr
            self.inv_dyn = cs.Function("inv_dyn", [state, qdd], [tau])

    def _extract_limits(self, attr, offset, default):
        """
        Helper to slice limits for the arm joints from the robot.
        attr: name of the limit array on the robot (e.g., _limit_pos_j).
        offset: index offset needed to map action index -> limit row (1 for positions/velocities).
        default: fallback absolute bound magnitude.
        """
        try:
            limits = getattr(self.robot, attr)
            lo = np.array([limits[0, idx + offset] for idx in self.arm_indices])
            hi = np.array([limits[1, idx + offset] for idx in self.arm_indices])
        except Exception:
            lo = -np.ones(self.n) * float(default)
            hi = np.ones(self.n) * float(default)
        return lo, hi

    def _build_mass_bias(self, mass_matrix_fun, bias_force_fun):
        """
        Build CasADi expressions for M(q) and h(q, qd).
        If user provides callables, use them; otherwise estimate once from PyBullet and
        freeze as constants to keep the expressions compatible with CasADi.
        """
        if mass_matrix_fun is not None:
            def _M(q_sym):
                return mass_matrix_fun(q_sym)
        else:
            M_const = self._estimate_mass_matrix()
            def _M(_q_sym):
                return cs.DM(M_const) if M_const is not None else cs.SX.eye(self.n)

        if bias_force_fun is not None:
            def _h(q_sym, qd_sym):
                return bias_force_fun(q_sym, qd_sym)
        else:
            h_const = self._estimate_bias_forces()
            def _h(_q_sym, _qd_sym):
                return cs.DM(h_const) if h_const is not None else cs.SX.zeros(self.n)

        q = cs.SX.sym("q_tmp", self.n)
        qd = cs.SX.sym("qd_tmp", self.n)
        M_expr = _M(q)
        h_expr = _h(q, qd)
        return M_expr, h_expr

    def _build_kindyn_functions(self, fk_fun_override, mass_matrix_fun, bias_force_fun):
        """
        Attempt to build FK and inverse dynamics using adam.casadi.computations.KinDynComputations.
        Returns (fk_function, inv_dyn_function) or (None, None) on failure.
        """
        if fk_fun_override is not None:
            # If user already provided FK, skip KinDyn for FK but still try for ID if possible
            fk_fun = fk_fun_override
        else:
            fk_fun = None
        inv_fun = None

        try:
            urdf_path = getattr(self.robot, "_urdf_file", None)
            if urdf_path is None:
                return fk_fun, inv_fun
            # Build joint name list for the selected arm indices
            joint_name_list = [self.robot._joint_names[idx] for idx in self.arm_indices]
            kinDyn = KinDynComputations(urdf_path, joint_name_list)

            # Base configuration for fixed base
            H_b = cs.SX.eye(4)
            v_b = cs.SX.zeros(6)

            # Mass matrix and bias
            bias_fun = kinDyn.bias_force_fun()
            mass_fun = kinDyn.mass_matrix_fun()

            bias_full = bias_fun(H_b, cs.SX.sym("q_tmp", self.n), v_b, cs.SX.sym("qd_tmp", self.n))
            M_full = mass_fun(H_b, cs.SX.sym("q_tmp2", self.n))
            # If floating base terms are present, strip first 6 rows/cols
            def _strip(mat):
                if mat.shape[0] >= self.n + 6:
                    return mat[6:, 6:] if len(mat.shape) == 2 else mat[6:]
                return mat
            h_expr = _strip(bias_full)
            M_expr = _strip(M_full)

            q = cs.SX.sym("q", self.n)
            qd = cs.SX.sym("qd", self.n)
            qdd = cs.SX.sym("qdd", self.n)
            state = cs.vertcat(q, qd)
            tau = cs.mtimes(M_expr, qdd) + h_expr
            inv_fun = cs.Function("inv_dyn", [state, qdd], [tau])

            # Forward kinematics: use provided override or build from KinDyn
            if fk_fun is None:
                # pick end-effector frame name
                if self.ee_link_name:
                    frame_name = self.ee_link_name
                else:
                    # try to infer from last arm joint's child link name
                    try:
                        pb_idx = self.robot._robot_joints[self.arm_indices[-1]]
                        frame_name = self.robot._link_names[pb_idx]
                    except Exception:
                        frame_name = None
                if frame_name is not None and frame_name in kinDyn.rbdalgos.model.links.keys():
                    fk_cas = kinDyn.forward_kinematics_fun(frame_name)
                    ee_pos = fk_cas(H_b, q)[:3, 3]
                    fk_fun = cs.Function("fk", [q], [ee_pos])
        except Exception:
            return fk_fun, inv_fun

        return fk_fun, inv_fun

    def _estimate_mass_matrix(self):
        """Estimate a constant mass matrix from the current PyBullet state."""
        if self.robot is None or not hasattr(self.robot, "_robot"):
            return None
        try:
            pb_joint_ids = [self.robot._robot_joints[idx] for idx in self.arm_indices]
            joint_states = p.getJointStates(self.robot._robot, self.robot._robot_joints)
            q_full = [js[0] for js in joint_states]
            M_full = np.array(p.calculateMassMatrix(self.robot._robot, q_full))
            # Slice to arm subspace
            M = M_full[np.ix_(pb_joint_ids, pb_joint_ids)]
            return M
        except Exception:
            return None

    def _estimate_bias_forces(self):
        """Estimate a constant bias term (e.g., gravity) from the current PyBullet state."""
        if self.robot is None or not hasattr(self.robot, "_robot"):
            return None
        try:
            pb_joint_ids = [self.robot._robot_joints[idx] for idx in self.arm_indices]
            joint_states = p.getJointStates(self.robot._robot, self.robot._robot_joints)
            q_full = [js[0] for js in joint_states]
            qd_full = [js[1] for js in joint_states]
            zero_acc = [0.0] * len(q_full)
            tau_full = np.array(p.calculateInverseDynamics(self.robot._robot, q_full, qd_full, zero_acc))
            tau = tau_full[pb_joint_ids]
            return tau
        except Exception:
            return None

    def current_ee_position(self):
        """
        Returns the current end-effector position from PyBullet if link index is known.
        """
        if self.ee_link_index is None or not hasattr(self.robot, "_robot"):
            return np.full(3, np.nan)
        state = p.getLinkState(self.robot._robot, self.ee_link_index, computeLinkVelocity=0)
        return np.array(state[0])
        
