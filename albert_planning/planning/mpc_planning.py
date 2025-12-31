import numpy as np
import casadi as cs
import pybullet as p


class BaseMPC:
    def __init__(self, dynamics, N, x_target, wx, wu, SOLVER_MAX_ITER=10, 
                 DO_WARM_START=True):
        """
        MPC Controller for differential drive robot
        
        Args:
            dynamics: DifferentialDriveDynamics object
            N: prediction horizon
            x_target: target state [x_des, y_des, theta_des]
            wx: state tracking weight
            wu: input weight
            SOLVER_MAX_ITER: max iterations for subsequent solves
            DO_WARM_START: enable warm start
        """
        self.dynamics = dynamics
        self.N = N
        self.x_target = x_target
        self.wx = wx
        self.wu = wu
        self.max_iter = SOLVER_MAX_ITER
        self.warm_start = DO_WARM_START
        
        # Create the optimization problem (done once)
        self.opti, self.X_var, self.U_var, self.p_x_init = self._create_optimization_problem()
    
    def _create_optimization_problem(self):
        """
        Create CasADi optimization problem structure
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
        
        # Minimize cost
        opti.minimize(cost)
        
        # Create the optimization solver with high max_iter for first solve
        print("Creating MPC optimization problem...")
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-4,             
            "ipopt.acceptable_tol": 1e-3,  
            "ipopt.acceptable_iter": 5,     # accept "good enough" solutions
        }
        opti.solver("ipopt", opts)
        
        # Solve the problem to convergence the first time
        print("Solving initial MPC problem to convergence...")
        opti.set_value(p_x_init, np.zeros(nx))
        opti.solve()
        print("Initial MPC problem solved!")
        
        # Set solver options for subsequent solves (lower max_iter)
        opts["ipopt.max_iter"] = self.max_iter
        opti.solver("ipopt", opts)
        
        return opti, X, U, p_x_init
    
    def solve(self, x_init):
        """
        Solve MPC problem for current state
        
        Args:
            x_init: current state [x, y, theta] (3,)
            
        Returns:
            u_opt: optimal input [v, omega] (2,)
            x_next: predicted next state (3,)
            x_traj: predicted state trajectory (3, N+1)
            theta_next: predicted next orientation (scalar)
        """
        
        # Debug: verify dimensions
        if x_init.shape[0] != 3:
            raise ValueError(f"Expected x_init to have 3 elements [x, y, theta], got {x_init.shape[0]}")
        
        try:
            # Set initial condition parameter
            self.opti.set_value(self.p_x_init, x_init)
            
            # Solve
            sol = self.opti.solve()
            
            # Warm start for next iteration
            # Use sol.value() directly after solving
            if self.warm_start:
                for k in range(self.N):
                    # Shift states: X[k] initial guess = current solution X[k+1]
                    self.opti.set_initial(self.X_var[k], sol.value(self.X_var[k + 1]))
                
                for k in range(self.N - 1):
                    # Shift inputs: U[k] initial guess = current solution U[k+1]
                    self.opti.set_initial(self.U_var[k], sol.value(self.U_var[k + 1]))
                
                # Initial guess for last state and control input
                self.opti.set_initial(self.X_var[-1], sol.value(self.X_var[-1]))
                self.opti.set_initial(self.U_var[-1], sol.value(self.U_var[-1]))
            
            # Extract solution
            u_opt = sol.value(self.U_var[0])
            x_next = sol.value(self.X_var[1])
            
            # Extract full trajectory for visualization
            x_traj = np.zeros((3, self.N + 1))
            for k in range(self.N + 1):
                x_traj[:, k] = sol.value(self.X_var[k])
            
            return u_opt, x_next, x_traj, x_next[2]
            
        except Exception as e:
            print(f"MPC solver failed: {e}")
            # Return safe defaults
            u_safe = np.zeros(2)
            x_traj = np.tile(x_init.reshape(-1, 1), (1, self.N + 1))
            
            # Make sure we return correct dimensions
            if len(x_init) == 3:
                return u_safe, x_init, x_traj, x_init[2]
            else:
                # Safety fallback
                x_init_fixed = np.array([x_init[0] if len(x_init) > 0 else 0.0,
                                        x_init[1] if len(x_init) > 1 else 0.0,
                                        0.0])
                return u_safe, x_init_fixed, x_traj, 0.0
            
class ArmMPC:
    """
    Acceleration-input MPC for a manipulator using ManipulatorDynamics.

    State x = [q, qdot], input u = qddot.
    Cost tracks end-effector position (via linearized FK), joint velocities,
    and accelerations, with terminal cost on end-effector position.
    Torque constraints are optional (off by default unless enforce_tau=True
    and inv_dyn is available).
    """

    def __init__(
        self,
        model,
        N,
        ee_target=np.zeros(3),
        w_p=50.0,
        w_v=1.0,
        w_a=1.0,
        w_final=None,
        u_min=None,
        u_max=None,
        SOLVER_MAX_ITER=50,
        DO_WARM_START=True,
        ENFORCE_TAU=False,
    ):
        self.model = model
        self.N = int(N)
        self.n = model.n
        self.dt = float(model.dt)
        self.ee_target = np.asarray(ee_target).reshape(3)
        self.w_p = float(w_p)
        self.w_v = float(w_v)
        self.w_a = float(w_a)
        self.w_final = float(w_final) if w_final is not None else 10.0 * float(w_p)
        self.max_iter = SOLVER_MAX_ITER
        self.warm_start = DO_WARM_START
        self.enforce_tau = ENFORCE_TAU and (self.model.inv_dyn is not None)
        self.pb_joint_ids = [self.model.robot._robot_joints[idx] for idx in self.model.arm_indices]
        self.end_link_index = self.model.ee_link_index if self.model.ee_link_index is not None else self.pb_joint_ids[-1]

        # State and torque bounds from the model
        self.q_min = np.asarray(model.q_min).reshape(self.n)
        self.q_max = np.asarray(model.q_max).reshape(self.n)
        self.qd_min = np.asarray(model.qd_min).reshape(self.n)
        self.qd_max = np.asarray(model.qd_max).reshape(self.n)
        self.tau_min = np.asarray(model.tau_min).reshape(self.n)
        self.tau_max = np.asarray(model.tau_max).reshape(self.n)

        # Acceleration bounds: prefer robot limits if available, else fallback
        if u_min is not None and u_max is not None:
            self.u_min = np.asarray(u_min).reshape(self.n)
            self.u_max = np.asarray(u_max).reshape(self.n)
        else:
            try:
                acc = model.robot._limit_acc_j
                self.u_min = np.array([acc[0, idx] for idx in model.arm_indices])
                self.u_max = np.array([acc[1, idx] for idx in model.arm_indices])
            except Exception:
                self.u_min = -np.ones(self.n) * 5.0
                self.u_max = np.ones(self.n) * 5.0

        (
            self.opti,
            self.X_var,
            self.U_var,
            self.p_x_init,
            self.p_ee_target,
            self.p_q_lin,
            self.p_ee_lin,
            self.p_J,
        ) = self._create_optimization_problem()

    def _create_optimization_problem(self):
        opti = cs.Opti()
        X = [opti.variable(2 * self.n) for _ in range(self.N + 1)]
        U = [opti.variable(self.n) for _ in range(self.N)]

        p_x_init = opti.parameter(2 * self.n)
        p_ee_target = opti.parameter(3)
        p_q_lin = opti.parameter(self.n)
        p_ee_lin = opti.parameter(3)
        p_J = opti.parameter(3, self.n)

        opti.subject_to(X[0] == p_x_init)

        cost = 0
        for k in range(self.N):
            qk = X[k][: self.n]
            qdk = X[k][self.n :]
            uk = U[k]

            opti.subject_to(opti.bounded(self.q_min, qk, self.q_max))
            opti.subject_to(opti.bounded(self.qd_min, qdk, self.qd_max))
            opti.subject_to(opti.bounded(self.u_min, uk, self.u_max))

            ee_pos = p_ee_lin + cs.mtimes(p_J, (qk - p_q_lin))
            cost += self.w_p * cs.sumsqr(ee_pos - p_ee_target)
            cost += self.w_v * cs.sumsqr(qdk)
            cost += self.w_a * cs.sumsqr(uk)

            opti.subject_to(X[k + 1] == self.model.discrete_f(X[k], uk))

            if self.enforce_tau:
                tau_k = self.model.inv_dyn(X[k], uk)
                opti.subject_to(opti.bounded(self.tau_min, tau_k, self.tau_max))

        qT = X[-1][: self.n]
        qdT = X[-1][self.n :]
        opti.subject_to(opti.bounded(self.q_min, qT, self.q_max))
        opti.subject_to(opti.bounded(self.qd_min, qdT, self.qd_max))

        ee_T = p_ee_lin + cs.mtimes(p_J, (qT - p_q_lin))
        cost += self.w_final * cs.sumsqr(ee_T - p_ee_target)

        opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-4,
            "ipopt.compl_inf_tol": 1e-4,
        }
        opti.solver("ipopt", opts)
        return opti, X, U, p_x_init, p_ee_target, p_q_lin, p_ee_lin, p_J

    def _full_q(self, q_arm):
        """
        Build joint vectors for all joints on the body (length = p.getNumJoints()).
        Overwrite arm joint slots with q_arm.
        """
        num_joints = p.getNumJoints(self.model.robot._robot)
        joint_ids = list(range(num_joints))
        joint_states = p.getJointStates(self.model.robot._robot, joint_ids)
        q_full = [js[0] for js in joint_states]  # length = num_joints
        for i_local, pb_idx in enumerate(self.pb_joint_ids):
            if 0 <= pb_idx < num_joints:
                q_full[pb_idx] = float(q_arm[i_local])
        return q_full

    def _linearized_fk(self, q_curr):
        if self.end_link_index is None:
            raise RuntimeError("End-effector link index is required for FK linearization.")
        # ensure link index is valid
        if not (0 <= self.end_link_index < p.getNumJoints(self.model.robot._robot)):
            raise RuntimeError(f"End-effector link index {self.end_link_index} invalid for body joints.")
        ee_state = p.getLinkState(self.model.robot._robot, self.end_link_index, computeLinkVelocity=0)
        ee_pos = np.array(ee_state[0])
        num_joints = p.getNumJoints(self.model.robot._robot)
        q_full = self._full_q(q_curr)
        zeros = [0.0] * num_joints
        J_t, _, _ = p.calculateJacobian(
            self.model.robot._robot,
            self.end_link_index,
            [0, 0, 0],
            list(q_full),
            zeros,
            zeros,
        )
        J_t = np.array(J_t)
        # J_t corresponds to joint indices 0..num_joints-1; pick arm columns
        cols = [pb_idx for pb_idx in self.pb_joint_ids if 0 <= pb_idx < J_t.shape[1]]
        J_arm = J_t[:, cols]
        return ee_pos, J_arm

    def solve(self, q_init, qd_init=None, ee_target=None):
        """
        Solve arm MPC for the current state.

        Args:
            q_init: joint positions (n,)
            qd_init: joint velocities (n,) (zeros if None)
            ee_target: desired EE position (3,) (defaults to stored ee_target)

        Returns:
            u_opt (qddot), q_next, qd_next, x_traj (2n x N+1)
        """
        q0 = np.asarray(q_init).reshape(self.n)
        qd0 = np.zeros(self.n) if qd_init is None else np.asarray(qd_init).reshape(self.n)
        if q0.shape[0] != self.n or qd0.shape[0] != self.n:
            raise ValueError("Initial state size mismatch with arm dimension")

        target = self.ee_target if ee_target is None else np.asarray(ee_target).reshape(3)
        ee_lin, J_lin = self._linearized_fk(q0)

        try:
            self.opti.set_value(self.p_x_init, np.concatenate([q0, qd0]))
            self.opti.set_value(self.p_ee_target, target)
            self.opti.set_value(self.p_q_lin, q0)
            self.opti.set_value(self.p_ee_lin, ee_lin)
            self.opti.set_value(self.p_J, J_lin)
            sol = self.opti.solve()

            if self.warm_start:
                for k in range(self.N):
                    self.opti.set_initial(self.X_var[k], sol.value(self.X_var[k + 1]))
                for k in range(self.N - 1):
                    self.opti.set_initial(self.U_var[k], sol.value(self.U_var[k + 1]))
                self.opti.set_initial(self.X_var[-1], sol.value(self.X_var[-1]))
                self.opti.set_initial(self.U_var[-1], sol.value(self.U_var[-1]))

            u_opt = np.array(sol.value(self.U_var[0])).reshape(self.n)
            x_next = np.array(sol.value(self.X_var[1])).reshape(2 * self.n)
            q_next = x_next[: self.n]
            qd_next = x_next[self.n :]

            x_traj = np.zeros((2 * self.n, self.N + 1))
            for k in range(self.N + 1):
                x_traj[:, k] = np.array(sol.value(self.X_var[k])).reshape(2 * self.n)

            return u_opt, q_next, qd_next, x_traj

        except Exception:
            u_safe = np.zeros(self.n)
            x0 = np.concatenate([q0, qd0])
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            return u_safe, q0, qd0, x_traj
