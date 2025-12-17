import numpy as np
import casadi as cs


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