import numpy as np
import casadi as cs

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