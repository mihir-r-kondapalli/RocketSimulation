import numpy as np
from rocket import Rocket, Motor, Orbit  # Assuming this handles the rocket physics
import copy

class RocketSimulation:
    def __init__(self, rocket, RADIUS, GM, target, max_iters):

        self.rocket = rocket
        self.init_rocket = copy.deepcopy(rocket)
        self.RADIUS = RADIUS
        self.GM = GM
        self.target = target
        self.max_iters = max_iters
        self.iters = 0

        # Rocket state: [a, e, inc, long_asc, omega, nu, delta_v]
        self.init_state = np.array([self.rocket.orbit.a, self.rocket.orbit.e, self.rocket.orbit.inc, self.rocket.orbit.long_asc,
                                    self.rocket.orbit.omega, self.rocket.orbit.nu])
        self.state = self.init_state
        # Simulation timestep
        self.dt = 0.05  # Time step (seconds)

    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        self.state = self.init_state
        self.rocket = self.init_rocket
        return self.get_state()

    def update(self, action):
        """
        Update the simulation by applying an action.
        Parameters:
        - action (array): [throttle, pitch, yaw]
        Returns:
        - next_state (array): Updated state of the rocket
        - done (bool): Whether the simulation is complete
        """

        throttle, pitch, yaw = action[0]

        # Set direction (rocket direction is 3d unit vector)
        self.rocket.set_dir_yaw_pitch(yaw, pitch)

        # Limit throttle to valid range
        throttle = (throttle+1)/2
        self.rocket.throttle(throttle)

        # Update rocket parameters
        self.rocket.update_pos(self.dt)
        self.rocket.update_orbit()
        self.rocket.update_mass(self.dt)

        # Update dt
        self.update_dt()

        # Update state
        self.state = np.array([self.rocket.orbit.a, self.rocket.orbit.e, self.rocket.orbit.inc, self.rocket.orbit.long_asc,
                                    self.rocket.orbit.omega, self.rocket.orbit.nu])
        
        self.iters += 1

        # Check termination conditions
        end_state = self.get_end_state()
        return self.get_state(), end_state

    def update_dt(self):
        if self.rocket.get_throttle() > 0 and self.rocket.motor.get_wet_mass() > 0:
            self.dt = 0.01
        elif self.rocket.get_altitude() < 1e2:
            self.dt = 0.01
        if self.rocket.get_altitude() < 1e3:
            self.dt = 0.05
        elif self.rocket.get_altitude() < 1e4:
            self.dt = 0.5
        elif self.rocket.get_altitude() < 1e5:
            self.dt = 1
        elif self.rocket.get_altitude() < 1e6:
            self.dt = 5
        elif self.rocket.get_altitude() < 1e7:
            self.dt = 10

    def get_initial_state(self):
        return self.init_state

    def get_state(self):
        return self.state
    
    def get_error(self, w_a=10, w_e=10, w_i=1, w_long_asc=1, w_omega=1, w_nu=0):
        param_error = self.rocket.orbit.get_error(self.target, self.GM, w_a, w_e, w_i, w_long_asc, w_omega, w_nu)
        delta_v_error = self.rocket.get_transfer_delta_v_error(self.target)
        return param_error + delta_v_error
    
    def get_end_state(self):
        # takes too long
        if(self.iters >= self.max_iters):
            return 0
        # runs out of fuel
        elif(self.rocket.motor.get_wet_mass() <= 0):
            return 1
        # crashes
        if(self.rocket.get_radial_dist() < self.RADIUS):
            return 2
        # leaves system
        if(self.rocket.orbit.e >= 1):
            return 3
        else:
            return -1



class RocketEnv:
    def __init__(self, max_iters, init_burn = 5):

        # Planet Parameters (Earth Moon)
        RADIUS = np.float64(1738.1e3) # m
        ROT_SPEED = 2.654e-6          # radians/second
        GM = np.float64(4.9048695e12) # m^3/s^2
        target = Orbit(np.array([0, 0, 4e6]), np.array([1107.5, 0, 0]), GM)

        # Initialize the rocket simulation
        mass_rocket = 0               # kg (0 for now just putting all dry mass on motor weight)
        ap11_motor = Motor(2287, 4200, 4.33, 3600) # dry mass (kg), wet mass(kg), fuel ejection rate (kg/s), fuel ejection speed (m/s)
        initial_pos = np.array([0, 0, RADIUS])
        initial_vel = np.array([ROT_SPEED*RADIUS, 0, 0])
        initial_dir = np.array([0, 0, 1])
        self.RADIUS = RADIUS
        self.rocket = Rocket(mass_rocket, initial_pos, initial_vel, initial_dir, ap11_motor, GM, RADIUS)
        self.sim_beginning(init_burn)


        self.simulation = RocketSimulation(self.rocket, RADIUS, GM, target, max_iters)

        # Define the state space and action space dimensions
        self.state_dim = 6  # Rocket state: [a, e, inc, long_asc, omega, nu]
        self.action_dim = 3  # Action: [throttle, pitch, yaw]

        # State and action limits
        self.state = np.zeros(self.state_dim)  # Placeholder for the initial state
        self.error = 1e8
        self.action_bounds = np.array([
            [-1.0, 1.0],  # Throttle: [0% to 100%]
            [-1.0, 1.0],  # Pitch: [-1 to 1]
            [-1.0, 1.0],  # Yaw: [-1 to 1]
        ])

    def sim_beginning(self, init_burn):

        self.rocket.throttle(1)
        dt = init_burn / 1000
        while self.rocket.clock < init_burn:
            self.rocket.update_pos(dt)
            self.rocket.update_orbit()
            self.rocket.update_mass(dt)
        self.rocket.throttle(0)

    def reset(self):
        """Reset the environment to its initial state."""
        self.simulation.reset()  # Reset the rocket simulation
        self.state = self.simulation.get_initial_state()  # Get the initial state from the simulation
        return self.state

    def step(self, action):
        """Apply an action and return the next state, reward, and done flag."""
        # Clip actions to valid bounds
        action = np.clip(action, self.action_bounds[:, 0], self.action_bounds[:, 1])

        # Apply the action in the simulation
        next_state, end_state = self.simulation.update(action)
        self.error = self.simulation.get_error()

        # Calculate reward (e.g., based on progress to orbit and fuel efficiency)
        reward = self.calculate_reward(next_state, end_state)

        # Update the state
        self.state = next_state

        done = end_state != -1

        return next_state, reward, done

    def calculate_reward(self, state, end_state):

        reward = -self.error

        """Define a reward function based on the current state."""
        if end_state == 0:   # takes too many iterations
            reward+=0
        elif end_state == 1: # runs out of fuel
            reward+=0
        elif end_state == 2: # crashes
            reward+=0
        elif end_state == 3: # leaves system
            reward+=0
        else:
            reward+=0

        return reward

    def render(self):
        """Optional: Render the current state."""
        print(f"State: {self.state}")
        print(f"Error: {self.error}")
        print()

    def get_state(self):

        return {
            "x": self.rocket.pos[0],
            "y": self.rocket.pos[1],
            "z": self.rocket.pos[2],
            "vx": self.rocket.vel[0],
            "vy": self.rocket.vel[1],
            "vz": self.rocket.vel[2],
            "delta_v": self.rocket.get_delta_v(),  # Current delta-v
            "fuel_mass": self.rocket.motor.get_wet_mass(),  # Remaining fuel mass
            "throttle": self.rocket.get_throttle(),  # Current throttle setting
            "altitude": self.rocket.get_altitude(),  # Rocket altitude
            "perigee": self.rocket.get_perigee(),  # Perigee of the orbit
            "apogee": self.rocket.get_apogee(),  # Apogee of the orbit
            "clock": self.rocket.clock,  # Current time
        }


class RocketLog:

    def __init__(self, rocketenv):
        self.rocketenv = rocketenv

        self.state_log = {
            "x": [],
            "y": [],
            "z": [],
            "vx": [],
            "vy": [],
            "vz": [],
            "delta_v": [],
            "fuel_mass": [],
            "throttle": [],
            "altitude": [],
            "perigee": [],
            "apogee": [],
            "clock": [],
        }


    def log(self):
        state = self.rocketenv.get_state()
        print(state)
        for key, value in state.items():
            self.state_log[key].append(value)

    def get_state_log(self):
        return self.state_log