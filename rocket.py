import numpy as np
import numpy.linalg as npla
from integration_methods import rk4_step, get_orb_params

class Orbit:
    def __init__(self, r, v, GM):
        self.a, self.e, self.perigee, self.apogee, self.inc, self.long_asc, self.omega, self.nu = get_orb_params(r, v, GM)

    def update(self, r, v, GM):
        self.a, self.e, self.perigee, self.apogee, self.inc, self.long_asc, self.omega, self.nu = get_orb_params(r, v, GM)

    # target is target orbit object
    def get_tansfer_delta_v(self, target, GM):
        r1 = self.a * (1-np.power(self.e, 2)) / (1 + self.e * np.cos(self.nu))
        v1 = np.sqrt(GM * (2/r1 - 1/self.a))
        r2 = target.a * (1-np.power(target.e, 2)) / (1 + target.e * np.cos(target.nu))
        v2 = np.sqrt(GM * (2/r2 - 1/target.a))
        delta_inc = target.inc - self.inc
        delta_v = np.sqrt(np.power(v1 - v2, 2) + np.power(2 * v1 * np.sin(delta_inc/2), 2))
        return delta_v

    # target is target orbit object
    def get_error(self, target, GM, w_a, w_e, w_i, w_long_asc, w_omega, w_nu):

        # Compute normalized orbital parameter errors

        error_a = abs(target.a - self.a) / target.a
        error_e = abs(target.e - self.e)

        error_i = abs(np.cos(target.inc) - np.cos(self.inc)) / 2

        error_long_asc = abs((target.long_asc - self.long_asc) % (2 * np.pi))
        error_long_asc = min(error_long_asc, 2 * np.pi - error_long_asc) / (2*np.pi)

        error_omega = abs((target.omega - self.omega) % (2 * np.pi))
        error_omega = min(error_omega, 2 * np.pi - error_omega) / (2*np.pi)

        error_nu = abs((target.nu - self.nu) % (2 * np.pi))
        error_nu = min(error_nu, 2 * np.pi - error_nu) / (2*np.pi)

        # Combine the errors using weights
        param_error = (w_a * error_a +
                    w_e * error_e +
                    w_i * error_i +
                    w_long_asc * error_long_asc +
                    w_omega * error_omega +
                    w_nu * error_nu)

        return param_error

class Motor:

    def __init__(self, dry_mass, wet_mass, fuel_ejection_rate, fuel_ejection_speed):
        self.dry_mass = dry_mass
        self.wet_mass = wet_mass
        self.fuel_ejection_rate = fuel_ejection_rate
        self.fuel_ejection_speed = fuel_ejection_speed
        self.throttle = 0

    def get_mass(self):
        return self.dry_mass + self.wet_mass
    
    def get_fuel_ejection_speed(self):
        if self.wet_mass <= 0: return 0
        else: return self.fuel_ejection_speed

    def set_throttle(self, throttle):
        if(throttle <= 0):
            self.throttle = 0
        elif(throttle >= 1):
            self.throttle = 1
        else:
            self.throttle = throttle
    
    def get_fuel_ejection_rate(self):
        if self.wet_mass <= 0: return 0
        else: return self.fuel_ejection_rate*self.throttle

    def get_wet_mass(self):
        return self.wet_mass

class Rocket:

    def __init__(self, mass, pos, vel, dir, motor, GM, RADIUS, init_time = np.float32(0)):

        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.accel = np.zeros(3)
        self.dir = dir/npla.norm(dir)
        self.motor = motor
        self.clock = init_time
        self.GM = GM
        self.orbit = Orbit(self.pos, self.vel, self.GM)
        self.RADIUS = RADIUS
        self.init_dv = self.get_delta_v()

    def get_mass(self):
        return self.mass + self.motor.get_mass()
    
    def get_dry_mass(self):
        return self.mass + self.motor.dry_mass
    
    def update_pos(self, dt):
        self.pos, self.vel, self.accel = rk4_step(self.get_mass(), self.clock, dt, self.pos, self.vel,
                                                  self.motor.get_fuel_ejection_speed(),
                                                  self.motor.get_fuel_ejection_rate(), self.dir, self.GM)
        self.clock += dt

    def update_mass(self, dt):
        if(self.motor.wet_mass > 0):
            self.motor.wet_mass -= self.motor.get_fuel_ejection_rate() * dt
        else:
            self.motor.wet_mass = 0

    def get_radial_dist(self):
        return npla.norm(self.pos)
    
    def get_radial_dir(self):
        return self.pos/npla.norm(self.pos)
    
    def get_tangential_vel(self):
        v_radial = (np.dot(self.vel, self.pos) / np.dot(self.pos, self.pos)) * self.pos
        return self.vel - v_radial
    
    def get_altitude(self):
        return self.get_radial_dist() - self.RADIUS
    
    def get_speed(self):
        return npla.norm(self.vel)
    
    def get_delta_v(self):
        return -self.motor.get_fuel_ejection_speed()*np.log(self.get_dry_mass()/self.get_mass())
    
    def throttle(self, throttle):
        self.motor.set_throttle(throttle)

    def get_thrust(self):
        return self.motor.get_fuel_ejection_speed() * self.motor.get_fuel_ejection_rate()

    def get_throttle(self):
        return self.motor.throttle
    
    def update_orbit(self):
        self.orbit.update(self.pos, self.vel, self.GM)

    def get_perigee(self):
        return self.orbit.perigee
    
    def get_apogee(self):
        return self.orbit.apogee
    
    def set_dir(self, dir):
        self.dir = dir / npla.norm(dir)

    def get_dir(self):
        return self.dir
    
    def reset_clock(self):
        self.clock = 0

    # Assuming yaw and pitch are ranged [-1, 1]
    def set_dir_yaw_pitch(self, yaw, pitch):
        # Map yaw and pitch to spherical angles
        phi = yaw * np.pi  # Azimuthal angle (yaw)
        theta = (1 - pitch) * np.pi / 2  # Polar angle (pitch)

        # Convert to Cartesian coordinates
        self.dir[0] = np.cos(phi) * np.sin(theta)
        self.dir[1] = np.sin(phi) * np.sin(theta)
        self.dir[2] = np.cos(theta)

    def get_transfer_delta_v_error(self, target):
        return (self.orbit.get_tansfer_delta_v(target, self.GM) / self.init_dv)