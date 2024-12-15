import numpy as np
import numpy.linalg as npla

class Rocket_Comp:

    def __init__(self, rocket, target):
        self.rocket = rocket
        self.target = target

    def get_tansfer_delta_v(self, nu):
        a = self.rocket.orbit.a
        e = self.rocket.orbit.e
        inc = self.rocket.orbit.inc

        r1 = a * (1-np.power(e, 2)) / (1 + e * np.cos(nu))
        v1 = np.sqrt(self.GM * (2/r1 - 1/a))
        r2 = self.target.a * (1-np.power(self.target.e, 2)) / (1 + self.target.e * np.cos(self.target.nu))
        v2 = np.sqrt(self.GM * (2/r2 - 1/self.target.a))
        delta_inc = self.target.inc - inc
        delta_v = np.sqrt(np.power(v1 - v2, 2) + np.power(2 * v1 * np.sin(delta_inc/2), 2))
        return delta_v
    
    def get_time_till_nu(self, nu):

        e = self.rocket.orbit.e
        n = np.sqrt(self.rocket.GM / np.power(self.rocket.orbit.a, 3))
        E_rocket = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(self.rocket.orbit.nu/2))
        M_rocket = E_rocket - e * np.sin(E_rocket)

        E_target = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu/2))
        M_target = E_target - e * np.sin(E_target)

        return (M_target - M_rocket) / n

    def calculate_burn_time(self, vi, vf):
        time = (self.rocket.get_mass() * (1 - np.exp((npla.norm(vi)-npla.norm(vf)) / self.rocket.motor.fuel_ejection_speed)) 
                / self.rocket.motor.fuel_ejection_rate)
        return time
    
    def get_rotation_matrix(self):
        # Precompute trigonometric values
        Omega = self.rocket.orbit.long_asc
        omega = self.rocket.orbit.omega
        inc = self.rocket.orbit.inc

        # Rotation matrix from perifocal frame to inertial frame
        cos_Omega = np.cos(Omega)
        sin_Omega = np.sin(Omega)
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        cos_i = np.cos(inc)
        sin_i = np.sin(inc)

        R = np.array([
            [cos_omega * cos_Omega - sin_omega * sin_Omega * cos_i,
            -sin_omega * cos_Omega - cos_omega * sin_Omega * cos_i,
            sin_Omega * sin_i],

            [cos_omega * sin_Omega + sin_omega * cos_Omega * cos_i,
            sin_omega * sin_Omega + cos_omega * cos_Omega * cos_i,
            cos_Omega * sin_i],

            [sin_omega * sin_i,
            cos_omega * sin_i,
            cos_i]
        ])

        return R
    
    # 0 for periapsis, pi for apoapsis
    def get_tangential_vel_at_nu(self, nu):
        
        p = self.rocket.orbit.a * (1 - np.power(self.rocket.orbit.e, 2))
        const = np.sqrt(self.GM / p)
        vp = np.array([-const * np.sin(nu), const*(self.rocket.orbit.e + np.cos(nu)), 0])
        return self.get_rotation_matrix() @ vp
    
    def get_orbital_velocity_at_nu(self, nu):
        # Semi-latus rectum
        p = self.rocket.orbit.a * (1 - np.power(self.rocket.orbit.e, 2))
        
        # Orbital velocity components
        vr = np.sqrt(self.rocket.GM / p) * self.rocket.orbit.e * np.sin(nu)
        vtheta = np.sqrt(self.rocket.GM / p) * (1 + self.rocket.orbit.e * np.cos(nu))
        
        # Orbital plane velocity vector
        v_orbital = np.array([
            vr * np.cos(nu) - vtheta * np.sin(nu),  # x-component in orbital plane
            vr * np.sin(nu) + vtheta * np.cos(nu),  # y-component in orbital plane
            0  # z-component in orbital plane
        ])

        return v_orbital

    def get_total_velocity_at_nu(self, nu):

        return self.get_rotation_matrix() @ self.get_orbital_velocity_at_nu(nu)
    
    def get_total_radius_at_nu(self, nu):
        # Calculate the position in the orbital frame (perifocal coordinates)
        r = (self.rocket.orbit.a * (1 - np.power(self.rocket.orbit.e, 2))) / (1 + self.rocket.orbit.e * np.cos(nu))
        x_orbital = r * np.cos(nu)
        y_orbital = r * np.sin(nu)
        z_orbital = 0  # Since we're in the orbital plane, z is 0
        
        # Orbital position vector in the orbital frame
        r_orbital = np.array([x_orbital, y_orbital, z_orbital])

        return self.get_rotation_matrix() @ r_orbital
    
    #def get_launch_burn(self):
    #    dir = lambda x: np.array([0, 1, 0])

    def get_circular_burn(self, nu, target_a):
        speed_orb = np.sqrt(self.rocket.GM / target_a)

        v_tar_intertial = self.get_rotation_matrix() @ self.get_orbital_velocity_at_nu(nu)
        target_v = speed_orb * v_tar_intertial / npla.norm(v_tar_intertial)

        orig_v = self.get_total_velocity_at_nu(nu)
        burn_time = np.abs(self.calculate_burn_time(orig_v, target_v))
        start_time = self.rocket.clock + self.get_time_till_nu(nu)
        delta_v = target_v - orig_v
        print("Required: " + str(npla.norm(delta_v)) + ", " + str(target_v))
        print("Current: " + str(self.rocket.get_delta_v()) + ", " + str(orig_v))
        if(npla.norm(delta_v) > self.rocket.get_delta_v()):
            print("NOT ENOUGH FUEL")
            return None
        burn_dir = delta_v/npla.norm(delta_v)
        # Equal time on both sides of burn
        return Burn(start_time-burn_time/2, start_time+burn_time/2, burn_dir, 1)


class Burn(object):
    
    def __init__(self, init_t, final_t, dir, throttle):
        self.init_t = init_t
        self.final_t = final_t
        self.dir = dir
        self.throttle = throttle

    def get_dir(self):
        return self.dir

    def get_throttle(self):
        return self.throttle

class Dynamic_Burn(Burn):

    def __init__(self, init_t, final_t, dir, throttle):
        super().__init__(init_t, final_t, dir, throttle)

    def get_dir(self, *args):
        return self.dir(args)

    def get_throttle(self, *args):
        return self.throttle(args)