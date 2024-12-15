import numpy as np
import numpy.linalg as npla

def get_thrust(fuel_ejection_speed, fuel_ejection_rate, dir):
        return fuel_ejection_speed * fuel_ejection_rate * dir
    
def drag_force(v):
    return np.zeros(3)

def gravity(mass, pos, vel, GM):
    return - pos * GM * mass / np.power(npla.norm(pos), 3)

# Acceleration function, accounting for thrust, drag, and gravity
def acceleration(mass, t, position, velocity, fuel_ejection_speed, fuel_ejection_rate, dir, GM):
    # Calculate net force
    thrust = get_thrust(fuel_ejection_speed, fuel_ejection_rate, dir)
    drag = drag_force(velocity)
    grav = gravity(mass, position, velocity, GM)
    net_force = thrust + drag + grav
    return net_force / mass  # Newton's second law: F = ma, so a = F/m

# RK4 integration function
def rk4_step(mass, t, dt, position, velocity, fuel_ejection_speed, fuel_ejection_rate, dir, GM):
    # Calculate the k values for position and velocity using RK4
    k1_v = acceleration(mass, t, position, velocity, fuel_ejection_speed, fuel_ejection_rate, dir, GM) * dt
    k1_r = velocity * dt

    k2_v = acceleration(mass, t + dt / 2, position + k1_r / 2, velocity + k1_v / 2, fuel_ejection_speed, fuel_ejection_rate, dir, GM) * dt
    k2_r = (velocity + k1_v / 2) * dt

    k3_v = acceleration(mass, t + dt / 2, position + k2_r / 2, velocity + k2_v / 2, fuel_ejection_speed, fuel_ejection_rate, dir, GM) * dt
    k3_r = (velocity + k2_v / 2) * dt

    k4_v = acceleration(mass, t + dt, position + k3_r, velocity + k3_v, fuel_ejection_speed, fuel_ejection_rate, dir, GM) * dt
    k4_r = (velocity + k3_v) * dt

    # Update velocity and position
    velocity_new = velocity + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    position_new = position + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6

    # returns new position, velocity, and acceleration
    return position_new, velocity_new, k4_v/dt

def get_orb_params(r, v, GMU):
    # Magnitudes of position and velocity vectors
    r_mag = npla.norm(r)
    v_mag = npla.norm(v)

    # Specific angular momentum vector
    h = np.cross(r, v)
    h_mag = npla.norm(h)

    # Inclination (i)
    i = np.arccos(h[2] / h_mag)

    # Node vector
    k = np.array([0, 0, 1])  # Unit vector in z-direction
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)

    # Longitude of the Ascending Node (Omega)
    if(n_mag!=0):
        Omega = np.arccos(n[0] / n_mag)
        if(n[1] < 0):
             Omega = 2 * np.pi - Omega
    else:
         Omega = 0

    # Eccentricity vector
    e_vec = (1 / GMU) * ((v_mag**2 - GMU / r_mag) * r - np.dot(r, v) * v)
    e = npla.norm(e_vec)

    # Argument of Periapsis (omega)
    if(n_mag != 0 and e > 0):
        omega = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if(e_vec[2] < 0): 
            omega = 2 * np.pi - omega
    else:
         omega = 0

    # True Anomaly (nu)
    if(e > 0):
        nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if(np.dot(r, v) < 0): 
            nu = 2 * np.pi - nu
    else:
         nu = 0

    # Semi-major axis (a)
    specific_energy = (v_mag**2) / 2 - GMU / r_mag
    if(specific_energy < 0):
        a = -GMU / (2 * specific_energy)
    else:
        a = np.inf

    # Returns orbital parameters along with periapsis and apoapsis
    return a, e, a*(1-e), a*(1+e), i, Omega, omega, nu