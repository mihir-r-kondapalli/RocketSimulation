import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla

@jax.jit
def get_thrust(fuel_ejection_speed, fuel_ejection_rate, dir):
        return fuel_ejection_speed * fuel_ejection_rate * dir
    
@jax.jit
def drag_force(v):
    return jnp.zeros(3)

@jax.jit
def gravity(mass, pos, vel, GM):
    return - pos * GM * mass / jnp.power(jnpla.norm(pos), 3)

# Acceleration function, accounting for thrust, drag, and gravity
@jax.jit
def acceleration(mass, t, position, velocity, fuel_ejection_speed, fuel_ejection_rate, dir, GM):
    # Calculate net force
    thrust = get_thrust(fuel_ejection_speed, fuel_ejection_rate, dir)
    drag = drag_force(velocity)
    grav = gravity(mass, position, velocity, GM)
    net_force = thrust + drag + grav
    return net_force / mass  # Newton's second law: F = ma, so a = F/m

# RK4 integration function
@jax.jit
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

@jax.jit
def get_orb_params(r, v, GMU):
    # Magnitudes of position and velocity vectors
    r_mag = jnpla.norm(r)
    v_mag = jnpla.norm(v)

    # Specific angular momentum vector
    h = jnp.cross(r, v)
    h_mag = jnpla.norm(h)

    # Inclination (i)
    i = jnp.arccos(h[2] / h_mag)

    # Node vector
    k = jnp.array([0, 0, 1])  # Unit vector in z-direction
    n = jnp.cross(k, h)
    n_mag = jnp.linalg.norm(n)

    # Longitude of the Ascending Node (Omega)
    Omega = jnp.where(n_mag!=0, jnp.arccos(n[0] / n_mag), 0)
    Omega = jnp.where(n[1] < 0, 2 * jnp.pi - Omega, Omega)

    # Eccentricity vector
    e_vec = (1 / GMU) * ((v_mag**2 - GMU / r_mag) * r - jnp.dot(r, v) * v)
    e = jnpla.norm(e_vec)

    # Argument of Periapsis (omega)
    omega = jnp.where(jnp.logical_and(n_mag != 0, e > 0), jnp.arccos(jnp.dot(n, e_vec) / (n_mag * e)), 0)
    omega = jnp.where(e_vec[2] < 0, 2 * jnp.pi - omega, omega)

    # True Anomaly (nu)
    nu = jnp.where(e>0, jnp.arccos(jnp.dot(e_vec, r) / (e * r_mag)), 0)
    nu = jnp.where(jnp.dot(r, v) < 0, 2 * jnp.pi - nu, nu)

    # Semi-major axis (a)
    specific_energy = (v_mag**2) / 2 - GMU / r_mag
    a = jnp.where(specific_energy < 0, -GMU / (2 * specific_energy), jnp.inf)

    # Returns orbital parameters along with periapsis and apoapsis
    return a, e, a*(1-e), a*(1+e), i, Omega, omega, nu