import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# both r (position) and velocity should be 1d numpy arrays of size 3
def calculate_orbital_parameters(r, v, GMU):

    # Magnitudes of position and velocity vectors
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    # Calculate specific orbital energy
    epsilon = (v_mag**2) / 2 - GMU / r_mag

    # Check for the sign of epsilon
    if epsilon >= 0:
        raise ValueError("Positive specific orbital energy indicates an unbound orbit.")

    # Calculate semi-major axis (a)
    a = -GMU / (2 * epsilon)
    
    # Calculate specific angular momentum vector (h) and its magnitude
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    
    # Calculate eccentricity vector (e) and its magnitude
    e_vector = (np.cross(v, h) / GMU) - (r / r_mag)
    e = np.linalg.norm(e_vector)

    # Calculate periapsis and apoapsis distances
    r_peri = a * (1 - e)
    r_apo = a * (1 + e)

    # Return the orbital parameters
    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "periapsis_distance": r_peri,
        "apoapsis_distance": r_apo
    }

def plot_traj_around_planet(radius, xs, ys, zs, psteps = 50, xlims = [-1e7, 1e7], ylims = [-1e7, 1e7], zlims = [-1e7, 1e7],
                            elevation = 30, azimuth=-60):

    # Sphere parameters
    theta, phi = np.linspace(0, np.pi, psteps), np.linspace(0, 2 * np.pi, psteps)
    theta, phi = np.meshgrid(theta, phi)
    x_sphere = radius * np.sin(theta) * np.cos(phi)
    y_sphere = radius * np.sin(theta) * np.sin(phi)
    z_sphere = radius * np.cos(theta)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.5, edgecolor='none')

    # Plot the trajectory
    ax.plot(xs, ys, zs, color='red', linewidth=2, label='Trajectory')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    # Control the orientation of the axes
    # elevation: Angle above the horizontal plane
    # azimuth: Angle around the vertical axis
    ax.view_init(elev=elevation, azim=azimuth)

    # Labeling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rocket Trajectory')
    ax.legend()

def plot_predicted_orbit(orb_params, RADIUS, rmin=0, rmax=1e7, num_ticks = 5):

    a = orb_params['semi_major_axis']
    e = orb_params['eccentricity']
    theta = np.linspace(0,2*np.pi, 4000)
    r = (a*(1-e**2))/(1+e*np.cos(theta))

    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    theta_earth = np.linspace(0,2*np.pi, 360)
    ax.set_ylim(0, a + RADIUS)  # Adjust as needed based on your data

    ax.plot(theta_earth, np.full(360, RADIUS), color = 'g', label='Gravitational Body')
    ax.plot(theta, r, color = 'r', label='Rocket Trajectory')
    ax.set_rticks(np.linspace(rmin, rmax, num_ticks))
    ax.set_rmin(rmin)
    ax.set_rmax(rmax)
    ax.legend()