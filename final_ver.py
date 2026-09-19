'''
This program models the swim start of a swimmer in four phases: take off,
projectile motion, underwater, and surface swimming. The underwater phase is
modelled using three numerical methods. Graphs of the position of the swimmer
in the length and depth axis is plotted as well as the total time of swimming
depending on the initial take-off angle and its residual plot.

Instructions: Ensure the installation of all dependencies including numpy, scipy, matplotlib, and math
Run the code directly to obtain graphs for timestep of h = 0.01.
To see Figure 1a (timestep h = 0.1), please go to line 347 and change the step size h.

Author: Jessie Guo and Josephine Yang
Last edited: 2025.12.18
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

'''Global Constants'''
# These constants are used in multiple phases of the simulation
g = 9.81 # gravity constant (m/s^2)


'''
Phase 1: Block take off  =====================================================================
'''
def take_off():
    '''
    This function outputs the time taken for takeoff from the starting block.
    The time is assumed to be constant for all swimmers so the function does
    not need an input.

    Input: None
    Output: time1 (float) as the time taken during the takeoff phase.
    '''
    # Assumption: take off is a constant 0.8 s from the start signal to
    # both feet leaving the block, as established from past studies
    # print("Phase 1 (Take-off):")
    # print("  Time = 0.8 s")
    # print("  Horizontal displacement = 0 m")
    time1 = 0.8 # seconds

    return time1

'''
Phase 2: Projectile Motion in Air =====================================================================
'''
def in_air(launch_angle=15):
    '''
    This function models the in-air phase of the swim start using fundamental
    projectile motion equations.

    Inputs: launch_angle (float) as the takeoff angle in degrees. This parameter
            will be an input parameter in optimization. CHANGE this to see
            impact of angle on takeoff.

    Outputs: time2 (float) as the time taken during in-air flight,
            flight_x (array of floats) as the x-coordinates during flight,
            flight_y (array of floats) as the y-coordinates during flight, and
            impact_velocity_x2, impact_velocity_y2 (floats) as the components
            of the impact velocity at the water entry point
    '''

    ini_speed = 4.0762 # m/s (takeoff speed for average female athlete from cited study)
    launch_angle_rad = math.radians(launch_angle) # convert launch angle to radians
    y0_block = 0.625 # metres (block height above water)

    # Initial vertical velocity components
    vy0_2 = ini_speed * math.sin(launch_angle_rad)
    vx0_2 = ini_speed * math.cos(launch_angle_rad)

    # time taken during in-air flight using projectile motion equations
    time2 = (vy0_2 + math.sqrt(vy0_2**2 + 2 * g * y0_block)) / g

    # air trajectory for plotting
    flight_t = np.linspace(0, time2, 200) # array of 200 time points during flight phase
    flight_x = vx0_2 * flight_t # array of x-coordinates during flight phase
    flight_y = y0_block + vy0_2 * flight_t - 0.5 * g * flight_t**2 # array of y-coordinates during flight phase

    # impact velocities (using projectile motion equations)
    impact_velocity_y2 = vy0_2 - g * time2
    impact_velocity_x2 = vx0_2
    impact_speed = math.sqrt(impact_velocity_y2**2 + impact_velocity_x2**2)

    # print phase 2 outputs
    # print("Phase 2 (Flight):")
    # print("  Time =", time2, "s")
    # print("  Horizontal displacement =", flight_x[-1], "m")
    # print("  Impact velocity =", impact_speed, "m/s")

    return time2, flight_x, flight_y, impact_velocity_x2, impact_velocity_y2

'''
Phase 3: Underwater Simulation NUMERICAL METHODS =====================================================================
'''
# create constants used in Phase 3
m = 55 # kg (mass of an average female competitive swimmer from a study)
rho = 997 # kg/m^3 (water density)
Volume = 0.06 # m^3 (approximate volume of the swimmer calculated using mass/density)
Cd = 0.7 # dimensionless (mean drag coefficient from a study on 40 female university swimmers)

# swimmer geometry modelled as a cylinder
L = 1.7 # m (body length, taken from primary data)
d = 0.30 # m (effective diameter, taken from primary data)

def projected_area(vx, vy, theta):
    """
    This function calculates the projected frontal cross sectional area of
    a swimmer modelled as a rigid cylinder moving through water. The area
    depends on the angle between the instantaneous velocity and swimmer's
    body axis.

    Inputs: vx and vy (floats) as the x and y component of velocity;
            theta (float) as the orientation angle of the swimmer body axis
            from the positive x direction.

    Output: A_proj (float) as the effective projected area normal to the
            direction of motion that directly impacts drag.
    """
    # calculate speed, if zero, default case
    v = np.sqrt(vx**2 + vy**2)
    if v == 0:
        return np.pi * d**2 / 4

    # find unit vector, velocity axis, and angle between axis.
    v_hat = np.array([vx, vy]) / v
    axis = np.array([np.cos(theta), np.sin(theta)])
    alpha = np.arccos(np.clip(np.dot(v_hat, axis), -1.0, 1.0))

    A_end = np.pi * d**2 / 4
    A_side = d * L
    return A_end * np.cos(alpha)**2 + A_side * np.sin(alpha)**2

def acceleration(vx, vy, theta):
    """
    This function compute the acceleration of the swimmer in x and y direction based
    on force equilibrium of drag, gravity, and buoyancy. Drag is modelled to be
    quadratic in speed and acts opposite to the direction of motion. Its magnitude
    depends on the projected frontal area, which varies with swimmer orientation.

    Inputs: vx and vy (floats) as the x and y component of velocity;
            theta (float) as the orientation angle of the swimmer body axis
            from the positive x direction.

    Outputs: ax, ay (floats) are the acceleration components of the swimmer in the
             x and y direction.
    """
    # calculate drag related coefficients
    v = np.sqrt(vx**2 + vy**2)
    A_proj = projected_area(vx, vy, theta)
    k = 0.5 * rho * Cd * A_proj

    # compute acceleration based on force equilibrium
    ax = -k * v * vx / m
    ay = -k * v * vy / m - g + rho * g * Volume / m
    return ax, ay

# Standard Euler method
def euler(x0, y0, vx0, vy0, t0, h):
    '''
    Forward Euler for 2D motion with drag, gravity, and buoyancy to model underwater
    behavior of the swimming. This model assumes no contribution from underwater kicks.
    Swimmer orientation is updated to horizontal after the lowest underwater point to
    better simulate real life swimming bahavior.
    '''

    # initialize solution lists, counter and flag for minimum.
    t_list = [t0]; x_list = [x0]; y_list = [y0]
    vx_list = [vx0]; vy_list = [vy0]
    n = 0
    passed_minimum = False

    # main loop
    while True:
        # current state data
        x, y, vx, vy, t = x_list[n], y_list[n], vx_list[n], vy_list[n], t_list[n]

        # check for minimum position
        if not passed_minimum and vy > 0:
            passed_minimum = True

        # update orientation depending on position.
        if passed_minimum:
            theta = 0.0
        else:
            theta = math.atan2(vy, vx)

        # call acceleration function to extract current acceleration.
        ax, ay = acceleration(vx, vy, theta)

        # Euler updates
        x_new = x + h * vx
        y_new = y + h * vy
        vx_new = vx + h * ax
        vy_new = vy + h * ay
        t_new = t + h

        # update list values
        x_list.append(x_new); y_list.append(y_new)
        vx_list.append(vx_new); vy_list.append(vy_new)
        t_list.append(t_new)
        n += 1

        # check if swimmer reaches water level
        if y_new >= 0 and t_new > 0.05:
            break

    return np.array(t_list), np.array(x_list), np.array(y_list), np.array(vx_list), np.array(vy_list)

def heun(x0, y0, vx0, vy0, t0, h):
    '''
    Improved Euler for 2D motion with drag, gravity, and buoyancy to model underwater
    behavior of the swimming. This model assumes no contribution from underwater kicks.
    Swimmer orientation is updated to horizontal after the lowest underwater point to
    better simulate real life swimming bahavior.
    '''

     # initialize solution lists, counter and flag for minimum.
    t_list = [t0]; x_list = [x0]; y_list = [y0]
    vx_list = [vx0]; vy_list = [vy0]
    n = 0
    passed_minimum = False

    # main loop
    while True:
        # current state data
        x, y, vx, vy, t = x_list[n], y_list[n], vx_list[n], vy_list[n], t_list[n]

        # check for minimum position
        if not passed_minimum and vy > 0:
            passed_minimum = True

        # update orientation depending on position.
        if passed_minimum:
            theta = 0.0
        else:
            theta = math.atan2(vy, vx)

        # call function to find acceleration
        ax1, ay1 = acceleration(vx, vy, theta)

        # Normal euler step
        vx_p = vx + h * ax1
        vy_p = vy + h * ay1
        theta_p = 0.0 if passed_minimum else math.atan2(vy_p, vx_p)
        ax2, ay2 = acceleration(vx_p, vy_p, theta_p)

        # take the average of the two slopes and update new value
        vx_new = vx + 0.5 * h * (ax1 + ax2)
        vy_new = vy + 0.5 * h * (ay1 + ay2)
        x_new = x + 0.5 * h * (vx + vx_p)
        y_new = y + 0.5 * h * (vy + vy_p)
        t_new = t + h

        # add results to list
        x_list.append(x_new); y_list.append(y_new)
        vx_list.append(vx_new); vy_list.append(vy_new)
        t_list.append(t_new)
        n += 1

        # check if resurfaced
        if y_new >= 0 and t_new > 0.05:
            break

    return np.array(t_list), np.array(x_list), np.array(y_list)

def rk4(x0, y0, vx0, vy0, t0, h):
    '''
    Fourth-order Runge Kutta 2D motion with drag, gravity, and buoyancy to model underwater
    behavior of the swimming. This model assumes no contribution from underwater kicks.
    Swimmer orientation is updated to horizontal after the lowest underwater point to
    better simulate real life swimming bahavior.
    '''
    # initialize values
    t_list = [t0]; x_list = [x0]; y_list = [y0]
    vx_list = [vx0]; vy_list = [vy0]
    n = 0
    passed_minimum = False

    # main loop
    while True:
        x, y, vx, vy, t = x_list[n], y_list[n], vx_list[n], vy_list[n], t_list[n]

        # check for minimum
        if not passed_minimum and vy > 0:
            passed_minimum = True
        theta = 0.0 if passed_minimum else math.atan2(vy, vx)

        # function for calculating acceleration for rk stages
        def f(vx_, vy_):
            ax_, ay_ = acceleration(vx_, vy_, theta)
            return np.array([ax_, ay_])

        # four slopes
        k1 = f(vx, vy)
        k2 = f(vx + 0.5*h*k1[0], vy + 0.5*h*k1[1])
        k3 = f(vx + 0.5*h*k2[0], vy + 0.5*h*k2[1])
        k4 = f(vx + h*k3[0], vy + h*k3[1])

        # weighted average for next velocity
        vx_new = vx + (h/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        vy_new = vy + (h/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

        # Update values
        x_new = x + h * vx
        y_new = y + h * vy
        t_new = t + h

        x_list.append(x_new); y_list.append(y_new)
        vx_list.append(vx_new); vy_list.append(vy_new)
        t_list.append(t_new)
        n += 1

        # check if resurfaced
        if y_new >= 0 and t_new > 0.05:
            break

    return np.array(t_list), np.array(x_list), np.array(y_list)

def underwater(flight_x, impact_velocity_x2, impact_velocity_y2):
    '''
    This function models the underwater phase of the swim start using the drag
    and buoyancy force equations and Newton's Second Law. The swimmer is assumed
    to glide underwater without additional propulsion other than the impact
    velocity. The differential equations are solved using three different numerical
    techniques: Euler's Method, Heun's Method, and 4th order Runge-Kutta (RK4).

    Inputs: flight_x (array of floats) as the x coordinates during in-air phase,
            impact_velocity_x2, impact_velocity_y2 (floats) as the components
            of the impact velocity at the water entry point

    Outputs: time4 (float) as the time taken during underwater phase, and
            uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r (arrays of floats) as
            the coordinates of the underwater phase calculated using Euler's
            Method, Heun's Method, and RK4, respectively
    '''

    # Create initial values for underwater phase
    x0 = flight_x[-1]
    y0 = -0.01 #swimmer starts slightly under water level
    vx0 = impact_velocity_x2
    vy0 = impact_velocity_y2
    t0 = 0
    # step size
    h = 0.01  # CHANGE THIS TO 0.1 to SEE the FIGURE 1a !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Run all three numerical methods, store values for plotting.
    # Variables: time, x, y
    t_e, uw_x_e, uw_y_e, _, _ = euler(x0, y0, vx0, vy0, t0, h)
    t_h, uw_x_h, uw_y_h = heun(x0, y0, vx0, vy0, t0, h)
    t_r, uw_x_r, uw_y_r = rk4(x0, y0, vx0, vy0, t0, h)

    # last element of time array is the final time. Same with the horizontal displacement.
    time4 = t_h[-1] # Heun's method results are chosen for optimal take-off calculations
    underwater_distance = uw_x_h[-1] - flight_x[-1]

    # output results.
    # print("Phase 3 (Underwater):")
    # print("  Time =", time4, "s")
    # print("  Horizontal distance =", underwater_distance, "m")

    return time4, uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r

'''
Phase 4: Surface Swimming =====================================================================
'''
def surface_swimming(uw_x):
    '''
    This function models the surface swimming phase of the swim start by
    assuming a constant freestyle stroke velocity. The distance travelled and
    time taken is calculated using the coordinates of the breakout point where
    the swimmer resurfaces from underwater.

    Inputs: uw_x (array of floats) as the x coordinates of the underwater phase.
            Can be from either Euler's, Heun's, or RK4 Method, but is chosen to
            be Heun's in this simulation as it was found to be most accurate.

    Outputs: time5 (float) as the time taken during surface swimming phase, and
            surf_x, surf_y (arrays of floats) as the coordinates of the surface
            swimming phase determined using the remaining distance.
    '''
    swim_velocity5 = 1.7 # m/s (average freestyle swimming speed)
    remaining_distance = 15 - uw_x[-1]
    time5 = remaining_distance / swim_velocity5
    surf_x = np.linspace(uw_x[-1], 15, 100)
    surf_y = np.zeros_like(surf_x)

    # print("Phase 4 (Surface Swimming):")
    # print("  Time =", time5, "s")

    return time5, surf_x, surf_y

'''
Total Swim Simulation ====================================================================
'''
def total_time(time1, time2, time4, time5):
    '''
    This function outputs the total time taken for the swim start by summing
    the time taken for each phase.

    Inputs: time1, time2, time4, time5 (floats) as the time taken for the
    takeoff, inair, underwater, and surface swimming phase, respectively.

    Output: time_total (float) as the total time taken during the swim start.
    '''
    # Total Time
    time_total = time1 + time2 + time4 + time5
    # print("Total time to 15m =", time_total, "seconds")
    return time_total

def swim_simulation(launch_angle):
    '''
    This function runs the full swim simulation for a takeoff angle input and
    returns the total time taken to travel 15m as well as all other variables
    needed to plot the velocity trajectory.

    Input: launch_angle (float) as the takeoff angle in degrees.

    Outputs: time_total (float) as the total time taken during the swim start,
            flight_x, flight_y (arrays of floats) as the coordinates of the
            in-air phase,
            uw_x_e, uw_y_e (arrays of floats) as the coordinates of the underwater
            phase calculated using Euler's Method,
            uw_x_h, uw_y_h (arrays of floats) as the coordinates of the underwater
            phase calculated using Heun's Method,
            uw_x_r, uw_y_r (arrays of floats) as the coordinates of the underwater
            phase calculated using RK4,
            surf_x, surf_y (arrays of floats) as the coordinates of the surface
            swimming phase.
    '''
    time_takeoff = take_off()
    time_inair, flight_x, flight_y, impact_velocity_x2, impact_velocity_y2 = in_air(launch_angle)
    time_underwater, uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r = underwater(flight_x, impact_velocity_x2, impact_velocity_y2)
    time_surface, surf_x, surf_y = surface_swimming(uw_x_h)
    time_total = total_time(time_takeoff, time_inair, time_underwater, time_surface)

    return time_total, flight_x, flight_y, uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r, surf_x, surf_y

'''
Full Trajectory Data and Plots
'''
def plot_trajectory(flight_x, flight_y, uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r, surf_x, surf_y):
    '''
    This function plots the velocity trajectory of the swimmer throughout all
    four phases of the swim start, given parameters corresponding to a given
    launch angle.

    Inputs: flight_x, flight_y (arrays of floats) as the coordinates of the
            in-air phase,
            uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r (arrays of floats)
            as the coordinates of the underwater phase calculated using Euler's
            Method, Heun's Method, and RK4, respectively.
            surf_x, surf_y (arrays of floats) as the coordinates of the surface
            swimming phase.

    Outputs: A plot of the full velocity trajectory of the swim start.
    '''
    plt.figure(figsize=(10,4))
    plt.plot(flight_x, flight_y, label='Flight in air')
    plt.plot(uw_x_e, uw_y_e, '--', label='Underwater Euler', alpha=0.7)
    plt.plot(uw_x_h, uw_y_h, label='Underwater Heun')
    plt.plot(uw_x_r, uw_y_r, label='Underwater RK4')
    plt.plot(surf_x, surf_y, label='Surface swimming')

    plt.axhline(0, color='black', linestyle='--', label='Water surface')
    plt.scatter(flight_x[-1], flight_y[-1], label='Water entry')
    plt.scatter(uw_x_r[-1], 0, label='Breakout')
    plt.scatter(15, 0,  label='15 m')

    plt.xlabel('Horizontal Position x (m)', fontsize = 15)
    plt.ylabel('Vertical Position y (m)', fontsize = 15)
    plt.tick_params(axis='both', labelsize = 14)
    # plt.title('Model of a Trajectory of a Full Swim Start Using Three Numerical Methods at a 15 Degree Takeoff Angle')
    plt.legend(fontsize = 14)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')

    plt.show()

def time_angle_graph():
    '''
    This function plots the total start times corresponding to launch angles
    and finds the minimum total start time within the interval of angles. It
    then returns the optimal angle corresponding to this time.

    Inputs: None

    Outputs: optimal_angle (float) as the launch angle in degrees that results
            in the minimum total start time, and a plot of total time vs launch
            angle.
    '''

    time = []
    angles = np.arange(-90, 90)
    for theta in angles:
        # print(theta)
        time.append((swim_simulation(theta))[0])

    # Prints each of the total start times corresponding to the respective launch angle
    # for i in range(len(angles)):
        # print(time[i])

    min_time = min(time)
    optimal_angle = angles[time.index(min_time)]
    print("\nOptimal launch angle:", optimal_angle, "degrees")
    print("Minimum total time to 15m:", min_time, "seconds")

    # Fit curve to a sine wave
    fit_sine(angles, time)

    # # Fit curve to a cubic polynomial
    # fit_cubic(angles, time)

    # Fit curve to a quartic polynomial
    # fit_quartic(angles, time)

    # Plot total time vs launch angle
    # plt.figure(figsize=(10,3))
    # plt.plot(angles, time, marker='o')
    # plt.xlabel("Launch Angle (degrees)", fontsize = 15)
    # plt.ylabel("Total Time to 15m (s)", fontsize = 15)
    # plt.tick_params(axis='both', labelsize = 14)
    # # plt.title("Swimmer Total Time vs Launch Angle")
    # plt.grid(True)
    # plt.show()

    return optimal_angle

# fitting Functions for curve fitting
def sine_fnc(x, a, b, c, d):
  '''
  This function describes the sine function with input parameters and will
  be used to fit the simulation data.

  Inputs: x (array of floats) as the independent variable (angles), and
          a, b, c, d (floats) as the parameters of the sine function.

  Outputs: The value of the sine function at x with the given parameters (time).
  '''
  return a * np.sin(b * x + c) + d

# Unused fit functions
# def quad_fnc(x, a, b, c):
#     return a * x**2 + b * x + c

# def cubic_fnc(x, a, b, c, d):
#     return a * x**3 + b * x**2 + c * x + d

# def quartic_fnc(x, a, b, c, d, e):
#     return a * x**4 + b * x**3 + c * x**2 + d * x + e

def fit_sine(angles, time):
    '''
    This function fits the simulation data to a sine curve with parameters
    found using scipy curve_fit. A graph is plotted comparing the two.

    Input: angles (array of floats) as the launch angles,
            time (array of floats) as the total start times corresponding to
            the launch angles.

    Output: A plot of the launch angles vs the resulting total start time,
            comparing the simulation data and the sine fit along with a
            residual plot.
    '''

    parameters, _ = curve_fit(sine_fnc, angles, time, p0 = [7, 0.05, 0, 15.5]) # Initial parameters guessed by inspection
    print("Sine Fit Parameters:", parameters)

    fitted_sine = sine_fnc(angles, *parameters)

    # reduced_chi2 = reduced_chi_squared(time, fitted_sine, len(parameters))
    # print("Reduced Chi-Squared:", reduced_chi2)

    # Plot total time vs launch angle
    plt.figure(figsize=(9,6))
    plt.plot(angles, time, marker='o', label='Simulation Data')
    plt.plot(angles, fitted_sine, label='Sine Fit', color='red')
    plt.xlabel("Launch Angle (degrees)", fontsize = 15)
    plt.ylabel("Total Time to 15m (s)", fontsize = 15)
    plt.tick_params(axis='both', labelsize = 14)
    plt.legend(fontsize = 14)
    # plt.title("Swimmer Total Time vs Launch Angle")
    plt.grid(True)
    plt.show()

    # Plot residuals
    # residual plot
    residuals = time - fitted_sine
    plt.errorbar(angles, residuals, yerr=None, fmt='o', capsize=3)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Launch Angle (degrees)", size = 14)
    plt.ylabel("Residuals (s)", size = 14)
    # plt.title("Figure 6: Residuals of Sine Fit", size = 15)
    plt.grid(True)
    plt.show()

'''
RUN CODE
Run a simulation with the optimal angle calculated from the time-angle graph.
Plot the velocity trajectory of the optimal takeoff angle
'''
optimal_angle = time_angle_graph()
total_time, flight_x, flight_y, uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r, surf_x, surf_y = swim_simulation(optimal_angle)
plot_trajectory(flight_x, flight_y, uw_x_e, uw_y_e, uw_x_h, uw_y_h, uw_x_r, uw_y_r, surf_x, surf_y)
