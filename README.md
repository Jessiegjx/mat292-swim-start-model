# mat292-swim-start-model
Modeling the Swim Start

This Python script models the swim start of a freestyle swim event by modeling each stage--takeoff, in-air flight, underwater, and surface swimming. Upon running the script, it tests launch angles from -90 to 90 degrees to find the optimal launch angle corresponding to the minimal total start time. The first graph it outputs is a total start time vs angle graph, and the second graph is the residual plot comparing the simulation data to a fitted sine function. The optimal launch angle is then used to plot the velocity trajectory and a graph is outputted displaying the trajectory of the entire swim start.

To run this script (FinalProjectCode.py), numpy, matplotlib, scipy, and math are used. Therefore, a version no older than Python 3 must be installed, and numpy, matplotlib, and scipy must be installed prior to running the script.

Currently, the underwater phase outputs are calculated using a step size of h = 0.01 for the numerical techniques Euler's Method, Improved Euler's Method, and 4th order Runge-Kutta Method. This can be modified to output with a different step size by changing the h variable in the underwater() function.
