README.txt

The program should contain 1 main file (Gravfield.py) consisting of the methods required to run the simulation
and another 5 files (EulerTest.py, EulerCromTest.py, EulerRichTest.py,VerletTest.py,RungeKutta.py) consisting of the initial starting conditions and the code functions required
to run each of the algorithms.

Gravfield.py should be unaltered.

To run the simulation a test file should be created.
In this test file, all the bodies should be entered into the array using Gravfield('name','mass','position','velocity')
The bodies must be entered with the correct formatting (i.e name,mass,position,velocity)
Name must be a string, mass should be an int or a float, position and velocity must both be a list or tuple containing exactly 3 values.

To run the program run:

EulerTest.py for the Euler Method - by default with a timestep of 1000 seconds

EulerCromTest.py for the Euler-Cromer Method - by default with a timestep of 1000 seconds

EulerRichTest.py for the Euler-Richardson Method - by default with a timestep of 10000 seconds

VerletTest.py for the Verlet Method

RungeKuttaTest.py for the Runge-Kutta (RK4) Method.

These functions will run the simulation and will give the results.

By default each of these files will run the simulation for a total time of 315360000 seconds (1 year).
 This value can be changed easily