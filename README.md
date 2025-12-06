# OrbitLaunch

# IDEA -
Simulation for finding the trajectory to orbit that minimizes fuel consumption. Goal for the results - making a plot with the earth and a few starting positions that then shows the trajectory to orbit
with the trajectories color-coded according to fuel-consumption in that point in space - maybe add a sketch of this.

A few parameters will be tested once the model is made, things that can be varied that are also relevant to real-life application would be mass (payload size), cross-sectional area for atmospheric
resistance. Location will naturally be tested - can see how the earths rotation affects efficiency of launch location.

If we get this working we could start to consider how trajectories would look like for payload dropoffs. Launching a payload into orbit (for example a satellite) and for the booster rocket to reach
the ground again.


# TO DO 
A way to track fuel - we need to know the necessary energy to stay in orbit depending on orbit altitude. Then, based on this, we can just calculate the energy necessary to reach that point
from different places on the earth (this is done by adding a rotational velocity based on initial position). Then calculate the fuel requirements - actually, it depends on how quickly we want to reach orbit no? But still, if we increase altitude too slowly then energy needs are higher so there is some middle ground that needs to be calculated. Either way, fuel needs to account for reaching
a certain height and also having a tangential velocity component big enough to allow the satellite to orbit based on the orbit height.

So the final simulation will look simething like - findLeastFuel(init_pos, orbit_altitude, mass, csa) - nothing else I think, csa is cross sectional area (of the part facing resistance due to atmosphere). That means the simulation will find the optimal acceleration pattern to minimize energy needs to reach orbit altitude with energy remaining for a sufficiently big tangential velocity.



# IDEALISATION 
cirular orbits 
orbits must go accross the earth 
modelling the rocket as a dot (no ordientation of the rocket, no rotation of the rocket)
perfect control 
circular earth





# PARTS
athmosphere 
time varying mass 
thrust direction and magnitude 
earth rotation (momentum)
varying height of orbit 
varying launch site on the earth
model of the rocket (BFR SpaceX)
numerical integrator (verlet?, rk4?)
varying payoad mass




# Notes
Spherical coordinates 
3D


