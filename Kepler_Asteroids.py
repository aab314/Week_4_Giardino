from matplotlib import pyplot as plt
from numpy.linalg import norm 
from numpy import arange, array, copy, pi, sqrt

class Planet_Orbit:

    def __init__(self, p1_i_vars, p2_i_vars, p1_M, p2_M, endtime, dt, p1_name,p2_name,color):  # let vars be the array of initial conditions 
        self.G = 6.67430e-11   # Gravitational Constant
        self.S_M = 1.9891e30   # mass of sun in kg
        self.dt = dt
        self.p1_i_vars = p1_i_vars
        self.p2_i_vars = p2_i_vars
        self.endtime = endtime
        self.color = color

        self.p1_ratio = 4*pi**2 * (p1_M/self.S_M)
        self.p2_ratio = 4*pi**2 * (p2_M/self.S_M)
        self.p1_name = p1_name
        self.p2_name = p2_name

    def derivs(self,p1_vars,p2_vars):
        self.p1_p = p1_vars[:2] 
        self.p1_v = p1_vars[2:]

        self.p1_xderiv = self.p1_v[0]
        self.p1_yderiv = self.p1_v[1]

        self.p1_R = norm(self.p1_p)

        self.p2_p = p2_vars[:2] 
        self.p2_v = p2_vars[2:]

        self.p2_xderiv = self.p2_v[0]
        self.p2_yderiv = self.p2_v[1]

        self.p2_R = norm(self.p2_p)

        self.r = sqrt(  (self.p1_p[0]-self.p2_p[0])**2 + (self.p1_p[1]-self.p2_p[1])**2  )
        
        
        self.p1_vxderiv = (-4 * pi**2 * self.p1_p[0])/(self.p1_R**3) - (self.p2_ratio * (self.p1_p[0]-self.p2_p[0]))/self.r**3         # using 4 * pi^2 in place of G * M_s for AU units 
        self.p1_vyderiv = (-4 * pi**2 * self.p1_p[1])/(self.p1_R**3) - (self.p2_ratio * (self.p1_p[1]-self.p2_p[1]))/self.r**3

        self.p2_vxderiv = (-4 * pi**2 * self.p2_p[0])/(self.p2_R**3) - (self.p1_ratio * (self.p2_p[0]-self.p1_p[0]))/self.r**3       # using 4 * pi^2 in place of G * M_s for AU units 
        self.p2_vyderiv = (-4 * pi**2 * self.p2_p[1])/(self.p2_R**3) - (self.p1_ratio * (self.p2_p[1]-self.p1_p[1]))/self.r**3
        
        return [self.p1_xderiv,self.p1_yderiv,self.p1_vxderiv,self.p1_vyderiv], [self.p2_xderiv,self.p2_yderiv,self.p2_vxderiv,self.p2_vyderiv]

    def rk4(self):
 
        self.tpoints = arange(0,self.endtime,self.dt)
        self.p1_allpoints = [self.p1_i_vars]
        self.p1_vars = copy(self.p1_i_vars)

        self.p2_allpoints = [self.p2_i_vars]
        self.p2_vars = copy(self.p2_i_vars)
        
        for t in self.tpoints:
            
            self.p1_k1,self.p2_k1 = self.dt * array(self.derivs(self.p1_vars,self.p2_vars))     # now for the method itself. Uses the time step and derivatives to calculate a new array 
            self.p1_k2,self.p2_k2 = self.dt * array(self.derivs(self.p1_vars + 1/2 * self.p1_k1, self.p2_vars + 1/2 * self.p2_k1))     # calculating k1, k2, k3, and k4 
            self.p1_k3,self.p2_k3 = self.dt * array(self.derivs(self.p1_vars + 1/2 * self.p1_k2, self.p2_vars + 1/2 * self.p2_k2))
            self.p1_k4,self.p2_k4 = self.dt * array(self.derivs(self.p1_vars + self.p1_k3, self.p2_vars + self.p2_k3))
        
            self.p1_vars += 1/6. * (self.p1_k1 + 2 * self.p1_k2 + 2 * self.p1_k3 + self.p1_k4)   # updating the vars array using a weghted average of k values
            self.p2_vars += 1/6. * (self.p2_k1 + 2 * self.p2_k2 + 2 * self.p2_k3 + self.p2_k4)   # updating the vars array using a weghted average of k values
            
            self.p1_allpoints.append(copy(self.p1_vars))  # Copy the current vars to store in allpoints
            self.p2_allpoints.append(copy(self.p2_vars))  # Copy the current vars to store in allpoints
            
        self.p1_allpoints_array = array(self.p1_allpoints)  # Convert allpoints to a 2D NumPy array. Rows are time steps, columns are x,y,vx,vy
        self.p2_allpoints_array = array(self.p2_allpoints)  # Convert allpoints to a 2D NumPy array. Rows are time steps, columns are x,y,vx,vy
        return self.p1_allpoints_array[:, 0], self.p1_allpoints_array[:, 1], self.p2_allpoints_array[:, 0], self.p2_allpoints_array[:, 1]     # returns all time steps for x, then y

    def graphing_p1(self,ax):
        self.p1_xpoints, self.p1_ypoints, self.p2_xpoints, self.p2_ypoints = self.rk4()
        ax.plot(self.p1_xpoints,self.p1_ypoints, linewidth = .05, label = (f"{self.p1_name}"), color = self.color)
        
        
    def graphing_p2(self,ax):
        self.p1_xpoints, self.p1_ypoints, self.p2_xpoints, self.p2_ypoints = self.rk4()
        ax.plot(self.p2_xpoints,self.p2_ypoints, linewidth = .2, label = (f"{self.p2_name}"), color = 'black')
        

################################################ Setting Initial Conditions to pass into Class ###############################

def Simulate(gap,t_end,time_step,ax, graph,color):

    p2 = [5.2, 2.755, 1.9e27, "Jupiter"]
    S_m = 1.9891e30
    G = 6.67430e-11

    jupiter_period = 5.2*2*pi/2.755
    asteroid_period = jupiter_period/gap[0]
    asteroid_radius = (asteroid_period)**(2/3)
    asteroid_velocity = (2*pi*asteroid_radius)/asteroid_period 

    p1 = [asteroid_radius, asteroid_velocity, gap[1], gap[2]]

    Simulation = Planet_Orbit([p1[0],0,0,p1[1]],[p2[0],0,0,p2[1]],p1[2],p2[2], t_end, time_step,p1[3],p2[3],color)

    if graph.lower() == "graph both":
        return Simulation.graphing_p2(ax), Simulation.graphing_p1(ax)
    if graph.lower() == "graph first":
        return Simulation.graphing_p1(ax)
    if graph.lower() == "graph second":
        return Simulation.graphing_p2(ax)

########################################## The final variables ############################

time_step = .005  # years
t_end = 200        # years

Asteroid_1 = [2.0/1, 10e10, "Asteroid 1 (2/1)"]
Asteroid_2 = [2.3/1, 10e10, "Asteroid 2 (2.3/1)"]
Asteroid_3 = [1.7/1, 10e10, "Asteroid 3 (1.7/1)"]

Asteroid_4 = [7/3,   10e10, "Asteroid 4 (7/3)"]
Asteroid_5 = [7.5/3, 10e10, "Asteroid 5 (7.5/3)"]
Asteroid_6 = [6.5/3, 10e10, "Asteroid 6 (6.5/3)"]

fig, ax = plt.subplots(dpi = 1000)   # Create the axis
Simulate(Asteroid_1,t_end,time_step,ax,"Graph Both", "green")
Simulate(Asteroid_2,t_end,time_step,ax,"Graph First", "red")
Simulate(Asteroid_3,t_end,time_step,ax,"Graph First", "blue")

ax.scatter(0, 0, color='black', marker='.', label='Sun')
ax.legend(loc='upper right')
ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")
ax.set_title(f"Orbit Around Sun ({t_end} Years)")
ax.axis('equal')

plt.show()
