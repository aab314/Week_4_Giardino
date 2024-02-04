from matplotlib import pyplot as plt
from numpy.linalg import norm 
from numpy import arange, array, copy, pi, sqrt

class Planet_Orbit:

    def __init__(self, p1_i_vars, p2_i_vars, p1_M, p2_M, endtime, dt, p1_name, p2_name, color):  # let vars be the array of initial conditions 
        self.S_M = 1.9891e30   # mass of sun in kg
        self.dt = dt
        self.p1_name = p1_name
        self.p2_name = p2_name
        
        self.p1_i_vars = p1_i_vars
        self.p2_i_vars = p2_i_vars
        
        self.endtime = endtime
        self.color = color

        self.p1_ratio = 4*pi**2 * (p1_M/self.S_M)
        self.p2_ratio = 4*pi**2 * (p2_M/self.S_M)
        

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
        ax.plot(self.p1_xpoints,self.p1_ypoints, linewidth = .1, label = (f"{self.p1_name}"), color = self.color)
        
    def graphing_p2(self,ax):
        self.p1_xpoints, self.p1_ypoints, self.p2_xpoints, self.p2_ypoints = self.rk4()
        ax.plot(self.p2_xpoints,self.p2_ypoints, linewidth = .3, label = (f"{self.p2_name}"), color = 'black')
        

################################################ Setting Initial Conditions to pass into Class ###############################

def Simulate(p1,p2,t_end,time_step,ax, graph, color):

    S_m = 1.9891e30   # mass of sun in kg

    p1_rmax =p1[2]*(1+p1[1])
    p2_rmax =p2[2]*(1+p2[1])

    p1_vmin = sqrt(4 * pi**2) * sqrt(((1-p1[1])/(p1[2]*(1+p1[1]))) * (1 + (p1[0]/S_m)))  #Calculating minimum velocity at aphelion
    p2_vmin = sqrt(4 * pi**2) * sqrt(((1-p2[1])/(p2[2]*(1+p2[1]))) * (1 + (p2[0]/S_m)))  #Calculating minimum velocity at aphelion

    Simulation = Planet_Orbit([p1_rmax,0,0,p1_vmin],[p2_rmax,0,0,p2_vmin],p1[0],p2[0], t_end, time_step,p1[3],p2[3], color)

    if graph.lower() == "graph both":
        return Simulation.graphing_p1(ax), Simulation.graphing_p2(ax)
    if graph.lower() == "graph first":
        return Simulation.graphing_p1(ax)
    if graph.lower() == "graph second":
        return Simulation.graphing_p2(ax)



########################################## The final variables ############################

time_step = .001  # years
t_end = 5      # years

Jupiter = [1.9e27 * 100,  .048,   5.20,  "Jupiter with 100x Mass"]   # mass, eccentricity, then radius
Mercury = [2.4e23,  .206,   .39,  "Mercury"]
Venus =   [4.9e24,  .007,   .72,    "Venus"]
Earth =   [5.972e24, .017,  1.00,    "Earth"]
Mars =    [6.6e23,  .093,  1.52,     "Mars"]
Saturn =  [5.7e26 * 1000,  .056,  9.54,   "Saturn with 1000x Mass"]
Uranus =  [8.8e25,  .046, 19.19,   "Uranus"]
Neptune = [1.03e26, .010, 30.06,  "Neptune"]
Pluto =   [6.0e24,  .248, 39.53,    "Pluto"]


fig, ax = plt.subplots(dpi = 600)   # Create the axis

Simulate(Earth,Jupiter,t_end,time_step,ax,"Graph Both","green")

ax.scatter(0, 0, color='black', marker='.', label='Sun')
ax.legend()
ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")
ax.set_title(f"Orbit Around Sun ({t_end} Years)")
ax.axis('equal')

plt.show()


fig, ax = plt.subplots(dpi = 600)   # Create the axis
Simulate(Mars,Saturn,t_end*5,time_step*5,ax,"Graph Both","red")

ax.scatter(0, 0, color='black', marker='.', label='Sun')
ax.legend()
ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")
ax.set_title(f"Orbit Around Sun ({t_end*5} Years)")
ax.axis('equal')

plt.show()
