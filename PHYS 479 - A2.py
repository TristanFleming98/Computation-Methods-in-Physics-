''' This document contains my code for the PHYS 479 Leapfrog techniques, Classical
Harmonic Oscillators and 3-body simulations assignment which is divided into two parts. 
The first part (Question 1) involves simulating a classical harmonic oscillator using 
a Leapfrog technique. The second part (Question 2) uses the same technique and an RK4 
to simulate 3 body planetary motion. The results from the Leapfrog and RK4 are compared 
with results from odeint (a Python ODE solver). The results for both are shown using 
simple animation. The planetary motion is done several times using different initial 
conditions. This code was coded using the Spyder IDE from anaconda and Python 3.7. It 
can run all at once using any Python IDE or by the cell in Spyder.

   Author: Tristan Fleming
   Date: 2021-02-11
   Student number 20076612
   Version:  6 '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import optimize 
#%% 1a) Solve for equations (2) and (4) using the Leapfrog Technique
# Functions: 
def leapfrog(diffeq, r0, v0, t, h):
    '''Vecotirzed leapfrog method (specifically the Verlet method) using numpy 
    arrays. This function is a class of ODE solver. It updates position and velocity 
    at interleaved time points that are staggered and returns the position and 
    velocity for a full timestep.'''
    
    hh = h / 2.0 # The interleaved time step
    r1 = r0 + hh * diffeq(0, r0, v0, t) # Position at hh 
    v1 = v0 + h * diffeq(1, r1, v0, t + hh) # Velocity at h using r1
    r1 = r1 + hh * diffeq(0, r0, v1, t + h) # New position at h
    
    return r1, v1    

def SHO(id, x, v, t):
    '''This function represents the 1st-order ODEs for a Simple Harmonic Oscillator
    with m = k = w_0 = 1. It uses an ID to correspond to either ODE. An ID of zero 
    returns dx/dt and dv/dt is returned otherwise. '''
    
    if id == 0:
        return v # dx/dt solution 
    else:
        return -x # dv/dt solution
    

def time(tMax, dt):
    '''This function creates and returns a time numpy array for a specified length 
    and time.'''
    
    t = np.arange(0, tMax, dt)
    return t

def createArray(npts):
    '''This function creates a n by 2 array for question 1. The arrays will be 
    passed in the RK4 and leafrog functions. The initial conditions for the arrays
    in question 1 is r = 1, v = 0. The function returns the n by 2 array and the initial
    array'''
    
    y = np.zeros((npts,2))
    yInit = np.array([1.0, 0.0]) # Set initial condition
    y1 = yInit
    y[0,:] = y1
    
    return y, y1

def leapfrogIter(diffeq, y, t, h):
    '''This function passes an array the same size as the time array and returns 
    a vector containing all positions and velocities from the differential 
    equation. It uses the leapfrog function to evaluate the differential equations.
    The function returns an array of all the positions and velocities and a list of
    the energies for every timestep.'''
    
    y1 = y[0,:] # Get inital input 
    energy = [] # Create list of energies 
    
    for i in range(1, npts):
        y1 = leapfrog(SHO, y1[0], y1[1], t[i-1], dt) # Compute next value 
        y[i,:] = y1
        E = (0.5 * y1[0] ** 2) + (0.5 * y1[1] ** 2)
        energy.append(E)
        
    return y, energy

def relativeError(actual, expected): 
    '''This function computes and returns a list of the relative error for each 
    point given actual and expected results of the same size.'''
    
    npts = len(actual) # Determine number of iterations 
    relativeError = []
    
    for i in range(npts):
        RE = np.abs(actual[i] - expected[i]) / expected[i]  # Compute Relative Error
        relativeError.append(RE)
        
    return relativeError # Return list of relative errors 

def absoluteError(actual, expected):
    '''This function computes and returns a list of the absolute error for each 
    point given actual and expected results of the same size.'''
    
    npts = len(actual)
    absoluteError = []
    
    for i in range(npts):
        AE = np.abs(actual[i] - expected[i]) 
        absoluteError.append(AE)
    
    return absoluteError 

def niceFigure(useLatex=True):
    '''This function makes the visual plots look a lot better. If you don't have 
    Latex on your device just set useLatex = False.'''
    
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 20})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return 

niceFigure(useLatex = False) # Make plots look better


# Parameters: 
w0 = 1
T0 = 2 * np.pi
tMax = 50 * T0
dt = 0.01 * T0
t = time(tMax, dt)
npts = len(t)
y, y1 = createArray(npts)


# Solutions:
# Leapfrog:
y, totalEnergy = leapfrogIter(SHO, y, t, dt)
r = y[:,0] # Create an array of positions
v = y[:,1] # Create an array of velocities 

# Analytical:
rExact = [np.cos(w0 * t[i]) for i in range(len(t))]
vExact = [-w0 * np.sin(w0 * t[i]) for i in range(len(t))]
eExact = [(0.5 * rExact[i] ** 2) + (0.5 * vExact[i] ** 2) for i in range(len(t))]

#Compute errors:
# Position:
AEr = absoluteError(r, rExact)
REr = relativeError(r, rExact)
# Velocity:
AEv = absoluteError(v, vExact)
REv = relativeError(v, vExact)
# Energy:
AEe = absoluteError(totalEnergy, eExact)
REe = relativeError(totalEnergy, eExact)


# Plotting 
fig, ax1 = plt.subplots()
ax1.semilogy(t[:-1] / (2 * np.pi), AEe)
ax1.set(ylabel = "Absolute Error $(e)$", xlabel = "$T_0$")
plt.savefig('Fig 0',  bbox_inches='tight', dpi = 300)

fig, ax3 = plt.subplots()
ax3.semilogy(t[:-1] / (2 * np.pi), REe)
ax3.set(ylabel = "Relative Error $(r)$", xlabel = "$T_0$")
plt.savefig('Fig 0b',  bbox_inches='tight', dpi = 300)

fig, ax1 = plt.subplots()
ax1.semilogy(t, AEv)
ax1.set(ylabel = "Absolute Error $(v)$", xlabel = "$T_0$")
plt.savefig('Fig 1',  bbox_inches='tight', dpi = 300)

fig, ax2 = plt.subplots()
ax2.semilogy(t, AEr)
ax2.set(ylabel = "Absolute Error $(r)$", xlabel = "$T_0$")
plt.savefig('Fig 2',  bbox_inches='tight', dpi = 300)

fig, ax3 = plt.subplots()
ax3.semilogy(t, REr)
ax3.set(ylabel = "Relative Error $(r)$", xlabel = "$T_0$")
plt.savefig('Fig 3',  bbox_inches='tight', dpi = 300)

fig, ax4 = plt.subplots()
ax4.semilogy(t, REv)
ax4.set(ylabel = "Relative Error $(v)$", xlabel = "$T_0$")
plt.savefig('Fig 4',  bbox_inches='tight', dpi = 300)
#%% 1b) Plot graphs in phase space for different time steps
# Functions: 
def RK4(f, y,t,h):
    '''This function is a vectorized rk4 ODE solver. It uses temporal discretization 
    to approximate a solution of the imputed ordinary differential equation. It 
    takes a ODE, vector, time array and timestep and returns an array of approximate
    solutions for each time'''
 
    k1 = h * np.asarray(f(y,t))
    k2 = h * np.asarray(f(y+(k1/2), t+(h/2)))
    k3 = h * np.asarray(f(y+(k2/2), t+(h/2)))
    k4 = h * np.asarray(f(y+k3, t+h))
    y = y + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return y

def derivs(y, t):
    '''This function represents the 1st-order coupled ODEs of a Simple harmonic
    Oscillator. It's used for the ODEs in my vectoried RK4'''
    
    dy = np.zeros(len(y))
    
    dy[0] = y[1]
    dy[1] = -y[0]
    
    return dy

def RK4Iter(ODEsolver, f, y, y1, t, h):
    '''This function passes an array the same size as the time array into the RK4
    and returns a vector containing all positions and velocities from the differential 
    equation. It iterates every time value through my RK4 to determine all solutions.
    It also computes the energy at each time step and retuns a list of energies.'''
    
    npts =  len(t)
    energy = []
    
    for i in range(1, npts):
        y1 = RK4(f, y1, t, dt)
        y[i,:] = y1
        E = (0.5 * y1[0] ** 2) + (0.5 * y1[1] ** 2)
        energy.append(E)
    return y, energy

def plotting(y, yr, figNum):
    '''This funciont takes the array of solutions for both the RK4 and leapfrog 
    and plots them and also plots their energy. It then saves each plot'''
    
    fig, ax1 = plt.subplots()
    ax1.plot(y[:,0],y[:,1])
    ax1.set(ylabel = "v (m/s)", xlabel = "x (m)")
    plt.savefig('Fig' + str(figNum),  bbox_inches='tight', dpi = 300)
    figNum += 1
    
    fig, ax2 = plt.subplots()
    ax2.plot(t[:-1], energyLeap)
    ax2.set(ylabel = "E (J)", xlabel = "t (s)")
    ax2.set_ylim(0,0.8)
    plt.savefig('Fig' + str(figNum),  bbox_inches='tight', dpi = 300)
    figNum += 1
    
    fig, ax1 = plt.subplots()
    ax1.plot(yr[:,0],yr[:,1])
    ax1.set(ylabel = "v (m/s)", xlabel = "x (m)")
    plt.savefig('Fig' + str(figNum),  bbox_inches='tight', dpi = 300)
    figNum += 1
    
    fig, ax2 = plt.subplots()
    ax2.plot(t[:-1], energyRK4)
    ax2.set(ylabel = "E (J)", xlabel = "t (s)")
    ax2.set_ylim(0,0.8)
    plt.savefig('Fig' + str(figNum),  bbox_inches='tight', dpi = 300)
    figNum += 1
    return 


# Parameters (for h = 0.02 *T_0):
tMax = 800 * T0 # Time for 800 cycles 
dt = 0.02 * T0
t = time(tMax, dt)
npts = len(t)
y, y1 = createArray(npts)
yr, yr1 = createArray(npts)



# Solutions:
# RK4:
yr, energyRK4 = RK4Iter(RK4,derivs, yr, yr1, t, dt)
# leapfrog:
y , energyLeap = leapfrogIter(SHO, y, t, dt)


# Plotting:
plotting(y, yr, 5)
#%% Parameters (for h = 0.04 * T0):
dt = 0.04 * T0
t = time(tMax, dt)
npts = len(t)
y, y1 = createArray(npts)
yr, yr1 = createArray(npts)


# Solutions:
# RK4
yr, energyRK4 = RK4Iter(RK4,derivs, yr, yr1, t, dt)
# leapfrog:
y, energyLeap = leapfrogIter(SHO, y, t, dt)


# Plotting:
plotting(y, yr, 9)
#%% Parameters (for h = 0.1 * T0):
dt = 0.1 * T0
t = time(tMax, dt)
npts = len(t)
y, y1 = createArray(npts)
yr, yr1 = createArray(npts)


# Solutions:
# RK4
yr, energyRK4 = RK4Iter(RK4,derivs, yr, yr1, t, dt)
# leapfrog:
y, energyLeap = leapfrogIter(SHO, y, t, dt)


# Plotting:
plotting(y, yr, 13)
#%% 2) Find the initial conditions 
# Functions:
def quinticEquation(l):
    '''This function is Eulers quintic equation of collinear motion used to pass
    in the optimize fsolve root solver to determine the value of lambda.'''
    
    return l**5 * (m2 + m3) + l**4 * (2 * m2 + 3 * m3) + l**3 * (m2 + 3 * m3) +\
    l**2 * (-3 * m1 - m2) + l * (-3 * m1 - 2 * m2) + (-m1 - m2)

def getInitial(w, m1, m2, m3, numPeriods):
    '''This function takes the omega, the mass of each planet and the number of 
    periods. It computes the initial conditions and returns them. It also computes
    a time array for a specified number of periods.'''
    
    
    # Need to solve lambda from Eulers quintic equation of collinear motion     
    lamb = optimize.fsolve(quinticEquation, x0 = 1) # Solve for lambda using Newtons method
    l = float(lamb[0]) # Give lambda a numeric value 
 
    a = ((1 / w**2) * (m2 + m3 - m1 * (1 + 2*l) / ((l**2) * (1 + l)**2)))**(1/3)

    # Initial conditions:
    x2 = 1 / (w**2 * a**2) * (m1/l**2 - m3) # Position
    x1 = x2 - l * a
    x3 = - ( m1 * x1 + m2 * x2) / m3 
    
    w0 = w
    v1y = w * x1 # Velocity 
    v2y = w * x2
    v3y = w * x3 
    
    T0 = 2 * np.pi / w0 # Create time array 
    dt = 0.001  # Use more accurate timestep 
    tMax = numPeriods * T0
    t = time(tMax,dt)
    
    return x1, x2, x3, v1y, v2y, v3y, w0, t, dt

# Parameters:
Delta = 1 * 10 ** -9
m1 = 1.0
m2 = 2.0
m3 = 3.0 
w = 1.0

x1, x2, x3, v1y, v2y, v3y, w0, t , dt= getInitial(w, m1, m2, m3, 4)
#%% 2a) 
# Functions: 
def distance(x1, x2, x3, y1, y2, y3):
    ''' This function computes the distance beteween two objects using their 
    x and y components'''    
    # The distance between the components
    r12 = [x1 - x2, y1 - y2]; r13 = [x1 - x3, y1 - y3]; r23 = [x2 - x3, y2 - y3]
    
    r12v = np.asarray(r12); r13v = np.asarray(r13); r23v = np.asarray(r23)
    # The distance between two masses 
    s12 = np.sqrt(r12v.dot(r12v))
    s13 = np.sqrt(r13v.dot(r13v))
    s23 = np.sqrt(r23v.dot(r23v))
    
    return r12, r13, r23, s12, s13, s23

def derivs(y, t):
    '''This function is my set of ODEs for Eulers collinear motion. I pass this 
    function through my RK4.'''
    
    dy = np.zeros(len(y))
    # The distance between the components
    r12, r13, r23, s12, s13, s23 = distance(y[0], y[4], y[8], y[1], y[5], y[9])
    # Positions:
    dy[0] = y[2]; dy[4] = y[6]; dy[8] = y[10] # x1, x2, x3
    dy[1] = y[3]; dy[5] = y[7]; dy[9] = y[11] # y1, y2, y3
    # Velocities:
    dy[2] = - m2 * r12[0] / s12**3 - m3 * r13[0] / s13**3 # v1x
    dy[3] = - m2 * r12[1] / s12**3 - m3 * r13[1] / s13**3 # v1y
    dy[6] = m1 * r12[0] / s12**3 - m3 * r23[0] / s23**3 # v2x
    dy[7] = m1 * r12[1] / s12**3 - m3 * r23[1] / s23**3 # v2y
    dy[10] = m1 * r13[0] / s13**3 + m2 * r23[0] / s23**3 # v3x
    dy[11] = m1 * r13[1] / s13**3 + m2 * r23[1] / s23**3 # v3y
    
    return dy

def EOM(id, r, v, t):
    '''This function takes a position and velocity array and passes them through
    my leapfrog. It takes an ID to determine what set of ODEs to pass through 
    the leapfrog and returns the updated values.''' 
    
    if id == 0:
        dx1 = v[0]; dx2 = v[1]; dx3 = v[2]
        dy1 = v[3]; dy2 = v[4]; dy3 = v[5]
        return np.array([dx1, dx2, dx3, dy1, dy2, dy3])
    
    else: 
        r12, r13, r23, s12, s13, s23 = distance(r[0], r[1], r[2], r[3], r[4], r[5])
               
        dv1x =  - m2 * r12[0] / s12**3 - m3 * r13[0] / s13**3 # v1x, v1y
        dv2x = m1 * r12[0] / s12**3 - m3 * r23[0] / s23**3 # v2x
        dv3x = m1 * r13[0] / s13**3 + m2 * r23[0] / s23**3 # v3x
        
        dv1y = - m2 * r12[1] / s12**3 - m3 * r13[1] / s13**3 # v1y
        dv2y = m1 * r12[1] / s12**3 - m3 * r23[1] / s23**3 # v2y
        dv3y = m1 * r13[1] / s13**3 + m2 * r23[1] / s23**3 # v3y
        
        return np.array([dv1x, dv2x, dv3x, dv1y, dv2y, dv3y])
        
    
def createArray(npts):
    '''This function creates a n by 12 array for question 2. They arrays will be 
    passed in the RK4 and leafrog functions'''
    
    y = np.zeros((npts,12))
    # [x1, y1, v1x, v1y, x2, y2, v2x, v2y, x3, y3, v3x, v3y]
    yInit = np.array([x1, 0.0, 0.0, v1y, x2, 0.0, 0.0, v2y, x3, 0.0, 0.0, v3y])
    y1 = yInit
    y[0,:] = y1
    
    return y, y1

def RK4Iter(ODEsolver, f, y, y1, t, h):
    '''This function passes an array the same size as the time array into the RK4
    and returns a vector containing all positions and velocities from the differential 
    equation. '''
    
    npts =  len(t)
    
    for i in range(1, npts):
        y1 = RK4(f, y1, t, dt)
        y[i,:] = y1

    return y

def leapfrogIter(diffeq, y, t, h):
    '''This function passes an array the same size as the time array and returns 
    a vector containing all positions and velocities from the differential 
    equation. It passes through the leapfrog and return a position array and a 
    velocity array.'''
    
    r = np.zeros((npts, 6)); v = np.zeros((npts, 6)) # Create two arrays
    rInitial = np.array([y[0,0], y[0,4], y[0,8], y[0,1], y[0,5], y[0,9]])
    vInitial = np.array([y[0,2], y[0,6], y[0,10], y[0,3], y[0,7], y[0,11]])
    r[0,:] = rInitial; v[0,:] = vInitial # Set first element to inital conditions
    
    for i in range(0, npts-1):
        r1, v1 = leapfrog(EOM, r[i,:], v[i,:], t[i], dt)
        r[i+1,:] = r1
        v[i+1,:] = v1
    return r, v

def Energy(x1, x2, x3, y1, y2, y3, v1x, v2x, v3x, v1y, v2y, v3y):
    '''This function computes and returns the energy of the system for every point 
    using the all the components of the system at that point'''
     
    r12, r13, r23, s12, s13, s23 = distance(x1, x2, x3, y1, y2, y3)
    # Compute the magnitude of the velocities    
    v1 = np.sqrt(v1x ** 2 + v1y ** 2)
    v2 = np.sqrt(v2x ** 2 + v2y ** 2)
    v3 = np.sqrt(v3x ** 2 + v3y ** 2)
    # Compute the energy 
    Energy = 0.5 * (m1 * (v1**2) + m2 * (v2**2) + m3 * (v3**2))  - m1 * m2 / s12 - m1 * m3 / s13 - m2 * m3 / s23
    
    return Energy

def animation(x1, x2, x3, y1, y2, y3, t, figNum, blim = 3 ):
    '''This function creates an animation of the three bodies position. Each new
    point also contains a velocity vector to indicate the direction and magnitude
    of the planets velocity. It also takes snapshots of the animation at different 
    intervals and saves them.''' 
    
    frame = 50 # Number of frames 
    fig = plt.figure(figsize = (10,5))
    fig, ax1 = plt.subplots()
    
    ax1.set_xlim(-blim,blim); ax1.set_ylim(-blim,blim)
    ax1.set(xlabel = "x (arb. units)", ylabel = "y (arb. units)")
    # Starting positions
    ax1.scatter([0, x1[0], x2[0], x3[0]],[0, y1[0], y2[0], y3[0]], s = 150, c = 'black', marker = 'x')
    
    
    x1Vals = []; x2Vals = []; x3Vals = []
    y1Vals = []; y2Vals = []; y3Vals = []
        
    line1, = ax1.plot(x1Vals, y1Vals, 'bo')
    line2, = ax1.plot(x2Vals, y2Vals, 'ro')
    line3, = ax1.plot(x3Vals, y3Vals, 'go')
    # Q = quiver(x, y, u, v, 'filled', 'Marker', 'o', 'LineWidth', 1.8, 'AutoScaleFactor', .1);

    plt.pause(2) # Wait 2 seconds to start animation 
    stopTime1 = len(y) // 4
    stopTime2 = len(y) // 2
    stopTime3 = len(y) * 3 // 4 
    stopTime4 = len(y) - 1
    
    for i in range(len(y)): # To plot every specified frame
        if i % frame == 0:
            
            x1Vals.append(x1[i]); x2Vals.append(x2[i]); x3Vals.append(x3[i])
            y1Vals.append(y1[i]); y2Vals.append(y2[i]); y3Vals.append(y3[i])
           
            line1.set_xdata(x1Vals); line2.set_xdata(x2Vals); line3.set_xdata(x3Vals)  
            line1.set_ydata(y1Vals); line2.set_ydata(y2Vals); line3.set_ydata(y3Vals)
            #V1 = ax1.quiver(x1[i], y1[i], v1x[i], v1y[i], color = 'r')
            #V1.set_UVC(v1x[i], v1y[i])
            #V1.set_XYC(x1[i], y1[i])
            
            plt.draw()
            plt.pause(.001)
            
        if i == stopTime1 or i == stopTime2 or i == stopTime3 or i == stopTime4:
            # Save figures for different periods
            plt.savefig('Fig ' + str(figNum),  bbox_inches='tight', dpi = 300)
            figNum += 1
            
     
def EnergyPlot(x1, x2, x3, y1, y2, y3, v1x, v2x, v3x, v1y, v2y, v3y, t, figNum):
    '''This function plots the energy of the system as a function of time. It iterates 
    every 10th points and computes the energy of the system. It displays the energy
    in a plot and saves the figure.'''
    
    frame = 10 # Only plot every 10th point 
    
    fig = plt.figure(figsize = (10,5))
    fig, ax2 = plt.subplots()
    ax2.set(xlabel = "Energy (arb. units)", ylabel = "Time ($T_0$)")
    energy = []
    time = []  
    
    for i in range(len(y)):     
        if i % frame == 0:
            E = Energy(x1[i], x2[i], x3[i], y1[i], y2[i], y3[i], v1x[i], v2x[i], v3x[i] , v1y[i], v2y[i], v3y[i])
            
            time.append(t[i] / (2 * np.pi)); energy.append(E)    
        
    ax2.plot(time, energy)
    plt.savefig('Fig ' + str(figNum),  bbox_inches='tight', dpi = 300)
     

        
                
                
# Parameters: 
npts = len(t)
y, y1 = createArray(npts)


# Solution:
# RK4:
y = RK4Iter(RK4, derivs, y, y1, t, dt)


# Plotting: 
animation(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], t, 17)
EnergyPlot(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], y[:,2], y[:,6], y[:,10], y[:,3], y[:,7], y[:,11], t, 21)
#%% Leapfrog (w0 = 1)
# Parameters:
yl, yl1 = createArray(npts)  
  

#Solutions:  
# Leapfrog:    
rl, vl = leapfrogIter(EOM, yl, t, dt)


# Plotting:
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 22)
EnergyPlot(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], vl[:,0], vl[:,1], vl[:,2], vl[:,3], vl[:,4], vl[:,5], t, 26)
#%% RK4 (w0 = 1 + 10 ** 9)
# Parameters:
Delta = 1 * 10 ** -9
x1, x2, x3, v1y, v2y, v3y, w0, t , dt = getInitial(w + Delta, m1, m2, m3, 4)
npts = len(t)
y, y1 = createArray(npts)

# Solution:
#RK4:
y = RK4Iter(RK4, derivs, y, y1, t, dt)


# Plotting:
animation(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], t, 27)
EnergyPlot(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], y[:,2], y[:,6], y[:,10], y[:,3], y[:,7], y[:,11], t, 31)
#%% Leapfrog (w0 = 1 + 10 ** 9)
# Parameters:
yl, yl1 = createArray(npts)  
  

#Solutions:  
# Leapfrog:    
rl, vl = leapfrogIter(EOM, yl, t, dt)


# Plotting:
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 32)
EnergyPlot(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], vl[:,0], vl[:,1], vl[:,2], vl[:,3], vl[:,4], vl[:,5], t, 36)
#%% RK4 (w0 = 1 - 10 ** 9)
# Parameters
x1, x2, x3, v1y, v2y, v3y, w0, t , dt= getInitial(w - Delta, m1, m2, m3,4)
npts = len(t)
y, y1 = createArray(npts)


# Solution:
#RK4:
y = RK4Iter(RK4, derivs, y, y1, t, dt)


# Plotting:
animation(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], t, 37)
EnergyPlot(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], y[:,2], y[:,6], y[:,10], y[:,3], y[:,7], y[:,11], t, 41)
#%% Leapfrog (w0 = 1 - 10 ** 9)
# Parameters:
yl, yl1 = createArray(npts)  
  

#Solutions:  
# Leapfrog:    
rl, vl = leapfrogIter(EOM, yl, t, dt)


# Plotting:
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 43)
EnergyPlot(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], vl[:,0], vl[:,1], vl[:,2], vl[:,3], vl[:,4], vl[:,5], t, 46)
#%%  2b) plot the results for w0 + 10 ** -9 in the odeint function  
# Function:
def model(y, t):
    '''This function expresses the equations of motion to be passed through the 
    ODE int function.'''
    
    r12, r13, r23, s12, s13, s23 = distance(y[0], y[4], y[8], y[1], y[5], y[9])
    
    dx1 = y[2] # x1
    dy1 = y[3] # y1
    dv1x = - m2 * r12[0] / s12**3 - m3 * r13[0] / s13**3 # v1x
    dv1y = - m2 * r12[1] / s12**3 - m3 * r13[1] / s13**3 # v1y
    dx2 = y[6] # x2
    dy2 = y[7] # y2
    dv2x = m1 * r12[0] / s12**3 - m3 * r23[0] / s23**3 # v2x
    dv2y = m1 * r12[1] / s12**3 - m3 * r23[1] / s23**3 # v2y
    dx3 = y[10] # x3
    dy3 = y[11] # y3
    dv3x = m1 * r13[0] / s13**3 + m2 * r23[0] / s23**3 # v3x
    dv3y = m1 * r13[1] / s13**3 + m2 * r23[1] / s23**3 # v3y
    
    return dx1, dy1, dv1x, dv1y, dx2, dy2, dv2x, dv2y, dx3, dy3, dv3x, dv3y


# Parameters:
x1, x2, x3, v1y, v2y, v3y, w0, t, dt = getInitial(w + Delta, m1, m2, m3, 4)
error = 10 ** -12 # Set error tolerance
y1 = [x1, 0.0, 0.0, v1y, x2, 0.0, 0.0, v2y, x3, 0.0, 0.0, v3y]


# Solution:
y = odeint(model, y1, t, rtol = error, atol = error)


# Plotting:
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 47)
EnergyPlot(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], vl[:,0], vl[:,1], vl[:,2], vl[:,3], vl[:,4], vl[:,5], t, 51)
#%% 2c) 
# Functions: 
def leapfrogIterReverse(diffeq, y, t, h):
    '''This function passes an array the same size as the time array and returns 
    a vector containing all positions and velocities from the differential 
    equation. Once it iterates through half the points and reverses the direction 
    for one iteration and continues'''
    
    r = np.zeros((npts, 6))
    v = np.zeros((npts, 6))
    rInitial = np.array([y[0,0], y[0,4], y[0,8], y[0,1], y[0,5], y[0,9]])
    vInitial = np.array([y[0,2], y[0,6], y[0,10], y[0,3], y[0,7], y[0,11]])
    r[0,:] = rInitial
    v[0,:] = vInitial
    
    for i in range(0, npts-1):
        
        if i == npts // 2: # Halfway through
            r1, v1 = leapfrog(EOM, r[i,:], v[i,:], t[i], dt)
            r[i+1,:] = r1
            v[i+1,:] = -v1 # Set velocity to -v
        
        else:
            r1, v1 = leapfrog(EOM, r[i,:], v[i,:], t[i], dt)
            r[i+1,:] = r1
            v[i+1,:] = v1
    return r, v

def RK4IterReverse(ODEsolver, f, y, y1, t, h):
    '''This function passes an array the same size as the time array into the RK4
    and returns a vector containing all positions and velocities from the differential 
    equation. '''
    
    npts =  len(t)
    
    for i in range(1, npts):
        if i == npts // 2:
            y1 = RK4(f, y1, t, dt)
            y1 = [y1[0], y1[1], -y1[2], -y1[3], y1[4], y1[5], -y1[6], -y1[7], y1[8], y1[9], -y1[10], -y1[11]]
            y[i,:] = y1
        else:
            y1 = RK4(f, y1, t, dt)
            y[i,:] = y1

    return y
       
# Parameters:
x1, x2, x3, v1y, v2y, v3y, w0, t, dt = getInitial(w, m1, m2, m3, 6)   
npts = len(t)   
yl, yl1 = createArray(npts) 


# Solution:
rl, vl = leapfrogIterReverse(EOM, yl, t, dt)


# Plotting: 
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 52)
EnergyPlot(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], vl[:,0], vl[:,1], vl[:,2], vl[:,3], vl[:,4], vl[:,5], t, 56)
#%% RK4 Reverse
# Parameters:
y, y1 = createArray(npts)


# Solution:
#RK4:
y = RK4IterReverse(RK4, derivs, y, y1, t, dt)


# Plotting:
animation(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], t, 57)
EnergyPlot(y[:,0], y[:,4], y[:,8], y[:,1], y[:,5], y[:,9], y[:,2], y[:,6], y[:,10], y[:,3], y[:,7], y[:,11], t, 61)
#%% 2d)i) Use leapfrog for different initial conditions 
#Parameters:
m1 = (1/3)
m2 = (1/3)
m3 = (1/3)
w0 = 3.3
T0 = 2 * np.pi / w0
tMax = 3 * T0
dt = 0.001
t = time(tMax, dt)
npts = len(t)

# Set new initial conditions 
x1 = -0.30805788
x2 = 0.15402894
x3 = 0.15402894
y1 = 0.0
y2 = -0.09324743
y3 = 0.09324743
v1x = 0.0
v2x = 0.96350817
v3x = -0.96350817
v1y = -1.015378093 
v2y = 0.507689046
v3y =  0.507689046

yl, yl1 = createArray(npts) 
yInitial = np.array([x1, y1, v1x, v1y, x2, y2, v2x, v2y, x3, y3, v3x, v3y])
yl[0,:] = yInitial


# Solution:
rl, vl = leapfrogIter(EOM, yl, t, dt)


# Plotting: 
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 62, blim = 0.5)
#%% ii)
# Parameters
m1 = 1
m2 = 1
m3 = 1
w0 = 2.47
T0 = 2 * np.pi / w0
tMax = 3 * T0
dt = 0.001
t = time(tMax, dt)

x1 = 0.97000436
x2 = -0.97000436
x3 = 0.0
y1 = -0.24308753
y2 = 0.24308753
y3 = 0.0
v1x = 0.93240737/2
v2x = 0.93240737/2
v3x = -0.93240737
v1y = 0.86473146/2
v2y =  0.86473146/2
v3y = -0.86473146

yl, yl1 = createArray(npts) 
yInitial = np.array([x1, y1, v1x, v1y, x2, y2, v2x, v2y, x3, y3, v3x, v3y])
yl[0,:] = yInitial

# Solution:
rl, vl = leapfrogIter(EOM, yl, t, dt)


# Plotting: 
animation(rl[:,0], rl[:,1], rl[:,2], rl[:,3], rl[:,4], rl[:,5], t, 66, blim = 1.5)
