""" This document contains my code for PHYS 479 assignment 3 which is devided into 
two parts. The first part (Question 1) involves solving the Time-Dependent Schodinger
Equation to simulate a quantim harmonic oscillator  using a Leapfrog technique. 
The second part (Question 2) uses the same technique to solve the Time-Dependent 
Schrodinger Equations for a double-wall potential problem. Results are checked by 
satisfying the Virial Theorem. The results are displayed using a simple animation 
that takes screenshots are specified time intervals. This code was coded using the 
Spyder IDE from anaonconda and Python 3.7. It can be ran all at once using any 
Python IDE or by cell in Spyder


   Author: Tristan Fleming
   Date: 2021-02-28
   Student number 20076612
   Version:  3  """

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as s
import scipy.integrate as integrate
import timeit
#%% 1a) Implement the space discreized leapfrog (SDLF)
# Functions: 
def createX(MIN, MAX, numPts):
    ''' This function takes in a minumum, maximum value and desired number of points. 
    It creates and returns a space array from the minimum value to the maximum value 
    with specified number of steps number of steps'''
    
    x = np.linspace(MIN, MAX, numPts)
    return x

def time(tMax, dt):
    '''This function creates and returns a time numpy array for a specified length 
    and timestep.'''
    
    t = np.arange(0, tMax, dt)
    return t

def createVj(x):
    '''This function creates and returns an array of potentials for a harmonic 
    osciallator with k = w0 = 1. It computes the potential for each value of x 
    of the input x array/list'''
    
    return 0.5 * x ** 2
        
def createA(Vj, h):
    '''This function creates and returns a tridagnol matrix  where the middle diagnol 
    is a function of h and Vj. The other two diagnols are given by h and have the 
    same value. Every other element has a value of 0'''
    
    N = len(Vj)
    A = np.zeros((N, N)) 
    b = -1 / (2 * (h ** 2))
    # Change first row
    A[0][0] = 1 / h ** 2 + Vj[0]
    A[0][1] = b
    # Change last row
    A[-1][-1] = 1 / h ** 2 + Vj[-1]
    A[-1][-2] = b
    
    for i in range(1,N - 1): # Update all diagnols for all rows except first/last
       row = np.zeros(N)
       row[i-1] = b
       row[i] = 1 / h ** 2 + Vj[i]
       row[i+1] = b
       A[i] = row
    
    return A

def calcPSI(x, x0, sigma, k0):
    '''This function calculates and returns the value of thewave function for time = 0 
    for a space discretization steps of x. It computes it using constants x0, sigma 
    and k0'''
    
    PSI = ((sigma * ((np.pi) ** 0.5)) ** -0.5) * np.exp((-(x - x0) ** 2) / (2 * (sigma ** 2)) + 1j*k0*x)
    return PSI

def createR(x, x0, sigma, k0):
    '''This function creates an array of the real components of the wave funcction 
    for all space deicretization steps of x. It does this by using the calcPSI 
    function'''
    
    N = len(x)
    R = np.zeros(N)
    
    for i in range(0, N):
        PSI = calcPSI(x[i], x0, sigma, k0)
        R[i] = PSI
    
    return R

def leapfrog(diffeq, R0, I0, t, h):
    '''Vecotirzed leapfrog method (specifically the Verlet method) using numpy 
    arrays. This function is a class of ODE solver. It updates position and velocity 
    at interleaved time points that are staggered and returns the position and 
    velocity for a full timestep.'''
    
    hh = h / 2.0 # The interleaved time step
    R12 = R0 + hh * diffeq(0, R0, I0, t) # Position at hh 
    I1 = I0 + h * diffeq(1, R12, I0, t + hh) # Velocity at h using R1
    R1 = R12 + hh * diffeq(0, R0, I1, t + h) # New position at h
    
    return R1, I1

def TDSE(id, R, I, t):
    '''This function represents the ODEs for the Harmonic Oscillator
    with m = k = w_0 = 1. It uses an ID to correspond to either ODE. An ID of zero 
    returns dR/dt and dI/dt is returned otherwise. '''
    
    if id == 0: # Generate velocity dR/dt
        dydt = A.dot(I)
    else: # Genertae acceleration dI/dt
        dydt = -A.dot(R)
    return dydt

def TDSEslice(id, R, I, t):
    '''This function represents the ODEs for the Harmonic Oscillator with 
    m = k = w_0 = 1. It uses an ID to correspond to either ODE. An ID of zero 
    returns dR/dt and dI/dt is returned otherwise. It slices all zero valued elements
    to reduce run time.'''
    
    b = 1 / (2 * (h**2))
    
    if id == 0: # Generate velocity dR/dt
        temp = -b * I
        dydt = (2 * b + createVj(x)) * I
        dydt[1:-1] += temp[:-2] + temp[2:]
        dydt[0] += temp[-1] + temp[1]
        dydt[-1] += temp[-2] + temp[0]
        return dydt
    else: # Genertae acceleration dI/dt
        temp = -b * R
        dydt = (2 * b + createVj(x)) * R
        dydt[1:-1] += temp[:-2] + temp[2:]
        dydt[0] += temp[-1] + temp[1]
        dydt[-1] += temp[-2] + temp[0]
        return -dydt

def createArrays(numSpace, numTime, R0):
    '''This function creates an array of arrays for the real and imaginary components 
    of the wave function'''
    
    Rt = np.zeros((numTime,numSpace))
    It = np.zeros((numTime, numSpace))
    Rt[0,:] = R0 # change initial array in Rt
    
    return Rt, It

def leapfrogIter(diffeq, Re, Im, t, dt):
    '''This function passes an array the same size as the time array and returns 
    a vector containing all positions and velocities from the differential 
    equation. It uses the leapfrog function to evaluate the differential equations.
    The function returns an array of all the positions and velocities and a list of
    the energies for every timestep.'''
    start = timeit.default_timer()  # start timer for solver
    for i in range(0, len(t) - 1):
        R1, I1 = leapfrog(diffeq, Re[i,:], Im[i,:], t[i], dt)
        Re[i+1,:] = R1
        Im[i+1,:] = I1
    stop = timeit.default_timer() # Stop timer
    time = stop - start
    return Re, Im, time

def niceFigure(useLatex=True):
    '''This function makes the visual plots look a lot better. If you don't have 
    Latex on your device just set useLatex = False.'''
    
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 25})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return 

def animation(x, PSI, V, xlim, ylim, figNum):
    '''This function creates an animation of the three bodies position. Each new
    point also contains a velocity vector to indicate the direction and magnitude
    of the planets velocity. It also takes snapshots of the animation at different 
    intervals and saves them.''' 
    
    frame = 1000 # Number of frames 
    fig = plt.figure()
    fig, ax1 = plt.subplots()
    
    ax1.set_xlim(-xlim,xlim); ax1.set_ylim(0,ylim)
    ax1.set(xlabel = "x (arb. units)", ylabel = "$|\psi(x,t)|^2$ (a.u.)")
    T = "0"
    T1 = ax1.text(-4, 1.15, T)
    
    yVals = PSI[0,:]
        
    plt.grid()
    plt.text(-9, 1, "$V(x)$")
    line1, = ax1.plot(x, yVals, 'b')
    line1.set_xdata(x)
    line2, = ax1.plot(x, V, 'g')

    plt.pause(2) # Wait 2 seconds to start animation 
    stopTime1 = len(PSI) // 3
    stopTime2 = len(PSI) // 4
    stopTime3 = len(PSI) // 9
    stopTime4 = len(PSI) - 1
    
    for i in range(len(PSI)): # To plot every specified frame
        if i % frame == 0: 
        
            yVals = PSI[i,:]
           
            line1.set_ydata(yVals)
            
            plt.draw()
            plt.pause(.1)
            
        if i == stopTime1 or i == stopTime2 or i == stopTime3 or i == stopTime4:
            # Save figures for different periods
            T = "frame = " + str(i) 
            T1.set_text(T)
            plt.savefig('Pic ' + str(figNum),  bbox_inches='tight', dpi = 300)
            figNum += 1

niceFigure(useLatex = False)
            
    
# Parameters: 
h = (10 - (-10)) / 1000 # Space discretization step size
x0 = -5
sigma = 0.5 
k0 = 0
w = 2 * np.pi # Normalized frequency 
tMax = 2 * w # Time for 2 periods
dt = 0.5 * h ** 2 # Time step
t = time(tMax, dt)
tnorm = [i / w for i in t] # Normalized time 
x = createX(-10, 10, 1000) # Create a space grid from -10 to 10 with 1000 points
R = createR(x, x0, sigma, k0)
Vj = createVj(x) # Create a list of potential energy for every space point
V = [i / 50 for i in Vj] # Normalized potential energy
A = createA(Vj, h) # Create a matrix A 
Atemp = A # Create a temporary array 
Re, Im = createArrays(len(x), len(t), R) # Create a matrix of the Re and Im parts


# Solution:
# Real and Imaginary components
Real, Imag, time = leapfrogIter(TDSEslice, Re, Im, t, dt)
print ("Time for the sliced matrix solver is: ", time) 
# Calculate PSI squared
PSIsqr = np.abs(Real + 1j*Imag)**2
PSIt = np.transpose(PSIsqr) # This is for the contourf plot



# Plotting:
animation(x, PSIsqr, V, 10, 1.25, 1)
#fig, ax = plt.subplots() 
#ax.contourf(tnorm, x, PSIt)
#ax.set(ylabel = "$x$ (a.u.)", xlabel = "$t$ ($T_0$)")
#plt.savefig('Pic 5',  bbox_inches='tight', dpi = 300)
#%% 1b) Full Matrix Solution: 
# Parameters:
Re, Im = createArrays(len(x), len(t), R) 


# Solutions:
Real, Imag, time = leapfrogIter(TDSE, Re, Im, t, dt)
print ("Time for the full matrix solver is: ", time) 
#%% Sparse Matrix solution:
# Parameters:
A = s.csr_matrix(A) # Create a sparse matrix
Re, Im = createArrays(len(x), len(t), R)


# Solutions:  
Real, Imag, time = leapfrogIter(TDSE, Re, Im, t, dt)
print ("Time for the sparse matrix solver is: ", time)
#%% Sclied approach Solution (2000 space points):
# Parameters:
x = createX(-10, 10, 2000) # Create a space grid from -10 to 10 with 2000 points
R = createR(x, x0, sigma, k0)
Vj = createVj(x) # Create a list of potential energy for every space point
A = createA(Vj, h) # Create a matrix A 
Re, Im = createArrays(len(x), len(t), R) 


# Solutions:
Real, Imag, time = leapfrogIter(TDSEslice, Re, Im, t, dt)
print ("Time for the sliced approach solver (using 2000 points) is: ", time) 
#%% Full Matrix Solution (2000 points):
# Parameters:
Re, Im = createArrays(len(x), len(t), R) # Create a matrix of the Re and Im parts


# Solutions:
Real, Imag, time = leapfrogIter(TDSE, Re, Im, t, dt)
print ("Time for the full matrix solver (using 2000 points) is: ", time)     
#%% Sparse Matrix solution (2000 space points):
A = s.csr_matrix(A) # Create a sparse matrix
Re, Im = createArrays(len(x), len(t), R)


# Solutions:  
Real, Imag, time = leapfrogIter(TDSE, Re, Im, t, dt)
print ("Time for the sparse matrix solver (using 2000 points) is: ", time)
#%% 1c) Check if the Viral Theorem is satisfied
# Calculate the expectation value of the potential 
Vtot = 0 # Counter for the sum of the integral 
for i in range(0, len(x)): # Evaluate every 
    V = 0.5 * (x[i] ** 2) / 50 * h
    Vtot += V
print(Vtot)


PSI = Real + 1j*Imag
Ttot = 0 
for i in range(0, len(PSI)): # Calculate for partial diff for every time
    diffPSI = np.diff(PSI[i,:]) / h # Compute diff for values of all x points
    
    for j in range(0, len(diffPSI)): # Compute the sum of every point 
        T = 0.5 * np.abs(diffPSI[j]**2) * h
        Ttot += T
Ttot = Ttot / len(PSI)
print(Ttot)
#%% 2a) Solve the TDSE for a double-wall potential 
# Functions:
def createVj(x):
    '''This function creates a list of potentials for a harmonic osciallator with 
    k = w0 = 1. It computes the potential for each value of x of the input x 
    array/list'''
    
    return (x**4) - 4 * (x**2)
    
def time(tMax, dt):
    '''This function creates and returns a time numpy array for a specified length 
    and timestep.'''
    
    t = np.arange(0, tMax, dt)
    return t

def animation(x, PSI, V, xlim, ylim, figNum):
    '''This function creates an animation of the three bodies position. Each new
    point also contains a velocity vector to indicate the direction and magnitude
    of the planets velocity. It also takes snapshots of the animation at different 
    intervals and saves them.''' 
    
    frame = 1000 # Number of frames 
    fig = plt.figure()
    fig, ax1 = plt.subplots()
    
    ax1.set_xlim(-xlim,xlim); ax1.set_ylim(0,ylim)
    ax1.set(xlabel = "x (arb. units)", ylabel = "$|\psi(x,t)|^2$ (a.u.)")
    T = "0"
    T1 = ax1.text(-2.5, 1.15, T)
    
    yVals = PSI[0,:]
        
    plt.grid()
    plt.text(-9, 1, "$V(x)$")
    line1, = ax1.plot(x, yVals, 'b')
    line1.set_xdata(x)
    line2, = ax1.plot(x, V, 'g')

    plt.pause(2) # Wait 2 seconds to start animation 
    stopTime1 = len(PSI) // 3
    stopTime2 = len(PSI) // 4
    stopTime3 = len(PSI) // 9
    stopTime4 = len(PSI) - 1
    
    for i in range(len(PSI)): # To plot every specified frame
        if i % frame == 0: 
        
            yVals = PSI[i,:]
           
            line1.set_ydata(yVals)
            
            plt.draw()
            plt.pause(.1)
            
        if i == stopTime1 or i == stopTime2 or i == stopTime3 or i == stopTime4:
            # Save figures for different periods
            T = "frame = " + str(i) 
            T1.set_text(T)
            plt.savefig('Pic ' + str(figNum),  bbox_inches='tight', dpi = 300)
            figNum += 1

niceFigure(useLatex = False)

# Parameters:
x0 = - np.sqrt(2)
sigma = 0.5
k0 = 0
h = (5 - (-5)) / 500 # Space discretization step size
tMax = 4 * w # Time for 2 periods
dt = 0.5 * h ** 2 # Time step
t = time(tMax, dt)
tnorm = [i / w for i in t] # Normalized time 
x = createX(-5, 5, 500) # Create a space grid from -10 to 10 with 1000 points
R = createR(x, x0, sigma, k0)
Vj = createVj(x) # Create a list of potential energy for every space point
V = [i / 50 for i in Vj] # Normalized potential energy
A = createA(Vj, h) # Create a matrix A 
Re, Im = createArrays(len(x), len(t), R) # Create a matrix of the Re and Im parts


# Solutions:
Real, Imag, time = leapfrogIter(TDSEslice, Re, Im, t, dt)
PSI = np.abs(Real + 1j*Imag)**2
PSIt = np.transpose(PSI) # This is for the contourf plot


# Plotting:
animation(x, PSI, V, 5, 1.25, 6)
#%%
Xtot = 0
for i in range(0, len(PSI)):
    for j in range(0, len(x)):
        X = PSI[i,j] * x[j] * h
        Xtot += X
    
X = Xtot / len(PSI)


#  Plotting: 
fig, ax = plt.subplots() 
ax.contourf(tnorm, x, PSIt)
ax.hlines(X, tnorm[0], tnorm[-1], color = 'r')
ax.set(ylabel = "$x$ (a.u.)", xlabel = "$t$ ($T_0$)")
plt.savefig('Pic 10',  bbox_inches='tight', dpi = 300)