''' This document contains my code for PHYS 479 density matrix theory and optical 
Bloch equation's assignment. My code is divided into 4 parts, one for each question. 
In part 1 (Question 1) I solve for laser excitation in the RWA using the Euler and 
Runge-Kutta methods. In part 2 (Question 2), I solve for a time-dependent pulse in 
the RWA. In part 3, (Question 3) I add laser detuning. In part 4 (Question 4), I 
compute the population densities for a variety of initial conditions. This code can 
be ran using the Spyder IDE from anaconda and Python 3.7. The code needs to run one 
cell at a time in Spyder. 

   Author: Tristan Fleming
   Date: 2021-01-18
   Student number 20076612
   Version:  5'''

import numpy as np
import matplotlib.pyplot as plt
import math as m
import timeit
from scipy.integrate import odeint 
#%% 
#1a) Begin with example 'sample bad code'
# Functions:
def FunctionQ1(y,t): 
    '''This function represents Optical Bloch equations in its simplest form
    for CW excitation in the RWA. I assume on resonance exitation'''
    
    dy = np.zeros((len(y))) # Create an array of zeros the size of y 
    
    dy[0] = 0. # Real component of change in coherence 
    dy[1] = Omega/2*(2.*y[2]-1.) # Imaginary component of change in coherence 
    dy[2] = -Omega*y[1] # Change in population density 
    return dy

def EulerForward(f,y,t, h=0.01): 
    '''This function is a vectorized forward Euler ODE solver'''
    
    k1 = h * np.asarray(f(y,t)) # Converts to numpy array                     
    y = y + k1
    return y 

def arrayInitializer(npts):
    ''' This function is to re-initialize the array that I pass in my ODE
    solvers'''
    
    y = np.zeros((npts,3)) # Create a 3 by npts matrix
    yinit = np.array([0.0,0.0,0.0]) # Initilize a row
    y1 = yinit # temporary variable
    y[0,:] = y1
    return y1, y
    

# Parameters: 
Omega = 2 * np.pi 
dt = 0.001 # Time step
tmax = 5. # Final time
t = np.arange(0.0, tmax, dt) 
npts = len(t)
y1, y = arrayInitializer(npts)


# ODE solutions: 
# Forward Euler solution 
start = timeit.default_timer()  # start timer for solver
for i in range(1,npts):   # loop over time
    y1 = EulerForward(FunctionQ1,y1,t[i-1],dt) 
    y[i,:]= y1
stop = timeit.default_timer() # Stop timer
print ("Time for Euler ODE Solver", stop - start) 

# Exact solution 
yexact = [m.sin(Omega*t[i]/2)**2 for i in range(npts)]


# Plotting: 
plt.plot(t, yexact, 'b')
plt.plot(t, y[:,2], 'r')

plt.legend(["Exact solution", "Forward Euler"], loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show() 

#%%
#1b) Implement an RK4 ODE solver and compare the solution with the Euler solution. 
# Plot all three solutions. 
# Functions:
def RK4(f, y,t,h):
    '''This function is a vectorized rk4 ODE solver'''
 
    k1 = h * np.asarray(f(y,t))
    k2 = h * np.asarray(f(y+(k1/2), t+(h/2)))
    k3 = h * np.asarray(f(y+(k2/2), t+(h/2)))
    k4 = h * np.asarray(f(y+k3, t+h))
    y = y + (k1 + 2*k2 + 2*k3 + k4)/6
    return y


# Parameters:
yr1, yr = arrayInitializer(npts)


# rk4 solutions 
start = timeit.default_timer()
for i in range(1,npts):   # loop over time
    yr1 = RK4(FunctionQ1,yr1,t[i-1],dt) # this is slightly longer
    yr[i,:]= yr1
stop = timeit.default_timer()
print ("Time for RK4 ODE Solver", stop - start) 


# Plotting: 
plt.plot(t, yexact, 'b')
plt.plot(t, y[:,2], 'r')
plt.plot(t, yr[:,2], 'y')

plt.legend(["Exact solution", "Forward Euler", "RK4"], loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show() 

#%% 
# 1c) Reduce the step size to the 'rule of thumb' (h = 0.01) and plot the RK4 
# solutions versus the analytical one 
#Parameters:
tRK4 = np.arange(0.0, 5., 0.01) # change number of incremenets to rule of thumb (100)
npts = len(tRK4)
yr1, yr = arrayInitializer(npts)


# ODE solutions: 
# RK4 solutions
start = timeit.default_timer()
for i in range(1,npts):
    yr1 = RK4(FunctionQ1,yr1,tRK4[i-1],0.01) # this is slightly longer
    yr[i,:]= yr1
stop = timeit.default_timer()
print ("Time for RK4 ODE Solver", stop - start) 


# Plotting:
plt.plot(t, yexact, 'b')
plt.plot(tRK4, yr[:,2], 'y')

plt.legend(["Exact solution", "RK4"], loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show() 

#%%
#1d) Substantially improve the graphics and plotting
def niceFigure(): 
    plt.rcParams.update({'font.size':20})
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['axes.ymargin'] = 0.0
    plt.rcParams['axes.xmargin'] = 0.0
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    return 

   
niceFigure()
# Plotting:
plt.plot(t, yexact, 'b')
plt.plot(tRK4, yr[:,2], 'y')

plt.xlabel('$Time$')
plt.ylabel('$Norm. amplitude$')
plt.ylim(0,1.25)
plt.savefig("Plot 1", bbox_inches='tight')
plt.tight_layout()
plt.show() 

#%%
# 2) Use the standard RWA equations with a time dependent Gaussian pulse
# Functions:
def functionQ2(y,t): 
    '''This function represents Optical Bloch equations in its simplest form
    for CW excitation in the RWA. I assume on-resonance excitation with a time-
    dependent pulse'''
    
    dy = np.zeros((len(y))) 
    
    dy[0] = 0. 
    dy[1] = Omega(t)/2*(2.*y[2]-1.)
    dy[2] = -Omega(t)*y[1] 
    return dy

def Omega(t):
    '''This function represents the time-dependent pulsed laser '''
    tp = 1 / (np.pi) ** 0.5 # Nomarlized pulse width 
    return 2 * np.pi * np.exp(-(t - 5*tp)**2 / tp**2)


# Parameters:
t = np.arange(0.0, 10., 0.01) # extend time to 10s still using the rule of thumb
npts = len(t) 
yr1, yr = arrayInitializer(npts)


# RK4 solution:
for i in range(1,npts):
    yr1 = RK4(functionQ2,yr1,t[i-1],0.01) # this is slightly longer
    yr[i,:]= yr1
    

# Plotting:
plt.plot(t, yr[:,2], 'b')
plt.ylabel('$Norm. amplitude$')
plt.xlabel('$Time$')
plt.ylim(0,1.25)
plt.savefig("Plot 2", bbox_inches='tight')
 
plt.show()
#%%
#3)a) Add a finite laser detuning and vary it from 0 to Omega_0. Plot maximum
# versus detuning
# Functions:
def functionQ3(y,t): 
    '''This function represents Optical Bloch equations with variable detuning
    and dephasing ith a time-dependent pulse'''
    dy = np.zeros((len(y))) 
    
    dy[0] = -gamma * y[0] + delta_0 * y[1] # Redefine the detunning term 
    dy[1] = -gamma*y[1] - delta_0 * y[0] +  Omega(t)/2*(2.*y[2]-1.) 
    dy[2] = -Omega(t)*y[1] 
    return dy


# Parameters: 
Omega_0 = 2 * np.pi
gamma = 0
yr1, yr = arrayInitializer(npts)
peak_delta = [] # Create a list for peak laser detuning 
peak_gamma = [] # Create a list for peak dephasing 
num_iterations = (Omega_0/100) # For looping below
iterations = np.linspace(0,Omega_0, 100)


# Solutions:
k = 0
while k < Omega_0: # Loop through variable detuning 
    yr1, yr = arrayInitializer(npts)
    delta_0 = k 
    
    for i in range(1,npts): # compute peak detuning 
        yr1 = RK4(functionQ3,yr1,t[i-1],0.01) # this is slightly longer
        yr[i,:]= yr1
    peak = yr[:,2].max()
    peak_delta.append(peak) # Append peak to peak_delta
    
    k = k + num_iterations
    
delta_0 = 0 # Set detuning to zero to find peak dephasing 

k = 0
while k < Omega_0:
    yr1, yr = arrayInitializer(npts)
    gamma = k
    
    for i in range(1,npts):
        yr1 = RK4(functionQ3,yr1,t[i-1],0.01) # this is slightly longer
        yr[i,:]= yr1
    peak = yr[:,2].max()
    peak_gamma.append(peak)
    
    k = k + num_iterations
    

# Plotting:
plt.plot(iterations, peak_delta[:len(iterations)], 'b')
plt.ylabel('$ Peak n_e$')
plt.xlabel('$ \Delta_{0L}$')
plt.ylim(0,1.25)
plt.savefig('Plot 3', bbox_inches='tight')
plt.show()

plt.plot(iterations, peak_gamma[:len(iterations)], 'b')
plt.ylabel('$ Peak n_e$')
plt.xlabel('$ \gamma_{d}$')
plt.ylim(0,1.25)
plt.savefig('Plot 4',  bbox_inches='tight')

plt.show()

#%%
# 4a) Investigate u(t) and n(t) when wl = different factors of Omega_0
#Functions: 
def functionQ4(y,t): 
    '''This represents the optical block equations with no rotating wave approximation.
    We use a full-wave Rabi field.'''

    dy = np.zeros((len(y)))  
    w_0 = w_l # Because we stay on resonance 
    
    dy[0] = -gamma * y[0] + w_0 * y[1] 
    dy[1] = -gamma * y[1] - w_0 * y[0] + Omega(t) * (2. * y[2] - 1.)  
    dy[2] = -2 *Omega(t) * y[1] 
    return dy

def Omega(t):
    '''This function represents a full-wave Rabi field with a normalized width '''
    tp = 1 
    return Omega_0 * np.exp(-(t - 5)**2/ tp**2) * np.sin(w_l * t + phi)


# Paramters:
phi = 0
Omega_0 = 2 * np.pi ** 0.5
w_l = 20 * Omega_0 # Vary the laser frequency 
gamma = 0
yr1, yr = arrayInitializer(npts)


# ODE solution: 
# RK4 solution 
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1
    

# Plotting: 
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.ylabel('$Norm. amplitude$')
plt.xlabel('$Time$')
plt.ylim(-.75,1.25)
plt.legend(["$n_e$", '$Re[u]$','$Im[u]$'], loc='best', fancybox = True)
plt.savefig('Plot 5',  bbox_inches='tight', dpi = 300)
plt.show()

#%%
# Parameters: 
yr1, yr = arrayInitializer(npts)
w_l = 10 * Omega_0 # Change laser frequency 


# Solution: 
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1
    

# Plotting 
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.ylabel('$Norm. amplitude$')
plt.xlabel('$Time$')
plt.ylim(-.75,1.25)
plt.savefig('Plot 6',  bbox_inches='tight', dpi = 300)

plt.show()

#%%
# Parameters: 
yr1, yr = arrayInitializer(npts)
w_l = 5 * Omega_0 # Change laser frequency 


# Solution: 
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1
    

# Plotting 
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.ylabel('$Norm. amplitude$')
plt.xlabel('$Time$')
plt.ylim(-.75,1.25)
plt.savefig('Plot 7',  bbox_inches='tight', dpi = 300)

plt.show()

#%%
# Parameters
yr1, yr = arrayInitializer(npts)
w_l = 2 * Omega_0
yRWA1, yRWA = arrayInitializer(npts) # Solutions with RWA


# Solutions: 
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1

for i in range(1,npts):
    yRWA1 = RK4(functionQ2, yRWA1,t[i-1],0.01)
    yRWA[i,:]= yRWA1
    

# Plotting: 
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.ylabel('$Norm. amplitude$')
plt.xlabel('$Time$')
plt.ylim(-.75,1.25)
plt.savefig('Plot 8',  bbox_inches='tight', dpi = 300)
plt.show()

plt.plot(t, yRWA[:,2], 'b')
plt.plot(t, yRWA[:,0], '--g')
plt.plot(t, yRWA[:,1], '--r',)
plt.ylabel('$Norm. amplitude$')
plt.xlabel('$Time$')
plt.ylim(-.5,0.5)
plt.savefig('Plot 9',  bbox_inches='tight', dpi = 300)
plt.show()

#%%
# Parameters:
phi = np.pi / 2 
yr1, yr = arrayInitializer(npts)
yRWA1, yRWA = arrayInitializer(npts)


# Solutions:
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1

for i in range(1,npts):
    yRWA1 = RK4(functionQ2, yRWA1,t[i-1],0.01) # Function from Q2 since we are on
    yRWA[i,:]= yRWA1                           # Resonance with no dephasing
    

# Plotting:
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.legend(["$n_e$", '$Re[u]$','$Im[u]$'], loc='best', fancybox = True)
plt.xlabel('$Time$')
plt.ylabel('$Norm. amplitude$')
plt.ylim(-.75,1.25)
plt.savefig('Plot 10',  bbox_inches='tight', dpi = 300)
plt.show()

plt.plot(t, yRWA[:,2], 'b')
plt.plot(t, yRWA[:,0], '--g')
plt.plot(t, yRWA[:,1], '--r',)
plt.xlabel('$Time$')
plt.ylabel('$Norm. amplitude$')
plt.ylim(-0.5,0.5)
plt.savefig('Plot 11',  bbox_inches='tight', dpi = 300)
plt.show()

#%% 
# 4b) Investigate different pulse areas for a laser frequency of 4 * sqrt(pi))
# Parameters:
w_l = 4 * np.pi ** 0.5
yr1, yr = arrayInitializer(npts)
phi = 0


# Solution:
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1


# Plotting:
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.xlabel('$Time$')
plt.ylabel('$Norm. amplitude$')
plt.ylim(-.75, 1.25)
plt.legend(["$n_e$", '$Re[u]$','$Im[u]$'], loc='best', fancybox = True)
plt.savefig('Plot 12',  bbox_inches='tight', dpi = 300)
plt.show()

#%%
# Parameters:
Omega_0 = 10 * np.pi ** 0.5  # For pulse area of 10pi
yr1, yr = arrayInitializer(npts)


# Solution:
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) # this is slightly longer
    yr[i,:]= yr1


# Plotting
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.xlabel('$Time$')
plt.ylabel('$Norm. amplitude$')
plt.ylim(-.75, 1.25)
plt.savefig('Plot 13',  bbox_inches='tight', dpi = 300)
plt.show()

#%%
# Parameters:
Omega_0 = 20 * np.pi ** 0.5 # For pulse area of 20pi
yr1, yr = arrayInitializer(npts)

for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) # this is slightly longer
    yr[i,:]= yr1


# Plotting:
plt.plot(t, yr[:,2], 'b')
plt.plot(t, yr[:,0], '--g')
plt.plot(t, yr[:,1], '--r',)
plt.ylim(-.75,1.25)
plt.xlabel('$Time$')
plt.ylabel('$Norm. amplitude$')
plt.savefig('Plot 14',  bbox_inches='tight', dpi = 300)
plt.show()

#%%
# 4c) Plot the power spectrum in normalzied units using fast fourier transforms
t = np.arange(0.0, 50., 0.01) # extend time to 10s still using the rule of thumb
npts = len(t) 
yr1, yr = arrayInitializer(npts)
Omega_0 = 2 * np.pi ** 0.5


# Solution:
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1
    

P = yr[:,0]
P_fft = np.fft.fftshift(P)
n = len(P)
omega = np.fft.fftfreq(n,d = 0.01)

omega_norm = [] # Normalize frequency 
for element in omega:
    omega_norm.append((element * 2*np.pi)/w_l)


#  Plotting: 
plt.semilogy(omega_norm, P_fft, 'b')
plt.ylabel('$Re[u]$')
plt.xlim(0,6)
plt.xlabel('$\omega / \omega_L $')
plt.ylim(-.5,.75)
plt.savefig('Plot 15',  bbox_inches='tight', dpi = 300)
plt.show()
#%%
#Parameters: 
Omega_0 = 10 * np.pi ** 0.5 
yr1, yr = arrayInitializer(npts)

# Solution:
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1
    

P = yr[:,0]
P_fft = np.fft.fftshift(P)
n = len(P)
omega = np.fft.fftfreq(n,d = 0.01)

omega_norm = [] # Normalize frequency 
for element in omega:
    omega_norm.append((element * 2*np.pi)/w_l)


#  Plotting: 
plt.semilogy(omega_norm, P_fft, 'b')
plt.ylabel('$Re[u]$')
plt.xlabel('$\omega / \omega_L $')
plt.ylim(-.5,.75)
plt.xlim(0,6)
plt.savefig('Plot 16',  bbox_inches='tight', dpi = 300)
plt.show()
#%%
#Parameters: 
Omega_0 = 20 * np.pi ** 0.5 
yr1, yr = arrayInitializer(npts)

# Solution:
for i in range(1,npts):
    yr1 = RK4(functionQ4,yr1,t[i-1],0.01) 
    yr[i,:]= yr1
    

P = yr[:,0]
P_fft = np.fft.fftshift(P)
n = len(P)
omega = np.fft.fftfreq(n,d = 0.01)

omega_norm = [] # Normalize frequency 
for element in omega:
    omega_norm.append((element * 2*np.pi)/w_l)


#  Plotting: 
plt.semilogy(omega_norm, P_fft, 'b')
plt.ylabel('$Re[u]$')
plt.xlabel('$\omega / \omega_L $')
plt.ylim(-.5,.75)
plt.xlim(0,6)
plt.savefig('Plot 17',  bbox_inches='tight', dpi = 300)
plt.show()
