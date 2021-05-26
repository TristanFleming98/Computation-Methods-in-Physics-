import random as rnd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 21})
#%%
'''1a) implement the Metropolis algorithm and show the initial spins configurations evolving to 
equilibirum with time. Choose a spin chain of N = 50 and the three temperatures of kbT = 0.1, 0.5
and 1.'''
# Functions:
def initialized(N):
    '''Sets the initial microstates using a ordered parameter, p and returns
    the spin for N microstates, the total energy and magnetisim.'''
    spin, E, M = np.ones(N), 0., 0. # Initialize spin, energy and magnetism
    for i in range(0,N-1): # assign a spin for each microstate
        if (rnd.random() < p): 
            spin[i] = -1
            E = E - spin[i-1] * spin[i] # Energy
            M = M + spin[i] # Magnetization 
    spin[N-1] = spin[0] # Boundary condition 
    return spin, E - spin[N-1]*spin[0], M + spin[0]

def initial():
    '''This functin re-initializes the list used to keep track of evolution of 
    specified parameters and returns them.'''
    Energies = []
    Magnetization = []
    return Energies, Magnetization 

def update(N, i, spin, kT, E, M):
    ''' This function takes updates the microstates by randomly seclecting a microstate and
    decides whether or not it should flip the spin of that value, then updates the microstate, 
    energy and magnetism of the overall microstate.'''
    flip = 0
    dE = 2 * spin[i] * (spin[i-1] + spin[(i+1)%N]) # periodic boundary condition 
    if (dE < 0.0): # if delta_E<0, accept flip
        flip = 1
    else:
        p = np.exp(-dE/kT)
        if (rnd.random() < p):
            flip = 1 # accept the flip 
    if flip == 1: # update new E and M
        E = E + dE
        M = M - 2 * spin[i]
        spin[i] = -spin[i]
    return E, M, spin

def implement(N, kT, numIter, part = 'a'):
    '''This function implements the update function until the system approaches equilibrium. It
    iterates until every microstate has had a chance to flip spins N times.'''
    spin, E, M = initialized(N)
    count = 0
    spinEvolution = np.ones(N)
    while count < 5000:
        i = rnd.randint(0, N-1)
        if numIter[i] < 2*N-1:
            E, M, spin = update(N, i, spin, kT, E, M)
            count += 1
            spinEvolution = np.vstack((spinEvolution, spin))
            Energies.append(E/N)
            if part == 'b': 
                Magnetization.append(M/N)
    return spinEvolution

def FigureEnergy(xdata, ydata, ydata2, xlab, ylab, figname):
    '''This function create a nice plot for the energy function and saves the figure.'''
    fig = plt.figure()
    plt.plot(xdata, ydata, 'b', label = '$E$')
    plt.plot(xdata, ydata2, 'r', label = '$E_{an}$')
    plt.xlim(-.1,100)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    fig.show()
    fig.savefig(figname, dpi = 500)
    
def FigureEvolution(xdata, ydata, zdata, xlab, ylab, figname):
    '''This function creates a nice plot for the evolution of the spin configurations and save the figure'''
    fig = plt.figure()
    plt.pcolormesh(xdata, ydata, zdata)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.show()
    plt.savefig(figname, dpi = 500)
    
    
# Parameters:
#p = 0.6 # Warm start
p = 0.95 # Cold start
kT = 0.1
Beta = 1/kT
N = 50
numIter = [0] * N
NIter = np.linspace(0,5000,5000)
Energies, Magnetization = initial()
E_a = [-np.tanh(Beta)] * 5000


# Results:
spinEvolution = implement(N, kT, numIter)
FigureEnergy(NIter/N, Energies, E_a, 'Iteration$/N$', 'Energy$/N\epsilon$', 'Fig1')
#%%
microstates = np.arange(1,51)
numIterations = np.arange(0,len(Energies))/N
FigureEvolution(numIterations, microstates, spinEvolution[1:].T, 'Iteration$/N$', '$N$ spins', 'Fig2')
#%% kT = 0.5
# Parameters:
kT = 0.5
Beta = 1/kT
E_a = [-np.tanh(Beta)] * 5000
Energies, Magnetization = initial()


# Results:
spinEvolution = implement(N, kT, numIter)
FigureEnergy(NIter/N, Energies, E_a, 'Iteration$/N$', 'Energy$/N\epsilon$', 'Fig3')
#%%
FigureEvolution(numIterations, microstates, spinEvolution[1:].T, 'Iteration$/N$', '$N$ spins', 'Fig4')
#%% kT = 1
# Parameters:
kT = 1
Beta = 1/kT
E_a = [-np.tanh(Beta)] * 5000
Energies, Magnetization = initial()


# Results:
spinEvolution = implement(N, kT, numIter)
FigureEnergy(NIter/N, Energies, E_a, 'Iteration$/N$', 'Energy$/N\epsilon$', 'Fig5')
#%%
FigureEvolution(numIterations, microstates, spinEvolution[1:].T, 'Iteration$/N$', '$N$ spins', 'Fig6')
#%%
'''1b) Compute the average energy, magnetization, and entropy for a temperature range that spans kT = 0 to
kT = 6.'''
# Parameters:
E_n = []
E_a = []
M_n = []
S_n = []
S_a = []
kT_vals = []


# Functions:
def Figure_b(xdata, ydata, ydata2, xlab, ylab, figname):
    '''This function create a nice plot for the compouted values and saves the figure.'''
    fig = plt.figure()
    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata, ydata2, 'r')
    plt.xlim(0,6)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    fig.show()
    fig.savefig(figname, dpi = 500)
    

# Results:  
for i in range(1,300):
    kT = i * 0.02
    Beta = 1 / kT
    Energies, Magnetization = initial()
    spinEvolution = implement(N, kT, numIter, 'b')
    Ave_E = np.mean(Energies[3000:-1])
    Ave_M = np.mean(Magnetization[3000:-1])
    E_n.append(Ave_E)
    E_a.append(-np.tanh(Beta))
    M_n.append(Ave_M)
    kT_vals.append(kT)
    if i >= 2:
        delta_E = E_n[i-1] - E_n[0]
        S_n.append(delta_E)
        S_a.append(np.log(2 * np.cosh(Beta)) - Beta * np.tanh(Beta))        
    
Figure_b(kT_vals, E_n, E_a, '$kT/\epsilon$', '$E/N\epsilon$', 'Fig7') # Energy
#%%
M_a = np.zeros(len(M_n))
Figure_b(kT_vals, M_n, M_a, '$kT/\epsilon$', '$M/N$', 'Fig8') # Magnitization 
#%%
Figure_b(kT_vals[0:-1], S_n, S_a, '$kT/\epsilon$', '$M/N$', 'Fig9') # Entropy