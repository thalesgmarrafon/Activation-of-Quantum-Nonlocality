from math import *
import numpy as np
from numpy.linalg import eigh
from tqdm import tqdm
from time import time
import SM as sm

C = np.array([[[[1, 1], [1, -3]], [[-1, 1], [1, -1]], [[-1, 1], [1, -1]]], [[[-1, 1], [1, -1]], [[1, 1], [1, -3]], [[1, -1], [-1, 1]]], 
[[[-1, 1], [1, -1]], [[1, -1], [-1, 1]], [[0, 0], [0, 0]]]]) #c[y,x,b,a], I3322 inequality

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------FUNCTION FOR CALCULATION OF PARTIAL TRACE WITH NUMPY-----------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def partial_trace(rho, subsystem):
    rho_tensor = rho.reshape(2, 2, 2, 2)
    if subsystem == 'A': #Tr_A(rho)
        return np.trace(rho_tensor, axis1 = 0, axis2 = 2)
    elif subsystem == 'B': #Tr_B(rho)
        return np.trace(rho_tensor, axis1 = 1, axis2 = 3)
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------FUNCTION FOR CREATION OF RANDOM PROJECTIVE MEASUREMENTS TO BE USED BY ALICE--------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def PArandom():
    thetaa00 = np.random.uniform(0, pi)
    phia00 = np.random.uniform(0, 2*pi)
    statea00 = np.array([[cos(thetaa00/2)], [sin(thetaa00/2)*e**((phia00)*1j)]])
    thetaa01 = np.random.uniform(0, pi)
    phia01 = np.random.uniform(0, 2*pi)
    statea01 = np.array([[cos(thetaa01/2)], [sin(thetaa01/2)*e**((phia01)*1j)]])
    thetaa02 = np.random.uniform(0, pi)
    phia02 = np.random.uniform(0, 2*pi)
    statea02 = np.array([[cos(thetaa02/2)], [sin(thetaa02/2)*e**((phia02)*1j)]])
    A00 = statea00@(statea00.conj().T)
    A01 = statea01@(statea01.conj().T)
    A02 = statea02@(statea02.conj().T)
    A10 = sm.I - A00
    A11 = sm.I - A01
    A12 = sm.I - A02

    PA = np.array([[A00, A01, A02], [A10, A11, A12]]) #Aax = A(a|x), measurement x and result a

    return PA

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------FUNCTION FOR ITERATION OF THE METHOD OF MAXIMIZING INEQUALITY OPERATOR VALUE------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def max_ineq(state, rep): #Iteration of the method a number 'rep' of times and then test inequality for state 'state'

    LAMBB = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) #Initializing Bob's eigenvalues to be calculated during the iteration
    LAMBVECB = np.empty(12, complex).reshape(3,2,2) #Initializing Bob's eigenvectors

    LAMBA = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) #Alice's eigenvalues
    LAMBVECA = np.empty(12, complex).reshape(3,2,2) #Alice's eigenvectors

    RHOB = np.zeros((24), complex).reshape(2,3,2,2) #Array to store operators rho_b
    QB = np.zeros((24), complex).reshape(2,3,2,2) #Array to store Bob's measurements

    RHOA = np.zeros((24), complex).reshape(2,3,2,2) #Array to store operators rho_a analogous to rho_b
    PA = PArandom() #Array to store Alice's measurements

    i = 0

    t0 = time()
    t1 = time()

    while i < rep and t1 - t0 < 60.0:
        t1 = time()
        for b in [0, 1]: #Criation of operators rho_b
                for y in [0, 1, 2]:
                    rhob = np.zeros((2, 2), complex)
                    for a in [0, 1]:
                        for x in [0, 1, 2]:
                            rhob += C[y, x, b, a]*partial_trace(state@(np.kron(PA[a, x], sm.I)), 'A')
                    RHOB[b, y] = rhob

        for y in [0, 1, 2]: #Spectral decomposition
            rhopm = RHOB[1, y] - RHOB[0, y]
            lamb, lambvec = eigh(rhopm)
            LAMBB[y*2], LAMBB[y*2 + 1] = lamb
            LAMBVECB[y] = lambvec
        
        if (LAMBB[0] <= 0 and LAMBB[1] <= 0) or (LAMBB[2] <= 0 and LAMBB[3] <= 0) or (LAMBB[4] <= 0 and LAMBB[5] <= 0): #If there are no positive eigenvalues for one of
            #the operators then there are no positive subspace to be used in the optimization, which leads to new random measurements for Alice
            PA = PArandom()
            i = 0
            continue
        
        QB = np.zeros((24), complex).reshape(2,3,2,2)
        for y in [0, 1, 2]: #Criation of Bob's measurements using the positive subspaces
            lamb = LAMBB[y*2], LAMBB[y*2 + 1]
            lambvec = LAMBVECB[y]
            for j, e in enumerate(lamb):
                if e > 0:
                    ev = np.array([[lambvec[0, j]], [lambvec[1, j]]])
                    QB[1, y] += ev@ev.conj().T
            QB[0, y] = sm.I - QB[1, y]


#Doing all the optimization method again, but now for Alice

        for a in [0, 1]:
                for x in [0, 1, 2]:
                    rhoa = np.zeros((2, 2), complex)
                    for b in [0, 1]:
                        for y in [0, 1, 2]:
                            rhoa += C[y, x, b, a]*partial_trace(state@(np.kron(sm.I, QB[b, y])), 'B')
                    RHOA[a, x] = rhoa

        for x in [0, 1, 2]:
            rhopm = RHOA[1, x] - RHOA[0, x]
            lamb, lambvec = eigh(rhopm)
            LAMBA[x*2], LAMBA[x*2 + 1] = lamb
            LAMBVECA[y] = lambvec
        
        if (LAMBA[0] <= 0 and LAMBA[1] <= 0) or (LAMBA[2] <= 0 and LAMBA[3] <= 0) or (LAMBA[4] <= 0 and LAMBA[5] <= 0):
            PA = PArandom()
            i = 0
            continue
        
        PA = np.zeros((24), complex).reshape(2,3,2,2)
        for x in [0, 1, 2]:
            lamb = LAMBA[x*2], LAMBA[x*2 + 1]
            lambvec = LAMBVECA[x]
            for j, e in enumerate(lamb):
                if e > 0:
                    ev = np.array([[lambvec[0, j]], [lambvec[1, j]]])
                    PA[1, x] += ev@ev.conj().T
            PA[0, x] = sm.I - PA[1, x]

        i += 1

    beta = np.zeros((4, 4), complex) #Initialization and definition of the inequality's operator

    for a in [0, 1]:
        for b in [0, 1]:
            for x in [0, 1, 2]:
                for y in [0, 1, 2]:
                    beta += C[y, x, b, a]*(np.kron(PA[a, x], QB[b, y]))

    S = np.trace(state@beta).real #Value of the operator of the inequality

    if t1 - t0 < 60.0:
        return S - 4.0, PA, QB
    else:
        return -2.0, PA, QB
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------FUNCTION FOR BISSECTION METHOD----------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def bis(ent, noise, acc_bis, iter_ineq): #Bissection method using optimization with 'iter_ineq' iterations to find minimum w , with accuracy 'acc_bis'
    #for which the state w*'ent' + (1 - w)*'noise' is nonlocal

    w0 = 0.0
    w1 = 1.0
    whalf = (w0 + w1)/2
    statehalf = whalf*ent + (1 - whalf)*noise
    S0 = -2.0
    ineqhalf = max_ineq(statehalf, iter_ineq)
    ineq1 = ineqhalf
    Shalf = ineqhalf[0]

    rep = int(np.ceil(np.log2(1/acc_bis) - 1))

    for counter_bis in tqdm(range(rep), desc = 'bissection loop', ncols = 190, leave = False):
        if S0*Shalf > 0:
            w0 = whalf
            whalf = (w0 + w1)/2
            statehalf = whalf*ent + (1 - whalf)*noise
            S0 = Shalf
            ineqhalf = max_ineq(statehalf, iter_ineq)
            Shalf = ineqhalf[0]

        else:
            w1 = whalf
            whalf = (w0 + w1)/2
            statehalf = whalf*ent + (1 - whalf)*noise
            ineq1 = ineqhalf
            ineqhalf = max_ineq(statehalf, iter_ineq)
            Shalf = ineqhalf[0]

    return w1, ineq1[1], ineq1[2]


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------FUNCTION FOR REPEATING BISSECTION AND OUTPUTING RESULTS FOR THE MINIMUM W FOUND-----------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def W(ent, noise, acc_bis, iter_ineq, rep_W): #The bissection method is repeated 'rep_W' times and the results outputs for the minimum w found are used
    W = []
    PAM = []
    QBM = []
    for counter_W in tqdm(range(rep_W), desc = 'recalculation loop'):
        MIN = bis(ent, noise, acc_bis, iter_ineq)
        W.append(MIN[0])
        PAM.append(MIN[1])
        QBM.append(MIN[2])

    min_index = np.argmin(W)
    w, PA, QB = W[min_index], PAM[min_index], QBM[min_index]
    print('The state is nonlocal for w >', str(w))
    print('The operators found which violates inequality for w =', str(w), 'are:')
    print('PA:\n', str(PA))
    print('QB:\n', str(QB))

    beta = np.zeros((4, 4), complex)
    state = w*ent + (1 - w)*noise

    for a in [0, 1]:
            for b in [0, 1]:
                for x in [0, 1, 2]:
                    for y in [0, 1, 2]:
                        beta += C[y, x, b, a]*(np.kron(PA[a, x], QB[b, y]))

    S = np.trace(state@beta).real
    print('For w, PA, QB given the value of the inequality is:', str(S))