import numpy as np
import numpy.linalg as la
from math import sqrt, sin, cos, pi
from time import time
from tqdm import tqdm
import SM as sm

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------BASIC FUNCTIONS----------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def obs(theta, phi): #Observable constructed from the angles 'theta' and 'phi' on the Bloch sphere of the associated dichotomous projective measurements
    state = np.array([[cos(theta/2)], [sin(theta/2)*np.exp((phi)*1j)]])
    return 2*state@np.transpose(np.conjugate(state)) - sm.I

def multikron(n, op): #Tensor product of operator 'op' with itself '(n - 1)' times
    a = op
    for i in range(n - 1):
        a = np.kron(a, op)
    return a

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------FUNCTIONS FOR CREATION OF AND OPERATIONS OVER THE OPERATOR OF THE INEQUALITY-----------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def G(n, A1, A2, B1, B2, scenario): #Creation of operators of the inequality for 'n' shared resources and specified 'scenario',
#fixed oeprators 'A1' and 'A2' for each independent Alice and fixed measurements for Bob
    if scenario == 'star':
        Is = (1/2**((n)/2))*multikron(n, np.kron((A1 + A2), B1))
        #Is = (1/2**(n))*np.kron(np.kron((A1 + A2), B), (A1 + A2))
        Js = (1/2**((n)/2))*multikron(n, np.kron((A1 - A2), B2))
        #Js = (1/2**(n))*np.kron(np.kron((A1 - A2), B), (A1 - A2))
    elif scenario == 'chain':
        Is = (1/2**(np.ceil(n+1)/2))*multikron(n, np.kron((A1 + A2), B1))
        Js = (1/2**(np.ceil(n+1)/2))*multikron(n, np.kron((A1 - A2), B2))
    elif scenario == 'cyclic':
        Is = (1/2**(np.floor(n-1)/2))*multikron(n, np.kron((A1 + A2), B1))
        Js = (1/2**(np.floor(n-1)/2))*multikron(n, np.kron((A1 - A2), B2))
    return Is, Js

def meanG(rho, n, A1, A2, B1, B2, scenario): #Value of the inequality for the shared resource 'rho'
    if scenario == 'star':
        return (abs(np.trace(G(n, A1, A2, B1, B2, 'star')[0]@rho)))**(1/((n)/2)) + (abs(np.trace(G(n, A1, A2, B1, B2, 'star')[1]@rho)))**(1/((n)/2))
    elif scenario == 'chain':
        return (abs(np.trace(G(n, A1, A2, B1, B2, 'chain')[0]@rho)))**(1/(np.ceil(n+1)/2)) + (abs(np.trace(G(n, A1, A2, B1, B2, 'chain')[1]@rho)))**(1/(np.ceil(n+1)/2))
    elif scenario == 'cyclic':
        return (abs(np.trace(G(n, A1, A2, B1, B2, 'cyclic')[0]@rho)))**(1/(np.floor(n-1)/2)) + (abs(np.trace(G(n, A1, A2, B1, B2, 'cyclic')[1]@rho)))**(1/(np.floor(n-1)/2))

def der_meanG(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, scenario, p): #Partial derivative of the value of the inequality
#with respect tho the angle 'p' given, used along with another one to construct the respective observable with the function "obs(theta, phi)"
    delta = 1e-5
    A1, A2 = obs(theta_A1, phi_A1), obs(theta_A2, phi_A2)
    B1, B2 = obs(theta_B1, phi_B1), obs(theta_B2, phi_B2)
    def mean(rho, n, A1, A2, B1, B2):
        if scenario == 'star':
            return meanG(rho, n, A1, A2, B1, B2, 'star')
        elif scenario == 'chain':
            return meanG(rho, n, A1, A2, B1, B2, 'chain')
        elif scenario == 'cyclic':
            return meanG(rho, n, A1, A2, B1, B2, 'cyclic')
    if p == 'theta_A1':
        return (mean(rho, n, obs(theta_A1 + delta, phi_A1), A2, B1, B2) - mean(rho, n, obs(theta_A1 - delta, phi_A1), A2, B1, B2))/(2*delta)
    elif p == 'phi_A1':
        return (mean(rho, n, obs(theta_A1, phi_A1 + delta), A2, B1, B2) - mean(rho, n, obs(theta_A1, phi_A1 - delta), A2, B1, B2))/(2*delta)
    elif p == 'theta_A2':
        return (mean(rho, n, A1, obs(theta_A2 + delta, phi_A2), B1, B2) - mean(rho, n, A1, obs(theta_A2 - delta, phi_A2), B1, B2))/(2*delta)
    elif p == 'phi_A2':
        return (mean(rho, n, A1, obs(theta_A2, phi_A2 + delta), B1, B2) - mean(rho, n, A1, obs(theta_A2, phi_A2 - delta), B1, B2))/(2*delta)
    elif p == 'theta_B1':
        return (mean(rho, n, A1, A2, obs(theta_B1 + delta, phi_B1), B2) - mean(rho, n, A1, A2, obs(theta_B1 - delta, phi_B1), B2))/(2*delta)
    elif p == 'phi_B1':
        return (mean(rho, n, A1, A2, obs(theta_B1, phi_B1 + delta), B2) - mean(rho, n, A1, A2, obs(theta_B1, phi_B1 - delta), B2))/(2*delta)
    elif p == 'theta_B2':
        return (mean(rho, n, A1, A2, B1, obs(theta_B2 + delta, phi_B2)) - mean(rho, n, A1, A2, B1, obs(theta_B2 - delta, phi_B2)))/(2*delta)
    elif p == 'phi_B2':
        return (mean(rho, n, A1, A2, B1, obs(theta_B2, phi_B2 + delta)) - mean(rho, n, A1, A2, B1, obs(theta_B2, phi_B2 - delta)))/(2*delta)
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------FUNCTION FOR CALCULATION OF THE JACOBIAN OF INEQUALITY OPERATOR'S VALUE--------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def J(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, scenario): #Jacobian of the value of the inequality with respect
#to the angles used to construct the observables with function "obs(theta, phi)"
    def der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, p):
        if scenario == 'star':
            return der_meanG(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'star', p)
        elif scenario == 'chain':
            return der_meanG(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'chain', p)
        elif scenario == 'cyclic':
            return der_meanG(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'cyclic', p)
    J = np.empty((1, 8), float)
    if theta_A1 != 0 and theta_A1 != pi:
        J[0,0] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'theta_A1')
    else:
        J[0,0] = 0
    if phi_A1 != 0 and phi_A1 != 2*pi:
        J[0,1] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'phi_A1')
    else:
        J[0,1] = 0
    if theta_A2 != 0 and theta_A2 != pi:
        J[0,2] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'theta_A2')
    else:
        J[0,2] = 0
    if phi_A2 != 0 and phi_A2 != 2*pi:
        J[0,3] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'phi_A2')
    else:
        J[0,3] = 0
    if theta_B1 != 0 and theta_B1 != pi:
        J[0,4] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'theta_B1')
    else:
        J[0,4] = 0
    if phi_B1 != 0 and phi_B1 != 2*pi:
        J[0,5] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'phi_B1')
    else:
        J[0,5] = 0
    if theta_B2 != 0 and theta_B2 != pi:
        J[0,6] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'theta_B2')
    else:
        J[0,6] = 0
    if phi_B2 != 0 and phi_B2 != 2*pi:
        J[0,7] = der(rho, n, theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2, 'phi_B2')
    else:
        J[0,7] = 0
    return J

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------NEWTON METHOD TO MAXIMIZE INEQUALITY'S VALUE------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def see_saw_newton(n, state, scenario, acc_new = 1e-4): #Newton method for maximization of inequality's value for 'n' parties sharing resource 'state'
#in the specified 'scenario' with accuracy 'acc_new' for the method

    rho = multikron(int(n/2), state) #State shared in the network

    gamma = 0.02 #Parameter to control steps for the angles in the maximization process
    delta_new = 1e-5 #Paremeter to make angles defining A1 and A2 different in case the method converges for respectively equal ones

    theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2 = np.random.uniform(acc_new, pi - acc_new), np.random.uniform(acc_new, 2*pi - acc_new), np.random.uniform(acc_new, pi - acc_new), np.random.uniform(acc_new, 2*pi - acc_new), np.random.uniform(acc_new, pi - acc_new), np.random.uniform(acc_new, 2*pi - acc_new), np.random.uniform(acc_new, pi - acc_new), np.random.uniform(acc_new, 2*pi - acc_new)
    A1, A2, B1, B2 = obs(theta_A1, phi_A1), obs(theta_A2, phi_A2), obs(theta_B1, phi_B1), obs(theta_B2, phi_B2) #Initializing with random observables

    x = np.array([[theta_A1, phi_A1, theta_A2, phi_A2, theta_B1, phi_B1, theta_B2, phi_B2]]) #Vector of angles
    dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #Initializing vector for changing the angles
    S = -1.0

    t1 = time()
    t2 = time()
    dt = t2 - t1 #Initializing time interval to limit the processing time

    while la.norm(dx) > acc_new and S <= 0 and dt < (60.0*4**(n - 3)): #Stop when angles differ from a value less than accuracy specified or violation is not
    #stablished or processing time is not too large (depending on the number of parties in the network)
        dx = J(rho, n, x[0,0], x[0,1], x[0,2], x[0,3], x[0,4], x[0,5], x[0,6], x[0,7], scenario)
        x = x + gamma*dx
        if (x[0,0] == x[0,2] and x[0,1] == x[0,3]) or (x[0,4] == x[0,6] and x[0,5] == x[0,7]):
            x[0,2] += delta_new
            x[0,4] += delta_new
        A1, A2, B1, B2 = obs(x[0,0], x[0,1]), obs(x[0,2], x[0,3]), obs(x[0,4], x[0,5]), obs(x[0,6], x[0,7])
        S = meanG(rho, n, A1, A2, B1, B2, scenario) - 1.0
        gamma = np.exp(-la.norm(dx))
        t2 = time()
        dt = t2 - t1
    else:
        return S
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------BISSECTION METHOD TO SEARCH FOR THE MINIMUM NOISE FOR WHICH RESOURCE IS NONLOCAL IN THE NETWORK--------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def bis(n, scenario, ent1, ent2, noise, acc_new = 1e-4, acc_bis = 1e-4): #Bissection method, 'n' parties in the network, specified 'scenario', 'ent1' and 'ent2'
#entangled states for consecutive parties sharing resulting state [(w*ent1 + (1-w)*noise) tensor (w*ent2 + (1-w)*noise] for given 'noise' state. 'acc_bis'
#accuracy for parameter w calculated by bissection method and acc_new for newton method

    w0 = 0.0
    w1 = 1.0
    whalf = (w0 + w1)/2
    esthalf = np.kron(whalf*ent1 + (1 - whalf)*noise,  whalf*ent2 + (1 - whalf)*noise)
    S0 = -1.0
    Shalf = see_saw_newton(n, esthalf, scenario)

    rep = int(np.ceil(np.log2(1/acc_bis) - 1))

    for i in tqdm(range(rep)):
        if S0*Shalf > 0:
            w0 = whalf
            whalf = (w0 + w1)/2
            esthalf = np.kron(whalf*ent1 + (1 - whalf)*noise,  whalf*ent2 + (1 - whalf)*noise)
            S0 = Shalf
            Shalf = see_saw_newton(n, esthalf, scenario)
        else:
            w1 = whalf
            whalf = (w0 + w1)/2
            esthalf = np.kron(whalf*ent1 + (1 - whalf)*noise,  whalf*ent2 + (1 - whalf)*noise)
            Shalf = see_saw_newton(n, esthalf, scenario)
    
    return w1

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------FUNCTION FOR USER'S INPUTS AND FINAL OUTPUT---------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def w_nonlocal_network(n, scenario, ent1, ent2, noise, acc_new = 1e-4, acc_bis = 1e-4):
    w = bis(n, scenario, ent1, ent2, noise, acc_new = 1e-4, acc_bis = 1e-4)
    print('The state is nonlocal in the newtwork with', str(n), 'resources and scenario', scenario, 'for all w >', str(w))