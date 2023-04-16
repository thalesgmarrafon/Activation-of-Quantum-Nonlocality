from math import *
import numpy as np
import picos as pc
from tqdm.auto import tqdm
import SM as sm

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------FUNCTION FOR GENERATING RANDOM OPERATOR-----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def rand_op():
    #Creation of random operators from random angles in te Bloch sphere defining the dichotomous projective measurements
    theta = np.random.uniform(0, pi)
    phi = np.random.uniform(0, 2*pi)
    state_rand = pc.Constant("state_rand", [cos(theta/2), sin(theta/2)*e**((phi)*1j)], (2, 1))
    #state = np.array([[cos(theta/2)], [sin(theta/2)*e**((phi)*1j)]])
    op = 2*state_rand*(state_rand.H) - sm.Ip
    
    return op

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------FUNCTION FOR DEFINING INEQUALITIES OPERATORS--------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def G_ineq(ineq, A_list, B_list): #Operator of inequality 'ineq' defined from operators 'A_list' of Alice and 'B_list' of Bob
    B1 = B_list[0]
    print(B1)
    if ineq == 'CHSH':
        G = A_list[0]@B_list[0] + A_list[0]@B_list[1] + A_list[1]@B_list[0] - A_list[1]@B_list[1]
    elif ineq == 'I3322':
        G = -A_list[0]@sm.I - A_list[1]@sm.I - sm.I@B_list[0] - sm.I@B_list[1] - A_list[0]@B_list[0] - A_list[1]@B_list[0] - A_list[2]@B_list[0] - A_list[0]@B_list[1] - A_list[1]@B_list[1] + A_list[2]@B_list[1] - A_list[0]@B_list[2] + A_list[1]@B_list[2]

    return G

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------FUNCTION FOR SEE-SAW---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def see_saw(state, ineq, rep): #Maximize operator of inequality 'ineq' over given 'state', with a number 'rep' of iterations in the optimization
    B1, B2, B3 = rand_op(), rand_op(), rand_op()
    for i in tqdm(range(rep), desc = 'see-saw loop', ncols = 150, leave = False):
        #Optimization over Alice's opertors
        PA = pc.Problem()
        Ax = PA.add_variable('Ax', (2,2), 'hermitian')
        Ay = PA.add_variable('Ay', (2,2), 'hermitian')
        Az = PA.add_variable('Az', (2,2), 'hermitian')
        PA.add_constraint((Ax + sm.Ip) >> 0)
        PA.add_constraint((Ay + sm.Ip) >> 0)
        PA.add_constraint((Az + sm.Ip) >> 0)
        PA.add_constraint((sm.Ip - Ax) >> 0)
        PA.add_constraint((sm.Ip - Ay) >> 0)
        PA.add_constraint((sm.Ip - Az) >> 0)
        #G = G_ineq(ineq, A_list, Blist)
        if ineq == 'CHSH':
            G = Ax@B1 + Ax@B2 + Ay@B1 - Ay@B2
        elif ineq == 'I3322':
            G = -Ax@sm.Ip - Ay@sm.Ip - sm.Ip@B1 - sm.Ip@B2 - Ax@B1 - Ay@B1 - Az@B1 - Ax@B2 - Ay@B2 + Az@B2 - Ax@B3 + Ay@B3
        PA.set_objective('max', pc.trace(state*G))
        PA.solve(solver = 'mosek')
        A1 = Ax.value
        A2 = Ay.value
        A3 = Az.value
        #Optimization over Bob's operators
        PB = pc.Problem()
        Bx = PB.add_variable('Bx', (2,2), 'hermitian')
        By = PB.add_variable('By', (2,2), 'hermitian')
        Bz = PB.add_variable('Bz', (2,2), 'hermitian')
        PB.add_constraint((Bx + sm.Ip) >> 0)
        PB.add_constraint((By + sm.Ip) >> 0)
        PB.add_constraint((Bz + sm.Ip) >> 0)
        PB.add_constraint((sm.Ip - Bx) >> 0)
        PB.add_constraint((sm.Ip - By) >> 0)
        PB.add_constraint((sm.Ip - Bz) >> 0)
        #G = G_ineq(ineq, A_list, B_list)
        if ineq == 'CHSH':
            G = A1@Bx + A1@By + A2@Bx - A2@By
        elif ineq == 'I3322':
            G = -A1@sm.Ip - A2@sm.Ip - sm.Ip@Bx - sm.Ip@By - A1@Bx - A2@Bx - A3@Bx - A1@By - A2@By + A3@By - A1@Bz + A2@Bz
        PB.set_objective('max', pc.trace(state*G))
        PB.solve(solver = 'mosek')
        if ineq == 'CHSH':
            sol_B = PB.value - 2.0
        elif ineq == 'I3322':
            sol_B = PB.value - 4.0
        B1 = Bx.value
        B2 = By.value
        B3 = Bz.value

    return sol_B

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------FUNCTION FOR BISSECTION METHOD----------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def bis(ent, noise, ineq, rep = 100, acc_bis = 1e-4):
    #Bissection method for finding minimum w for which the state w*'ent' + (1 - w)*'noise' is nonlocal for inequality 'ineq',
    #with accuracy 'acc_bis' and 'rep' number of repetitions for see-saw
    w0 = 0.0
    w1 = 1.0
    whalf = (w0 + w1)/2
    statehalf = whalf*ent + (1 - whalf)*noise
    S0 = -1.0
    Shalf = see_saw(statehalf, ineq, rep)

    rep = int(np.ceil(np.log2(1/acc_bis) - 1))

    for j in tqdm(range(rep), desc = 'bissection loop', ncols = 150, leave = False):
        if S0*Shalf > 0:
            w0 = whalf
            whalf = (w0 + w1)/2
            statehalf = whalf*ent + (1 - whalf)*noise
            S0 = Shalf
            Shalf = see_saw(statehalf, ineq, rep)
        else:
            w1 = whalf
            whalf = (w0 + w1)/2
            statehalf = whalf*ent + (1 - whalf)*noise
            Shalf = see_saw(statehalf, ineq, rep)
            
    return w1

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------FUNCTION TO REPEAT SEE-SAW SEVERAL TIMES AND GIVE SMALLEST VALUE OF W FOUND------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def W(ent, noise, ineq, rep = 100, acc_bis = 1e-4, rep_W = 20):
    W = []
    for i in tqdm(range(rep_W), desc = 'repetition loop', ncols = 150):
        W.append(bis(ent, noise, ineq, rep, acc_bis))
    w = np.amin(W)
    print(f'The state is nonlocal for all w > {w}')

W(sm.rhopsim, sm.rho_00, 'I3322')