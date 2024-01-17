# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:54:33 2024

@author: Chia-Tso, Lai
"""


import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


data2 = {"qubits":3,"h":[0],"cx":[(1,2)]}


def statevector(states_list):
    
    output = states_list[0]
    for i in range(1,len(states_list)):
        output = np.kron(output,states_list[i])
        
    return output


def unitary(unitary_list):
    
    output = unitary_list[0]
    for i in range(1,len(unitary_list)):
        output = np.kron(output,unitary_list[i])
        
    return output


#turn statevector into density matrix
def density_matrix(vector):
    d = len(vector)
    density = vector.reshape(d,1)@vector.reshape(1,d)
    return density

#noise model
def noise(density,mu,q):
    return (1-mu)*density+(mu/2**q)*np.identity(2**q)


def simulator(data,mu):
    
    
    q = data["qubits"]
    h_list = data["h"]
    cx_list = data["cx"]
    
    d = 2**q
    
    zero_state = np.array([1,0])
    one_state = np.array([0,1])
    
    ref_state = (1/2**0.5)*(statevector([zero_state]*q)+statevector([one_state]*q))
    
    initial = statevector([zero_state]*q)
    initial = density_matrix(initial)
    
    I = np.identity(2)
    h = (1/2**0.5)*np.array([[1,1],[1,-1]])
    x = np.array([[0,1],[1,0]])
    
    unitary_list = [I]*q
    for i in h_list:
        unitary_list[i] = h 
        
    H = unitary(unitary_list)
    
    state = H@initial@H
    state = noise(state,mu,q)
    
    cx_seq = []
    for pair in cx_list:
        control_list = [I]*q
        target_list = [I]*q
        
        control_list[pair[0]] = zero_state.reshape(2,1)@zero_state.reshape(1,2)
        target_list[pair[0]] = one_state.reshape(2,1)@one_state.reshape(1,2)
        target_list[pair[1]] = x
        
        cx_gate = unitary(control_list)+unitary(target_list)
        cx_seq.append(cx_gate)
    
    for gate in cx_seq:
        state = gate@state@gate
        state = noise(state,mu,q)
        
    fidelity = ref_state.reshape(1,d)@state@ref_state.reshape(d,1)
    fidelity = fidelity[0][0]
    
    #compute von Neumann entropy
    e, v = linalg.eig(state)
    non_zero_e = e[e>1e-5]
    entropy= -np.sum(non_zero_e * np.log(non_zero_e))


    
    return [fidelity,entropy]



fidelity = [simulator(data2,mu)[0] for mu in np.arange(0,1,0.05)]
entropy = [simulator(data2,mu)[1] for mu in np.arange(0,1,0.05)]


print("fidelity:",fidelity)
print("entropy:",entropy)


plt.subplot(1,2,1)
plt.plot(np.arange(0,1,0.05),fidelity)
plt.xlabel("μ")
plt.ylabel("F")

plt.subplot(1,2,2)
plt.plot(np.arange(0,1,0.05),entropy)
plt.xlabel("μ")
plt.ylabel("S")

plt.tight_layout()
