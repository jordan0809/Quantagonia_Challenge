# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:48:38 2024

@author: Chia-Tso, Lai
"""



import numpy as np



#Sample data
data = {"qubits":4,"h":[0],"cx":[(0,1),(1,2),(2,3)]}


#create statevector from a list of single-qubit states
def statevector(states_list):
    
    output = states_list[0]
    for i in range(1,len(states_list)):
        output = np.kron(output,states_list[i])
        
    return output

#create a multi-qubit unitary gate from a list of single-qubit gate
def unitary(unitary_list):
    
    output = unitary_list[0]
    for i in range(1,len(unitary_list)):
        output = np.kron(output,unitary_list[i])
        
    return output


def simulator(data):
    
    states = []
    
    q = data["qubits"]
    h_list = data["h"]
    cx_list = data["cx"]
    
    zero_state = np.array([1,0])
    one_state = np.array([0,1])
    initial = statevector([zero_state]*q)
    
    I = np.identity(2)
    h = (1/2**0.5)*np.array([[1,1],[1,-1]])
    x = np.array([[0,1],[1,0]])
    
    unitary_list = [I]*q
    for i in h_list:
        unitary_list[i] = h 
        
    H = unitary(unitary_list)
    
    state = H@initial
    states.append(state)
    
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
        state = gate@state
        states.append(state)
    
    for step,s in enumerate(states):
        print(f"step{step+1}:",s)