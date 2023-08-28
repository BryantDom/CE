import numpy as np


def SSD(states, filtered_signal):
    ssd = 0
    for i in range(states.size):
        ssd = ssd + (states[i] - filtered_signal[i]) ** 2
    return ssd

def MAD(states, filtered_signal):
    mad = abs(states[0] - filtered_signal[0])
    for i in range(states.size):
        temp =  abs(states[i] - filtered_signal[i])
        mad = max(mad,temp)
    return mad

def PRD(states,ssd):
    mean = sum(states)/len(states)
    cleanssd = 0
    for i in range(states.size):
        cleanssd = cleanssd + (states[i] - mean) * (states[i] - mean)
    prd = (ssd/cleanssd) ** (1/2)
    return prd

def CosineSim(states, filtered_signal):
    cosineSim = states.dot(filtered_signal) / (np.linalg.norm(states) * np.linalg.norm(filtered_signal))
    return cosineSim