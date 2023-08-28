import torch
from math import pi
m = 2 # state dimension = 2, 5, 10, etc.
n = 2 # observation dimension = 2, 5, 10, etc.

##################################
### Initial state and variance ###
##################################
m1_0 = torch.zeros(m, 1)
m2_0 = 0 * 0 * torch.eye(m)

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
# F in canonical form
F = torch.eye(m)
H = torch.eye(2)

# Noise variance takes the form of a diagonal matrix
Q = torch.eye(m)
R = torch.eye(n)