"""# **Class: System Model for Linear Cases**

1 Store system model parameters: 
    state transition matrix F, 
    observation matrix H, 
    process noise covariance matrix Q, 
    observation noise covariance matrix R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test, etc.

2 Generate dataset for linear cases
"""

import torch

class SystemModel:

    def __init__(self, F, Q, H, R):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]
        self.R = R

        ################
        ### Taylor ###
        ################
        # self.derivative_coefficients = derivative_coefficients
        # self.basis_functions = basis_functions

    def f(self, x):
        a=torch.matmul(self.F, x)
        # b=(torch.matmul(self.F, x.reshape(2,1))+torch.mm(self.derivative_coefficients[self.count].T, self.basis_functions))
        # b=b.reshape(2)
        return a
    
    def h(self, x):
        return torch.matmul(self.H, x)
        
    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q
        self.R = R


