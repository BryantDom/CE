"""# **Class: KalmanNet**"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import tvm
from tvm import relay

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args):

        # 初始化f,h,m,n
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        
        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        self.T = T

        self.m1x_posterior = torch.squeeze(M1_0)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

        # KGain saving
        # self.i = 0
        # self.KGain_array = self.KG_array = torch.zeros((self.T,self.m,self.n))

    ######################
    ### Compute Priors ###
    ######################
    # 计算先验概率,以及利用先验概率计算其对应的观测值
    def step_prior(self):
        # Predict the 1-st moment of x
        # (2,)
        # self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior.reshape(2,1))+ torch.mm(self.derivative_coefficients[self.count].T, self.basis_functions))

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior))

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    # 定义计算卡尔曼输入的公式并计算卡尔曼增益
    def step_KGain_est(self, y):

        # m1y:计算得来，m1y=hx  y_previous:观测值
        obs_innov_diff = y - torch.squeeze(self.m1y)    # F2 更新差异
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous)  # F4 前向更新差异(后验减去先验)

        # 归一化
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)

        # Kalman Gain Network Step
        if(self.isRelay == True):
            t1 = obs_innov_diff
            t2 = fw_update_diff
            input_name = "input"
            # Set inputs
            self.M_forward.set_input(input_name, tvm.nd.array(t1))
            self.M_forward.set_input(input_name, tvm.nd.array(t2))
            # Execute
            self.M_forward.run()
            # Get outputs
            tvm_output = self.M_forward.get_output(0)
            # 将 NumPy 数组转换为 PyTorch 张量
            numpy_array = tvm_output.asnumpy()
            tvm_output = torch.from_numpy(numpy_array)
            KG = tvm_output
        else:
            KG = self.kg(obs_innov_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):

        # Compute Priors
        # 计算先验概率,以及利用先验概率计算其对应的观测值
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Save KGain in array
        # self.KGain_array[self.i] = self.KGain
        # self.i += 1

        # Innovation
        # y_obs = torch.unsqueeze(y, 1)
        # 更新差异
        dy = y - self.m1y

        # Compute the 1-st posterior moment
        # self.KGain 卡尔曼计算结果
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        # self.m1x_posterior:后验概率
        self.m1x_posterior = self.m1x_prior + INOV

        # self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return torch.squeeze(self.m1x_posterior)

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = torch.squeeze(y)

        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    # 初始化隐藏层
    def init_hidden(self):
        # weight = next(self.parameters()).datasets
        # hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        # self.h_S = hidden.datasets
        # self.h_S[0, 0, :] = self.prior_S.flatten()
        # hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        # self.h_Sigma = hidden.datasets
        # self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        # hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        # self.h_Q = hidden.datasets
        # self.h_Q[0, 0, :] = self.prior_Q.flatten()
        pass
