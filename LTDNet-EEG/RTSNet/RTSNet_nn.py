"""# **Class: RTSNet**"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import tvm
from torch.nn import init
from tvm import relay
from RTSNet.KalmanNet_nn import KalmanNetNN
from RTSNet.forward_kg import forward_kg
from RTSNet.backward_kt import backward_kt
from SystemModels.LinearizationProcessing.TaylorPrior import TaylorPrior
from scipy.special import factorial
from torch.nn.functional import pad

class RTSNetNN(KalmanNetNN):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.count=0
        self.isRelay=False


    #############
    ### Build ###
    ############
    def NNBuild(self, ssModel, args):

        # self.derivative_coefficients = ssModel.derivative_coefficients
        # self.basis_functions = ssModel.basis_functions
        # 初始化状态矩阵和转移矩阵
        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n)

        # 前向传递初始化卡尔曼增益
        self.kg = forward_kg()

        # 后向传递初始化卡尔曼增益
        self.kt = backward_kt()

        # 模型计算量和参数量
        self.kg.computeComplexity_forward()
        self.kt.computeComplexity_backward()


    def Taylor_Linear(self,data):
        data = data.permute(0, 2, 1)
        taylor_order = 7
        basis_functions = np.array([[(1/512) ** k / factorial(k)] for k in range(1, taylor_order + 1)])
        factorial_functions =basis_functions = torch.from_numpy(basis_functions).float()
        basis_functions = basis_functions.reshape((1, 1, -1))
        basis_functions = basis_functions.repeat((taylor_order, 10, 1))
        derivative_coefficients = torch.zeros(512, taylor_order, 2)
        half_window_size = int(7/ 2)
        lower_bound_index = - half_window_size
        upper_bound_index = half_window_size + 1
        padded_data = pad(data, (0, 0, -lower_bound_index, upper_bound_index), 'replicate')
        weights = torch.from_numpy(np.array([1 for _ in range(7)]))
        for t in range(512):
            current_state = padded_data[:,
                            t - lower_bound_index - half_window_size:t + half_window_size - lower_bound_index + 1]
            observations = padded_data[:,
                           t - lower_bound_index - half_window_size + 1:t + half_window_size - lower_bound_index + 2]
            target_tensor = (observations - current_state).reshape(7, 10, -1)
            covariance = torch.bmm(basis_functions.mT, basis_functions)
            cross_correlation = torch.bmm(basis_functions.mT, target_tensor)
            weights = weights.reshape(-1, 1, 1)
            weighted_covariance = (weights * covariance).sum(0)
            weighted_cross_correlation = (weights * cross_correlation).sum(0)
            derivatives_t = torch.mm(torch.linalg.pinv(weighted_covariance), weighted_cross_correlation)
            derivative_coefficients[t] = derivatives_t
        self.derivative_coefficients = derivative_coefficients
        self.basis_functions = factorial_functions


    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward(self, filter_x):
        # x_t+1|T
        self.s_m1x_nexttime = torch.squeeze(filter_x)

    ##############################
    ### Innovation Computation ###
    ##############################
    def S_Innovation(self, filter_x):
        # self.filter_x_prior = self.f(filter_x)
        # 用510去预测先验的510
        self.filter_x_prior = torch.squeeze(self.f(filter_x.reshape(2,1))+ torch.mm(self.derivative_coefficients[self.count].T, self.basis_functions))

        # 511-510
        self.dx = self.s_m1x_nexttime - self.filter_x_prior


    def relayBuild_forward(self):
        t1 = torch.tensor([0.2356, 0.2356])
        t2 = torch.tensor([0.2356, 0.2356])
        scripted_model = torch.jit.trace(self.kg, [t1, t2])
        input_name = "input"
        shape_list = [(input_name, t1.shape), (input_name, t2.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        target = tvm.target.Target("llvm", host="llvm")
        with tvm.transform.PassContext(opt_level=2):
            lib = relay.build(mod, target, params=params)
            improvemod = tvm.relay.transform.InferType()(mod)

        from tvm.contrib import graph_executor

        dtype = "float32"
        dev = tvm.cpu(0)
        self.M_forward = graph_executor.GraphModule(lib["default"](dev))

    def relayBuild_backword(self):
        t1 = torch.tensor([0.2356, 0.2356])
        t2 = torch.tensor([0.2356, 0.2356])
        scripted_model = torch.jit.trace(self.kt, [t1, t2], check_trace=False)
        input_name = "input"
        shape_list = [(input_name, t1.shape), (input_name, t2.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        target = tvm.target.Target("llvm", host="llvm")

        with tvm.transform.PassContext(opt_level=2):
            lib = relay.build(mod, target, params=params)
            improvemod = tvm.relay.transform.InferType()(mod)

        from tvm.contrib import graph_executor

        dtype = "float32"
        dev = tvm.cpu(0)
        self.M_backward = graph_executor.GraphModule(lib["default"](dev))

    def relayBuild(self):
        if self.isRelay==True:
            self.relayBuild_forward()
            self.relayBuild_backword()
        else:
            pass


    ################################
    ### Smoother Gain Estimation ###
    ################################
    def step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2):

        # B2 Reshape and Normalize Delta tilde x_t+1 = x_t+1|T - x_t+1|t+1
        # filter_x_nexttime = x_t+1|t+1
        dm1x_tilde = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_tilde_reshape = torch.squeeze(dm1x_tilde)
        bw_innov_diff = func.normalize(dm1x_tilde_reshape, p=2, dim=0, eps=1e-12, out=None)  # b2前向后向差异

        # B1 Feature 7:  x_t+1|T - x_t+1|t      B1更新差异
        dm1x_f7 = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_f7_reshape = torch.squeeze(dm1x_f7)
        bw_update_diff = func.normalize(dm1x_f7_reshape, p=2, dim=0, eps=1e-12, out=None)  # B1更新差异

        # ********************上述把B1、B2、B3定义清楚,作为输入
        # Smoother Gain Network Step
        if(self.isRelay == True):
            t1 = bw_innov_diff
            t2 = bw_update_diff
            input_name = "input"
            # Set inputs
            self.M_backward.set_input(input_name, tvm.nd.array(t1))
            self.M_backward.set_input(input_name, tvm.nd.array(t2))
            # Execute
            self.M_backward.run()
            # Get outputs
            tvm_output = self.M_backward.get_output(0)

            # 将 NumPy 数组转换为 PyTorch 张量
            numpy_array = tvm_output.asnumpy()
            tvm_output = torch.from_numpy(numpy_array)
            SG = tvm_output
        else:
            SG = self.kt(bw_innov_diff, bw_update_diff)
        # SG = self.RTSGain_step(bw_innov_diff, bw_evol_diff, bw_update_diff)

        # Reshape Smoother Gain to a Matrix
        self.SGain = torch.reshape(SG, (self.m, self.m))

    ####################
    ### RTS Net Step ###
    ####################
    def RTSNet_step(self, filter_x, filter_x_nexttime, smoother_x_tplus2):
        # 开始反向: filter_x:510,filter_x_nexttime:511,smoother_x_tplus2=none
        # filter_x = torch.squeeze(filter_x)
        # filter_x_nexttime = torch.squeeze(filter_x_nexttime)
        # smoother_x_tplus2 = torch.squeeze(smoother_x_tplus2)
        # Compute Innovation
        self.S_Innovation(filter_x)

        # Compute Smoother Gain
        # 计算获得反向卡尔曼增益
        self.step_RTSGain_est(filter_x_nexttime, smoother_x_tplus2)

        # Compute the 1-st posterior moment
        # self.s_m1x_nexttime 返回向后传递,修正之后的X
        INOV = torch.matmul(self.SGain, self.dx)
        self.s_m1x_nexttime = filter_x + INOV

        # return 返回向后传递,修正之后的X
        return torch.squeeze(self.s_m1x_nexttime)

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, filter_x, filter_x_nexttime, smoother_x_tplus2):
        if yt is None:
            # BW pass
            return self.RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
        # 开始 filter_x:510,filter_x_nexttime:511
        else:
            # FW pass
            return self.KNet_step(yt)
    
    #########################
    ### Init Hidden State ###
    #########################
    # 初始化5个参数: 3个前向和2两个后向
    def init_hidden(self):
        pass
        # ### FW GRUs
        # weight = next(self.parameters()).datasets
        # hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        #
        # self.h_S = hidden.datasets
        # self.h_S[0, 0, :] = self.prior_S.flatten()
        #
        # hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        # self.h_Sigma = hidden.datasets
        # self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        #
        # hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        # self.h_Q = hidden.datasets
        # self.h_Q[0, 0, :] = self.prior_Q.flatten()
        #
        # ### BW GRUs
        # weight = next(self.parameters()).datasets
        # hidden = weight.new(1, self.batch_size, self.d_hidden_Q_bw).zero_()
        # self.h_Q_bw = hidden.datasets
        # self.h_Q_bw[0, 0, :] = self.prior_Q.flatten()
        #
        # hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma_bw).zero_()
        # self.h_Sigma_bw = hidden.datasets
        # self.h_Sigma_bw[0, 0, :] = self.prior_Sigma.flatten()
        #
        # # self.h_Q=torch.tensor([[[0.0473, 0.0150,0.0150, 0.0473]]])
        # self.h_Q_bw=torch.tensor([[[0.0473, 0.0150,0.0150, 0.0473]]])
        # # self.h_Q_bw = torch.randn(self.seq_len_input, self.batch_size, 4)
        #
