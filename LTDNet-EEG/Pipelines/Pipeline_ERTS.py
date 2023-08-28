"""
This file contains the class Pipeline_ERTS, 
which is used to train and test RTSNet in both linear and non-linear cases.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import random
import datetime

logs = open('Results/mse.txt', mode='w', encoding='utf-8')
train_mse=[]
val_mse=[]
test_mse=[]
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Pipeline_ERTS:

    def __init__(self):
        super().__init__()

    def setModel(self,args, ssModel, model,alpha = 0.5):
        self.SysModel = ssModel
        self.model = model
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch # Number of Samples in Batch
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
        self.alpha = alpha # Composition loss factor 成分损失系数
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, args,SysModel, cv_input, cv_target, train_input, train_target, path_results):

        # train_input = Tensor:(10,1,512) self.N_E:10
        self.N_E = len(train_input)
        # cv_input = Tensor:(10,1,512)  self.N_CV:10
        self.N_CV = len(cv_input)

        # MSE_cv_linear_batch = Tensor:(10)
        MSE_cv_linear_batch = torch.empty([self.N_CV])
        # self.MSE_cv_linear_epoch = Tensor:(2000)
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        # self.MSE_cv_dB_epoch = Tensor:(2000)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        # MSE_train_linear_batch = Tensor:(10)
        MSE_train_linear_batch = torch.empty([self.N_B])
        # 下面两个 = Tensor:(2000)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        self.cv_out_list = []

        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # self.model = torch.quantization.quantize_dynamic(
        #     self.model, {nn.Linear}, dtype=torch.qint8
        # )
        # self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # torch.quantization.prepare(self.model, inplace=True)

        for ti in range(0, self.N_steps):
            starttime = datetime.datetime.now()
            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode

            self.model.train()

            # 边训练边优化
            # if args.isOptimize ==True:
            #     if ti ==1:
            #         self.model.isRelay = True
            #     self.model.relayBuild()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            taylor_train = train_input.repeat(1, 2, 1)
            self.model.Taylor_Linear(taylor_train)

            for j in range(0, self.N_B):

                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e]

                # SysModel.T :形状512
                SysModel.T = y_training.size()[-1]

                # x_out_training_forward = Tensor:(2,512)
                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                
                for t in range(0, SysModel.T):    # 0-511
                    self.model.count=t
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                x_out_training[:, SysModel.T-1] = x_out_training_forward[:, SysModel.T-1]  # backward smoothing starts from x_T|T   511
                # 反向操作
                self.model.InitBackward(x_out_training[:, SysModel.T-1])    # 初始化后向传播   赋值511
                x_out_training[:, SysModel.T-2] = self.model(None, x_out_training_forward[:, SysModel.T-2], x_out_training_forward[:, SysModel.T-1],None)  # 预测出510
                for t in range(SysModel.T-3, -1, -1): # t=509开始到0
                    self.model.count = t
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t], x_out_training_forward[:, t+1],x_out_training[:, t+2])

                # Compute Training Loss
                LOSS = 0
                LOSS = self.loss_fn(x_out_training, train_target[n_e].repeat((2, 1)))
                # 存储每一次计算得出的LOSS
                MSE_train_linear_batch[j] = LOSS.item()

                # 将每一次的LOSS进行求和
                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])


            ##################
            ### Optimizing ###
            ##################
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.requires_grad_(True)
            Batch_Optimizing_LOSS_mean.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################
            # Cross Validation Mode
            self.model.eval()
            taylor_cv = cv_input.repeat(1, 2, 1)
            self.model.Taylor_Linear(taylor_cv)

            with torch.no_grad():
                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)

                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
 
                    for t in range(0, SysModel.T_test):
                        self.model.count = t
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                    x_out_cv[:, SysModel.T_test-1] = x_out_cv_forward[:, SysModel.T_test-1] # backward smoothing starts from x_T|T
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test-1]) 
                    x_out_cv[:, SysModel.T_test-2] = self.model(None, x_out_cv_forward[:, SysModel.T_test-2], x_out_cv_forward[:, SysModel.T_test-1],None)
                    for t in range(SysModel.T_test-3, -1, -1):
                        self.model.count = t
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t+1],x_out_cv[:, t+2])                       

                    if(j==0):
                        cv_out=x_out_cv

                    # Compute CV Loss
                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j].repeat((2, 1))).item()

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    # torch.save(self.model, path_results + 'model.pt')

                    model_dict = self.model.state_dict()
                    for key in model_dict:
                        model_dict[key] = model_dict[key].numpy()
                    torch.save(model_dict,path_results + 'model.pt')


            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
            
            
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

            train_mse.append(self.MSE_train_dB_epoch[ti])
            val_mse.append(self.MSE_cv_dB_epoch[ti])
            endtime = datetime.datetime.now()
            print (endtime - starttime)

        print('train_mse:',[np.array(train_mse)],file=logs)
        print('val_mse:',[np.array(val_mse)],file=logs)

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch,cv_out,cv_target]

    def NNTest(self,args, SysModel, test_input, test_target, path_results ):

        self.N_T = len(test_input)

        self.MSE_test_linear_arr = torch.empty([self.N_T])
        test_denoise = torch.zeros([5120])
        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        static_dict = torch.load(path_results + 'model.pt')
        for key in static_dict:
            static_dict[key] = torch.from_numpy(static_dict[key])
        self.model.load_state_dict(static_dict)

        # self.model = torch.quantization.quantize_dynamic(
        #     self.model, {nn.Linear}, dtype=torch.qint8
        # )

        self.model.eval()

        if args.isOptimize ==True:
            self.model.isRelay = True
        self.model.relayBuild()

        torch.no_grad()

        # x_out_list = []
        start = time.time()

        taylor_test = test_input.repeat(1, 2, 1)
        self.model.Taylor_Linear(taylor_test)

        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j]
            SysModel.T_test = y_mdl_tst.size()[-1]

            x_out_test_forward_1 = torch.empty(SysModel.m,SysModel.T_test)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test)

            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
           
            for t in range(0, SysModel.T_test):
                self.model.count = t
                x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
            x_out_test[:, SysModel.T_test-1] = x_out_test_forward_1[:, SysModel.T_test-1] # backward smoothing starts from x_T|T 
            self.model.InitBackward(x_out_test[:, SysModel.T_test-1]) 
            x_out_test[:, SysModel.T_test-2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test-2], x_out_test_forward_1[:, SysModel.T_test-1],None)
            for t in range(SysModel.T_test-3, -1, -1):
                self.model.count = t
                x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t+1],x_out_test[:, t+2])

            test_denoise[j * 512:(j + 1) * 512] = 0.5 * x_out_test[0, :] + 0.5 * x_out_test[1, :]

            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j].repeat((2, 1))).item()

            # x_out_list = x_out_test.unsqueeze(0).repeat(10, 1, 1)
        
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")

        print(str, self.MSE_test_dB_avg, "[dB]",file=logs)

        str = "STD Test:"
        print(str, self.test_std_dB, "[dB]")

        print(str, self.test_std_dB, "[dB]",file=logs)

        # Print Run Time
        print("Inference Time:", t)
        logs.close()

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, test_denoise, t]

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)