import torch
import torch.jit
import torch.nn
import torch.nn as nn
import torch.onnx
import torchsummary
from thop import profile
from RTSNet.lightWeight.DSCNet import DSCNet



class backward_kt(torch.nn.Module):
    def __init__(self):
        super().__init__()


        # BW GRU to track Sigma
        self.Vgg16 = DSCNet(28, 4)

        # BW Fully connected 1
        self.d_input_FC1_bw = 4
        self.d_output_FC1_bw = 4
        self.d_hidden_FC1_bw = 160
        self.FC1_bw = nn.Sequential(
            nn.Linear(self.d_input_FC1_bw, self.d_hidden_FC1_bw),
            nn.LeakyReLU(),
            nn.Linear(self.d_hidden_FC1_bw, self.d_output_FC1_bw))

        # BW Fully connected 2
        self.d_input_FC2_bw = 8
        self.d_output_FC2_bw = 4
        self.FC2_bw = nn.Sequential(
            nn.Linear(self.d_input_FC2_bw, self.d_output_FC2_bw),
            nn.LeakyReLU())

        # BW Fully connected 4
        self.d_input_FC4_bw = 4
        self.d_output_FC4_bw = 24
        self.FC4_bw = nn.Sequential(
            nn.Linear(self.d_input_FC4_bw, self.d_output_FC4_bw),
            nn.LeakyReLU())

    def init_hidden(self):
        self.prior_Q = torch.tensor([[[1000.0, 955.0, 73.0, 0.0]]])
        self.prior_R = torch.tensor([[[1000.0, 955.0, 73.0, 0.0]]])
        # self.prior_Q = torch.tensor([[[0.0473, 0.0150, 0.0150, 0.0473]]])
        # self.prior_R = torch.tensor([[[0.0390, 0.0148,0.0148, 0.0390]]])
        self.h_Sigma_bw = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])

    def forward(self, bw_innov_diff, bw_update_diff):
        self.init_hidden()
        bw_innov_diff = bw_innov_diff.view(1, 1, -1)
        bw_update_diff = bw_update_diff.view(1, 1, -1)
        x = torch.cat((bw_innov_diff, bw_update_diff), 2)

        out_FC4 = self.FC4_bw(x)
        in_sigma = torch.cat((self.prior_Q, out_FC4), 2)
        # in_Sigma 是Sigma-GRU右边的输入, self.h_Sigma_bw 是左边的输入
        out_Sigma, self.h_Sigma_bw = self.Vgg16(in_sigma, self.h_Sigma_bw)

        # FC 1 计算得到反向卡尔曼增益
        in_FC1 = out_Sigma
        out_FC1 = self.FC1_bw(in_FC1)
        # out_FC1 = self.kg_gain(in_FC1)
        #####################
        ### Backward Flow ###
        #####################

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw(in_FC2)
        # out_FC2 = self.compute_sigma_bw(in_FC2)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma_bw = out_FC2

        # 返回反向卡尔曼增益
        return out_FC1

    def computeComplexity_backward(self):
        model = backward_kt()
        t1 = torch.tensor([0.7071, 0.7071])
        t2 = torch.tensor([0.2053, 0.2053])
        flops, params = profile(model, (t1, t2))
        print('后向平滑模型计算量: ', flops, '后向平滑模型参数量: ', params)

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = backward_kt()
    t1 = torch.tensor([0.7071, 0.7071])
    t2 = torch.tensor([0.2053, 0.2053])
    torchsummary.summary(model,[(1,1),(1,1)],device='cpu')
    print('parameters_count:', count_parameters(model))

    # model = backward_kt()
    # t1 = torch.tensor([0.7071, 0.7071])
    # t2 = torch.tensor([0.2053, 0.2053])
    #
    # scripted_model = torch.jit.trace(model, [t1, t2],check_trace=False).eval()
    # input_name = "input"
    # shape_list = [(input_name, t1.shape), (input_name, t2.shape)]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # target = tvm.target.Target("llvm", host="llvm")
    #
    # with tvm.transform.PassContext(opt_level=2):
    #     lib = relay.build(mod, target, params=params)
    #     improvemod = tvm.relay.transform.InferType()(mod)
    #
    # since = time.time()
    # from tvm.contrib import graph_executor
    #
    # dtype = "float32"
    # dev = tvm.cpu(0)
    # m = graph_executor.GraphModule(lib["default"](dev))
    # # Set inputs
    # m.set_input(input_name, tvm.nd.array(t1))
    # m.set_input(input_name, tvm.nd.array(t2))
    # # Execute
    # m.run()
    # # Get outputs
    # tvm_output = m.get_output(0)
    #
    # # 将 NumPy 数组转换为 PyTorch 张量
    # numpy_array = tvm_output.asnumpy()
    # tvm_output = torch.from_numpy(numpy_array)
    # print(tvm_output.size())
    # print(tvm_output)
    #
    # time_elapsed = time.time() - since
    # print("优化等级", 2, "所消耗时间:", time_elapsed)
    # 优化等级 0 所消耗时间: 1.0374362468719482
    # 优化等级 1 所消耗时间: 0.569605827331543
    # 优化等级 2 所消耗时间: 0.48335957527160645
    # 优化等级 3 所消耗时间: 0.44747185707092285