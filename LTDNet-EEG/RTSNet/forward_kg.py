import sys
import torch.jit
import torch.nn
import torch.nn as nn
import torch.onnx
import torchsummary
import tvm
from tvm import relay
from thop import profile
from RTSNet.lightWeight.DSCNet import DSCNet


class forward_kg(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 三个GRU
        # GRU to track Sigma
        self.VGG = DSCNet(14, 4)

        # 防止NAN
        self.d_input_S = 24
        self.d_hidden_S = 4
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)

        # 7个全连接层
        # Fully connected 2
        self.d_input_FC2 = 8
        self.d_output_FC2 = 4
        self.d_hidden_FC2 = 320
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.LeakyReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = 8
        self.d_output_FC3 = 4
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.LeakyReLU())

        # Fully connected 4
        self.d_input_FC4 = 8
        self.d_output_FC4 = 4
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.LeakyReLU())

        # Fully connected 6
        self.d_input_FC6 = 4
        self.d_output_FC6 = 10
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.LeakyReLU())

        # 求S
        self.d_input_FC8 = 8
        self.d_output_FC8 = 4
        self.FC8 = nn.Sequential(
            nn.Linear(self.d_input_FC8, self.d_output_FC8),
            nn.LeakyReLU())


    def init_hidden(self):
        self.prior_Q = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        self.prior_R = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        # self.prior_Q = torch.tensor([[[0.0473, 0.0150, 0.0150, 0.0473]]])
        # self.prior_R = torch.tensor([[[0.0390, 0.0148,0.0148, 0.0390]]])
        self.h_Sigma = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])

    def forward(self, obs_innov_diff, fw_update_diff):
        self.init_hidden()
        obs_innov_diff = obs_innov_diff.view(1, 1, -1)
        fw_update_diff = fw_update_diff.view(1, 1, -1)
        x = torch.cat((obs_innov_diff, fw_update_diff), 2)

        output_FC6 = self.FC6(x)
        in_sigma = torch.cat((self.prior_Q, output_FC6), 2)
        out_Sigma, self.h_Sigma = self.VGG(in_sigma, self.h_Sigma)
        h_Sigma = self.h_Sigma
        in_FC8 = torch.cat((self.prior_R, h_Sigma), 2)
        out_S = self.FC8(in_FC8)

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        # out_FC2 即计算出的卡尔曼增益

        return out_FC2

    def computeComplexity_forward(self):
        model = forward_kg()
        t1 = torch.tensor([0.7071, 0.7071])
        t2 = torch.tensor([0.2053, 0.2053])
        flops, params = profile(model, (t1, t2))
        print('前向平滑模型计算量: ', flops, '前向平滑模型参数量: ', params)

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = forward_kg()
    t1 = torch.tensor([0.7071, 0.7071])
    t2 = torch.tensor([0.2053, 0.2053])
    torchsummary.summary(model,[(1,1),(1,1)],device='cpu')
    print('parameters_count:', count_parameters(model))

    flops, params = profile(model, (t1,t2))
    print('flops: ', flops, 'params: ', params)

    params = list(model.parameters())
    params_size = sum([sys.getsizeof(param.data) for param in params])

    print("模型参数所占内存大小：", params_size, "bytes")

    import torch
    from torchsummary import summary

    t1 = torch.tensor([0.2356, 0.2356])
    t2 = torch.tensor([0.2356, 0.2356])
    scripted_model = torch.jit.trace(model, [t1, t2])
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
    M_forward = graph_executor.GraphModule(lib["default"](dev))

    M_forward.set_input(input_name, tvm.nd.array(t1))
    M_forward.set_input(input_name, tvm.nd.array(t2))
    # Execute
    M_forward.run()
    # Get outputs
    tvm_output = M_forward.get_output(0)
    # 将 NumPy 数组转换为 PyTorch 张量
    numpy_array = tvm_output.asnumpy()
    tvm_output = torch.from_numpy(numpy_array)
    KG = tvm_output

    params = list(model.parameters())
    params_size = sum([sys.getsizeof(param.data) for param in params])

    print("优化后的模型参数所占内存大小：", params_size, "bytes")