import torch
import torch.nn as nn

def compute_num(input_num,output_num,k):
    num = input_num + output_num
    for i in range(k):
        num = int((num + 2) * 0.5)
    return num

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DSCNet(nn.Module):
    def __init__(self,input_num,output_num):
        super(DSCNet, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(1, 1, 1, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(1, 1, 1, padding=1),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(compute_num(input_num,output_num,2), 4))


    def forward(self, x,hx):
        x = torch.unsqueeze(x, dim=0)
        hx = torch.unsqueeze(hx, dim=0)
        x = torch.cat((x, hx), dim=3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, compute_num(self.input_num,self.output_num,2))
        x = self.fc(x)
        x = x.reshape(1,1,4)
        return x,x

if __name__ == '__main__':
    spp = DSCNet(24,4)
    a = torch.randn(1, 1, 24)
    b = torch.randn(1, 1, 4)
    out,out2=spp(a,b)
    print(out.size())
    from thop import profile
    t1 = torch.randn(1,1,24)
    t2 = torch.randn(1,1,4)
    flops, params = profile(spp, (t1, t2))
    print('flops: ', flops, 'params: ', params)