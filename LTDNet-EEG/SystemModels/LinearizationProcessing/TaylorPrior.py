# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# 泰勒展开（Taylor series expansion）是一种数学方法，用于通过在函数的某一点周围展开函数，以近似地描述该函数的形状。
# 最小二乘法（Least Squares Method）是一种数学算法，用于在线性回归中找到最佳拟合直线。它通过最小化观察值和拟合值之间的差距来实现此目的。
# _____________________________________________________________

from scipy.special import factorial
import torch
import numpy as np
from SystemModels.LinearizationProcessing.BasePrior import BasePrior
from torch.nn.functional import pad


class TaylorPrior(BasePrior):

    def __init__(self, **kwargs):

        super(TaylorPrior, self).__init__()

        # Get Taylor model parameters
        self.taylor_order = kwargs['taylor_order'] if 'taylor_order' in kwargs.keys() else 7
        #泰勒展开的阶数，默认值为5。
        self.delta_t = kwargs['delta_t'] if 'delta_t' in kwargs.keys() else 1/512
        #时间间隔，默认值为1。
        self.channels = kwargs['channels']
        #通道数量
        self.window_type = kwargs['window_type'] if 'window_type' in kwargs.keys() else 'rectangular'
        #窗口类型，默认为“矩形”
        self.window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() else 7
        #窗口大小，默认为5
        self.window_parameter = kwargs['window_parameter'] if 'window_parameter' in kwargs.keys() else 1
        #窗口参数，默认为1
        assert self.taylor_order >= 1, 'Taylor order must be at least 1'
        assert self.delta_t > 0, 'Time delta needs to be positive'
        # 应为泰勒的阶数最小为1，delta为正。所以这里检查是否满足执行泰勒的公式再决定要不要执行
        self.basis_functions = np.array([[self.delta_t ** k / factorial(k)] for k in range(1, self.taylor_order + 1)])
        self.basis_functions = torch.from_numpy(self.basis_functions).float()
        #基函数是一个数组形式，factorial(k)是阶乘函数是K!,delta_t ** k是delta_t的k次方
        #k是从1到taylor_order（包括）的整数，表示每一阶的泰勒展开

        # 将上面定义的Numpy数组转换为PyTorch的Tensor。torch.from_numpy函数可以将Numpy数组转换为PyTorch Tensor。
        # .float()方法将Tensor的数据类型从默认的整型转换为浮点型。
        # 这样，在PyTorch中，我们可以对该Tensor进行更多的操作，比如矩阵运算，梯度下降等。
        self.derivative_coefficients = torch.ones(self.taylor_order,1)
        # 表示导数系数
        self.window_weights = self.create_window(self.window_type, self.window_size, self.window_parameter)
        # 输入窗口类型，窗口大小和窗口参数，调用下面的"create_window"的方法来创建窗口权重

    def create_window(self, window_type: str, window_size: int, window_parameter: float) -> torch.Tensor:
        """
        Create the weights given by the window parameters
        :param window_type: The type of window used, can be 'rectangular',  'exponential', 'gaussian', 'linear'
        窗口的类型，可以是“矩形”，“指数”，“高斯”或“线性”。
        :param window_size: Size of the window
        窗口的大小
        :param window_parameter: Parameter corresponding to the window type, e.g. standard deviation for 'gaussian'
        与窗口类型相关的参数，例如对于高斯用标准差来表征
        :return: Weights of the window
        在后续的数据处理中对数据进行加权。
        """

        if window_type == 'rectangular':
            weights = torch.from_numpy(np.array([window_parameter for _ in range(window_size)]))

        elif window_type == 'exponential':
            weights = np.array([window_parameter ** (np.abs(w - int(window_size / 2))) for w in range(window_size)])
            weights = torch.from_numpy(weights)

        elif window_type == 'gaussian':
            weights =  np.array([window_parameter * np.exp(- (w - int(window_size / 2)) ** 2 / (2 * window_parameter)) for w in range(window_size)])
            weights = torch.from_numpy(weights)

        elif window_type == 'linear':
            slope = 1 / window_size
            weights = np.array([-slope*np.abs(w-int(window_size / 2)) + window_parameter for w in range(window_size)])
            weights = torch.from_numpy(weights)

        else:
            raise ValueError('Window not supported')

        return weights

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """
        用泰勒最小二乘的方法对一个时间序列数据x-t（datasets）进行拟合，以得到数据的阶导数的近似值。
        """
        # Get datasets dimensions
        batch_size, self.time_steps, self.channels = data.shape
        #从输入的数据获取batch size、time steps （τ）和 channels 的维度信息

        # Reshape basis functions to match datasets 基函数:泰勒前面系数
        # (1,1,5)
        basis_functions = self.basis_functions.reshape((1, 1, -1))
        # (5,10,5)
        basis_functions = basis_functions.repeat((self.window_size, batch_size, 1))
        #重新设置基函数（文中的φ**T）的形状来适应数据
        #窗口大小和批次大小次数不变，基函数要重复使用。

        # Initialize coefficient buffer
        derivative_coefficients = torch.zeros(self.time_steps, self.taylor_order, self.channels)
        #用τ、泰勒阶数、和通道数来初始化一个名为 derivative_coefficients 的张量
        #该张量将存储每个τ的一阶导数近似值。

        # Calculate the size of half a window
        half_window_size = int(self.window_size / 2)
        #计算一半的窗口大小

        # Calculate the amount of padding necessary
        lower_bound_index = - half_window_size
        upper_bound_index = half_window_size + 1
        #如果需要填充的话，计算上界下届填充量

        # Pad datasets for the averaging
        padded_data = pad(data, (0, 0, -lower_bound_index, upper_bound_index), 'replicate')
        #window type是矩形，用对应的weiht取平均

        # Solve T-LS problems
        for t in range(self.time_steps):
            #使用循环枚举每一个时间步τ

            # Get current observations and next observations
            current_state = padded_data[:,t-lower_bound_index-half_window_size:t+half_window_size-lower_bound_index+1]
            observations = padded_data[:,t-lower_bound_index-half_window_size+1:t+half_window_size-lower_bound_index+2]
            #定义了一个滑动窗口，该窗口在数据中移动，并在每个时间步获取数据的一部分。
            #它获取padded_data中从索引为（t-lower_bound_index-half_window_size）到（t+half_window_size-lower_bound_index+1）的所有数据列

            # Target for the LS-regression  计算观察值和拟合值之间的差距
            # (10,5,2)变成(5,10,2)
            target_tensor = (observations - current_state).reshape(self.window_size, batch_size, -1)

            # Covariance matrix of the basis function 计算基函数的协方差矩阵
            # 矩阵乘法
            covariance = torch.bmm(basis_functions.mT, basis_functions)

            # Get the cross correlation for each time step 计算每一个时间步的基函数与目标的交叉相关系数
            cross_correlation = torch.bmm(basis_functions.mT, target_tensor)
            #用来表征基函数与莫表的相关程度，越大越匹配

            # Weight both the correlation and the covariance
            # 对协方差矩阵和交叉相关系数使用窗口权重进行加权
            weights = self.window_weights.reshape(-1,1,1)
            #weights 将 self.window_weights 的维度重新定义为 (-1, 1, 1)。其中，-1 表示让 Python 自动计算维度数值。

            # c = nn.Sequential(
            #     nn.Linear(7, 4),
            #     nn.Sigmoid(),
            #     nn.Linear(4, 7))
            # weighted_covariance = c(weights + covariance).sum(0)

            weighted_covariance = (weights * covariance).sum(0)
            #weights 与 covariance 相乘再求和得到的协方差矩阵
            weighted_cross_correlation = (weights * cross_correlation).sum(0)

            # Perform regression  计算一元线性回归的导数
            derivatives_t = torch.mm(torch.linalg.pinv(weighted_covariance), weighted_cross_correlation)
            """
            实现的是最小二乘回归的公式：
            derivatives_t = (weighted_covariance ^ -1) * weighted_cross_correlation
            其中，(weighted_covariance ^ -1)表示weighted_covariance的逆矩阵， *表示矩阵乘法。
            """
            # Fill buffer  将导数的值存入缓存
            derivative_coefficients[t] = derivatives_t

        self.derivative_coefficients = derivative_coefficients

        return derivative_coefficients,self.basis_functions


