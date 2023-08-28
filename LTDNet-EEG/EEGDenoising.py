import torch
from datetime import datetime
from SystemModels.Linear_sysmdl import SystemModel
import utils.config as config
from SystemModels.parameters import F, H, Q, R, m1_0, m2_0
from RTSNet.RTSNet_nn import RTSNetNN
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from dataloaders.EEG_all_DataLoader import EEG_all_Dataloader
from dataloaders.GetSubset import get_subset
from utils.set_seed import set_seed
from Plot_test import plot_test_results

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

####################
### Design Model ###
####################

args = config.general_settings()
set_seed(args)
# 保存模型权重参数路径
path_results = 'checkpoints/'

# 训练参数设置
args.n_steps = 300
args.n_batch = 8
args.lr = 1e-4
args.wd = 1e-3
args.isOptimize = False

###################################
### Data Loader (Generate Data) ###
###################################
print("Data Load")
list_of_sample = [0]
number_of_datasamples = 512
snr_db = 0 # 信噪比，表示信号强度和噪声强度相等，即信噪比为0dB
noise_color = 0
dataloader = EEG_all_Dataloader
dataloader = dataloader(number_of_datasamples, list_of_sample, snr_db, noise_color)
prior_loader, test_loader = get_subset(dataloader, 10)  # 获得10个子集
noiseEEG_train = torch.zeros((10,1,512))
EEG_train = torch.zeros((10,1,512))
noiseEEG_val = torch.zeros((10,1,512))
EEG_val = torch.zeros((10,1,512))
noiseEEG_test = torch.zeros((10,1,512))
EEG_test = torch.zeros((10,1,512))
for i in range(10):
    index = 512*i
    noiseEEG_train[i] = prior_loader.dataset.observations[0,index:index+512,0].float()
    EEG_train[i] = prior_loader.dataset.dataset[0, index:index + 512, 0].float()
    noiseEEG_val[i] = prior_loader.dataset.observations[0, index + 5120:index + 5632, 0].float()
    EEG_val[i] = prior_loader.dataset.dataset[0, index + 5120:index + 5632, 0].float()
    noiseEEG_test[i] = prior_loader.dataset.observations[0, index + 10240:index + 10752, 0].float()
    EEG_test[i] = prior_loader.dataset.dataset[0, index + 10240:index + 10752, 0].float()

sys_model = SystemModel(F, Q, H, R)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)

### RTSNet with full info ###
# Build Neural Network
print("RTSNet with EEG")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))

## Train Neural Network
RTSNet_Pipeline = Pipeline()
RTSNet_Pipeline.setModel(args,sys_model,RTSNet_model)

# 训练和验证过程
# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch,cv_out,cv_target] = RTSNet_Pipeline.NNTrain(args,sys_model, noiseEEG_val,EEG_val,noiseEEG_train, EEG_train, path_results)
# 测试过程
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Smother_out,RunTime] = RTSNet_Pipeline.NNTest(args,sys_model, noiseEEG_test ,EEG_test, path_results)

# plot_results(cv_out, cv_target, noiseEEG_val,MSE_train_dB_epoch ,MSE_cv_dB_epoch)
plot_test_results(Smother_out, EEG_test, noiseEEG_test,MSE_test_dB_avg)