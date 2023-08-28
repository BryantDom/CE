"""This file contains the settings for the simulation"""
import torch
import argparse

def general_settings():
    ### Dataset settings
    parser = argparse.ArgumentParser(prog = 'RTSNet',description = 'Training parameters')

    ### Training settings
    parser.add_argument('--n_steps', type=int, default=1000, metavar='N_steps',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--n_batch', type=int, default=20, metavar='N_B',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    # 优化策略
    parser.add_argument('--isOptimize', type=bool, default=False, metavar='isOptimize',
                        help='need to perform TVM optimization ( False:no optimize , True:Optimize )')

    # 随机种子
    parser.add_argument('-seed', default=11, type=int, help='set seed for model')


    args = parser.parse_args()
    return args

    # 随机种子:2023 VGG 855lstm
    #        :800  SPP_improve+kt_gain
    #        :1000 SPP_improve
    #        :35   SPP_3+S_Attention
    #        :200,101  kt_gain+kg_gain
    #        18 16 V2 v3