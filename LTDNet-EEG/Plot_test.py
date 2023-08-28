import numpy as np
from matplotlib import pyplot as plt
from utils.four_evaluating_indicator import SSD, MAD, PRD, CosineSim

def plot_test_results(Smother_out, EEG_test, noiseEEG_test,MSE_test_dB_avg):
    states = EEG_test.reshape(5120, )
    filtered_signal = Smother_out.reshape(5120, )
    states = states.detach().numpy()
    filtered_signal = filtered_signal.detach().numpy()

    loss = MSE_test_dB_avg
    ssd = mad = prd = cosineSim = 0
    for i in range(10):
        index = 512 * i
        ssd_index = SSD(states[index:index + 512], filtered_signal[index:index + 512])
        ssd = ssd + ssd_index
        mad = mad + MAD(states[index:index + 512], filtered_signal[index:index + 512])
        prd = prd + PRD(states[index:index + 512], ssd_index)
        cosineSim = cosineSim + CosineSim(states[index:index + 512], filtered_signal[index:index + 512])

    print('mse是', loss, 'db')
    print('SSD是', ssd / 10, 'uv')
    print('MAD是', mad / 10, 'uv')
    print('PRD是', prd, '%')
    print('Cosine Sim是', cosineSim / 10)

    test_out = Smother_out[:1000]
    test_target = EEG_test.reshape(5120,)[:1000]
    noiseEEG_test = noiseEEG_test.reshape(5120,)[:1000]

    test_out = test_out.detach().numpy()
    test_target = test_target.detach().numpy()
    noiseEEG_test = noiseEEG_test.detach().numpy()
    # 形状(2,512)

    fig = plt.figure()
    plt.rcParams['lines.linewidth'] = 0.8
    plt.plot(np.arange(0, 1, 1 / 1000), test_out, label="denoise", color='k')
    plt.plot(np.arange(0, 1, 1 / 1000), test_target, label="ground-truth", color='royalblue')
    plt.plot(np.arange(0, 1, 1 / 1000), noiseEEG_test, label="noisy", color='tan')
    plt.title("val")
    plt.legend()
    plt.savefig('Results/three_results.png')
    plt.show()


    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # 绘制第一条数据
    axs[0].plot(np.arange(0, 1, 1 / 1000), noiseEEG_test, label='noisy', color='red')
    axs[0].set_title('noisy')

    # 绘制第二条数据
    axs[1].plot(np.arange(0, 1, 1 / 1000), test_target, label='ground-truth', color='green')
    axs[1].set_title('ground-truth')

    # 绘制第三条数据
    axs[2].plot(np.arange(0, 1, 1 / 1000),test_out, label='denoise' , color='blue')
    axs[2].set_title('denoise')

    # 添加图例和设置布局
    for ax in axs:
        ax.legend()
    plt.tight_layout()

    # 显示图形
    plt.show()