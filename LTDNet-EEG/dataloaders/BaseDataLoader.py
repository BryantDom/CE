import numpy as np
from colorednoise import powerlaw_psd_gaussian as ppg
from torch.utils.data.dataloader import Dataset
import os
import torch
from scipy.signal import find_peaks



class BaseEEGLoader(Dataset):

    def __init__(self, datapoints: int, samples: list, snr_db: int, noise_color: int = 0):

        super(BaseEEGLoader, self).__init__()

        self.datapoints = datapoints

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        # Load dataset
        self.dataset, self.fs = self.load_data(samples)

        # Get dataset dimensions
        self.samples, self.signal_length, self.num_channels = self.dataset.shape

        # Add gaussian white noise
        self.observations = self.add_noise(self.dataset, snr_db, noise_color)

        # Get EEG-peak labels
        print('--- Centering EEG_signals ---')
        self.labels = self.find_peaks(self.observations)

        # Center datasets
        self.centered_observations, self.centered_states, self.overlaps = self.center(self.observations,
                                                                                      self.dataset,
                                                                                      datapoints,
                                                                                      self.labels)



    def load_data(self, samples: list) -> (torch.Tensor, int):
        """
        Load the dataset as a tensor with dimensions: (Samples, Time, channel)
        :param samples: Array of samples to choose from
        :return: Raw dataset and sampling frequency
        """
        raise NotImplementedError

    def add_noise(self, dataset: torch.Tensor, snr_db: int, noise_color: int) -> torch.Tensor:
        """
        Add noise of a specified color and snr
        :param snr_db: 信噪比，以分贝为单位。
        :param noise_color: 噪声的颜色，0表示白噪声，1表示粉噪声，2表示棕噪声，等等。
        :return: Tensor of noise datasets
        """
        # 是一个计算数据集在时间域上信号功率的张量。
        # 信号功率是指信号在单位时间内所传输的平均能量，通常用单位为瓦特（W）或分贝（dBm）来表示。
        # 在这个函数中，首先使用 dataset.var(1) 计算数据集在第1维（即时间域）上的方差，然后使用 dataset.mean(1) 计算数据集在第1维上的均值。
        # 这两个值的平方相加，得到信号的平均功率。然后将其乘以一个常数10，再使用 torch.log10() 函数取对数，最终得到信号功率的分贝表示。
        signal_power_db = 10 * torch.log10(dataset.var(1) + dataset.mean(1) ** 2)

        # Calculate noise power
        noise_power_db = signal_power_db - snr_db
        noise_power = torch.pow(10, noise_power_db / 20)
        # noise_power相当于 冷大 ,用于调整噪声信号的功率

        # Set for reproducibility
        random_state = 42

        # Generate noise
        noise = [ppg(noise_color, self.signal_length, random_state=random_state) for _ in range(self.num_channels)]
        noise = torch.tensor(np.array(noise)).T.float() * noise_power

        # Add noise
        noisy_data = self.dataset + noise

        return noisy_data
    # ppg 是一种噪声模型，用于生成具有幂律功率谱密度 (Power Spectral Density, PSD) 特征的高斯噪声。
    # 它是通过在频率域上将白噪声信号的幅度谱乘以幂律函数得到的。
    # 具体来说，幂律函数的形式为 f^(-beta)，其中 f 是频率，beta 是一个控制 PSD 特征的参数。
    # 这种噪声模型在信号处理和物理建模中被广泛使用。例如，在通信系统中，噪声通常被建模为高斯白噪声或带有幂律 PSD 的噪声。
    # 在神经科学中， PSD 特征是一种用于描述神经信号的重要特征之一。
    # 因此，使用 powerlaw_psd_gaussian 可以方便地生成具有幂律 PSD 特征的高斯噪声，以模拟实际应用场景中的噪声信号。

    def find_peaks(self, observations: torch.Tensor) -> list:
        """
        Find peaks from observations
        :param observations: Tensor of observations
        :return: List of peak indices
        """
        # Create label list
        labels = []

        for sample in observations:

            # Get labels using a peak-finding algorithm for EEG signals
            sample_labels, _ = find_peaks(sample[:, 0], distance=self.fs*2, width=10)
            labels.append(sample_labels)

        return labels

    def center(self, observations: torch.Tensor, states: torch.Tensor, datapoints: int, labels: list) \
            -> (torch.Tensor, torch.Tensor, list):
        """
        Center observations and noiseless datasets with given labels and window size
        :param observations: Tensor of noisy observations
        :param states: Tensor of noiseless states
        :param datapoints: Size of the window around each peak
        :param labels: Labels of EEG peaks
        :return: Centered observation, centered states and a list overlaps
        """
        # Allocate datasets buffers
        centered_states = []
        centered_observations = []
        overlaps = []

        for n_sample, (obs_sample, state_sample, label) in enumerate(zip(observations, states, labels)):

            last_upper_index = 0

            # Create datasets buffers for the current sample
            sample_centered_observations = []
            sample_centered_states = []
            sample_overlaps = []

            for n_peak, peak in enumerate(label):

                # Get lower and upper indices
                lower_index = peak - int(datapoints / 2)
                upper_index = peak + int(datapoints / 2)

                # Ignore first and last detected peak, since we can not determine where they started/ended
                if lower_index < 0 or upper_index > self.signal_length:
                    last_upper_index = upper_index
                    continue

                else:

                    # Cut out datasets around peak
                    single_peak_observation = obs_sample[lower_index: upper_index]
                    single_peak_state = state_sample[lower_index: upper_index]

                    # Calculate the overlap for stiching everything back together
                    overlap = max(last_upper_index - lower_index, 0)

                    # Append to datasets buffers
                    sample_centered_observations.append(single_peak_observation)
                    sample_centered_states.append(single_peak_state)
                    sample_overlaps.append(overlap)

            # Append to datasets buffers
            centered_observations.append(torch.stack(sample_centered_observations))
            centered_states.append(torch.stack(sample_centered_states))
            overlaps.append(sample_overlaps)

        return torch.cat(centered_observations), torch.cat(centered_states), overlaps

    def __len__(self):
        return self.centered_states.shape[0]

    def __getitem__(self, item):
        return self.centered_observations[item], self.centered_states[item]