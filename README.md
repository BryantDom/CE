# LTDNet-EEG: A Lightweight Network of Portable/Wearable Devices for Real-Time EEG Signal Denoising

A PyTorch implementation of LTDNet-EEG "LTDNet-EEG: A Lightweight Network of Portable/Wearable Devices for Real-Time EEG Signal Denoising". 

## Abstract

Portable/Wearable (P/W) electroencephalography (EEG) devices capture and analyze EEG signals, which are widely used in different research fields, such as consumer psychology prediction, attention and fatigue monitoring. Nonetheless, EEG signals obtained through P/W devices are sensitive to environmental conditions and physiological activities, rendering real-time denoising a challenge on computation and memory limited consumer electronics (CE). In this work, we propose a lightweight network of P/W devices for real-time EEG signal denoising (LTDNet-EEG). Specifically, LTDNet-EEG performs automatic linearized modeling of nonlinear EEG signals via Taylor series expansion, then utilizes a Kalman smoothing filter to remove noise from EEG signals and designs a lightweight network based on depthwise separable convolution (DSC) to update Kalman gain and other parameters. Besides, it applies data layout and common subexpression elimination to optimize model structure and code computation respectively. Experiments on the benchmark EEGdenoiseNet database show that LTDNet-EEG outperforms the existing state-of-the-art algorithm. Additionally, the LTDNet-EEG can be effectively implemented on the hardware platform equipped with a 4th generation Raspberry Pi (4GB RAM, 16GB Flash). Compared to training and reasoning on CPU, the LTDNet-EEG with optimized approaches achieves approximately a 2.5-fold reduction in execution time which has great potential widely to be used in CE.

### Dependencies

```
python 3.9.12
pytorch 1.10.0
```

### Code Architecture

```
|── dataloaders                    # datasets and load scripts
|── utils                          # Common useful modules
|── Models                         # models of LTDNet-EEG
└── EEGDenoising                   # train scripts
```

## Citation
