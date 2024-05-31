# Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration
Deformable image registration is a fundamental step for medical image analysis. Recently, transformers have been used for registration and outperformed Convolutional Neural Networks (CNNs). Transformers can capture long-range dependence among image features, which have been shown beneficial for registration. However, due to the high computation/memory loads of self-attention, transformers are usually used at downsampled feature resolutions and cannot capture fine-grained long-range dependence at the full image resolution. This limits deformable registration as it necessitates precise dense correspondence between each image pixel. Multi-layer Perceptrons (MLPs) without self-attention are efficient in computation/memory usage, enabling the feasibility of capturing fine-grained long-range dependence at full resolution. Nevertheless, MLPs have not been extensively explored for registration and are lacking the consideration of inductive bias crucial for medical registration tasks. In this study, we propose the first correlation-aware MLP-based registration network (CorrMLP) for deformable medical image registration. Our CorrMLP introduces a correlation-aware multi-window MLP block in a novel coarse-to-fine registration architecture, which captures fine-grained multi-range dependence to perform correlation-aware coarse-to-fine registration. Extensive experiments with seven public medical datasets show that our CorrMLP outperforms state-of-the-art registration methods.

## Overview
![Overview](https://github.com/MungoMeng/Registration-CorrMLP/blob/master/Figure/Overview.png)

## Notification
The official code will be released soon.

## Publication
For more details, please refer to our paper:
* **Mingyuan Meng, Dagan Feng, Lei Bi, and Jinman Kim, "Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration," IEEE/CVF conference on Computer Vision and Pattern Recognition (CVPR), 2024. (Oral Presentation && Best Paper Candidate)**
