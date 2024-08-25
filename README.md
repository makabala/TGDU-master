# Transmission Map-Guided Deep Unfolding Prior Network for underwater image enhancement

## Introduction
In this project, we use Ubuntu 18.04, Python 3.7, Pytorch 1.13.1 and one NVIDIA TITAN RTX. 

## Abstract
In recent years, with the advancement of maritime industries, the importance of underwater image enhancement and restoration has become increasingly evident. However, due to the effects of light scattering, underwater images often suffer from color distortion and low contrast. Most existing networks employ an end-to-end mapping approach, which overlooks the critical role of prior information in the image enhancement process, leading to a lack of interpretability. To address these challenges, we propose a novel deep unfolded prior network guided by transmission maps for underwater image enhancement. Our approach consists of three core components: Adaptive Masked Illumination Dynamic Prior (AMIDP), Transmission-Guided Multi-Scale Convolutional Dictionary (TGMCD), and Constant Spatial Aggregation Module (CSAM). The AMIDP component leverages a masked autoencoder combined with dynamic convolution to extract illumination characteristics from images, allowing us to model illumination and reflection information as shared and unique features, respectively. These features are then fed into the TGMCD module, where they undergo iterative optimization guided by transmission maps. In this process, we replace traditional proximal operators with learnable multi-scale residual blocks, introducing prior knowledge and constraints to enhance model performance. Additionally, the CSAM is designed to enhance the fusion of various features, ensuring that the final enhanced image corrects distortions while preserving critical details. Extensive experiments on multiple underwater datasets demonstrate that our method achieves state-of-the-art performance, validating its effectiveness and superiority. 
Our model and code can be found in the [link](https://github.com/makabala/CDDU-master).

## Keywords
Underwater image enhancement; Deep unfloding network; Feature fusion; Optimization algorithm

## Codes
We would upload our code here as soon as possible, please wait.
