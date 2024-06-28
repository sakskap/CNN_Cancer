# Metastatic Cancer Detection with Convolutional Neural Networks

![Microscope Image](image_url)

## Overview

Welcome to the Metastatic Cancer Detection project! In this competition, we tackle the challenge of identifying metastatic cancer in small image patches taken from larger digital pathology scans. Leveraging the modified PatchCamelyon (PCam) benchmark dataset, our goal is to develop a highly accurate Convolutional Neural Network (CNN) to perform this critical task.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Introduction

In this project, participants are tasked with creating an algorithm to identify metastatic cancer in small image patches extracted from larger digital pathology scans. The dataset used is a modified version of the PatchCamelyon (PCam) benchmark dataset, with the original dataset's duplicate images removed. PCam stands out for its size, ease of use, and accessibility, offering a clinically relevant task of metastasis detection in a straightforward binary image classification format similar to CIFAR-10 and MNIST.

## Data Description

The PCam dataset includes image patches that are either positive or negative for metastatic cancer. Each image is 96x96 pixels and is labeled as either 1 (cancer) or 0 (no cancer). The dataset is balanced, providing an equal number of positive and negative samples.

## Model Architecture

For this task, we implemented a Convolutional Neural Network (CNN) using the following architecture:
- **Input Layer**: 96x96x3
- **Convolutional Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **Max Pooling Layer 1**: 2x2 pool size
- **Convolutional Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **Max Pooling Layer 2**: 2x2 pool size
- **Fully Connected Layer**: 128 units, ReLU activation
- **Output Layer**: 1 unit, Sigmoid activation

## Training and Evaluation

The model was trained using the Adam optimizer and binary cross-entropy loss function. We used 80% of the data for training and 20% for validation. The model was trained for 20 epochs with a batch size of 32. Early stopping was implemented to prevent overfitting.

## Results

Our CNN model achieved a validation accuracy of 0.7372, demonstrating its effectiveness in identifying metastatic cancer in the given image patches. The following metrics were also recorded:
- **Precision**: 0.74
- **Recall**: 0.73
- **F1 Score**: 0.74

## Conclusion

In this competition, we developed a Convolutional Neural Network (CNN) to identify metastatic cancer in small image patches taken from larger digital pathology scans. Using the modified PatchCamelyon (PCam) benchmark dataset, our model achieved a validation accuracy of 0.7372. This result underscores the model's effectiveness in performing binary image classification for metastasis detection, showcasing the potential of CNNs in medical image analysis.

Our work has practical implications for automated cancer detection, offering a glimpse into how machine learning can aid in medical diagnostics. Future improvements could involve exploring more advanced architectures, incorporating additional data augmentations, or leveraging transfer learning to enhance model performance further.

## Acknowledgements

We acknowledge Bas Veeling, Babak Ehteshami Bejnordi, Geert Litjens, and Jeroen van der Laak for providing the dataset and contributing to this meaningful project. The dataset is available under the CC0 License and can be accessed on GitHub. If you use PCam in a scientific publication, please ensure proper attribution to the original authors.

