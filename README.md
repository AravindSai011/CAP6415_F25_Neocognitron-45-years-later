## Abstract
This project revisits the Neocognitron (Fukushima, 1980), one of the earliest neural architectures capable of shift-invariant pattern recognition. We implement a modern PyTorch-based version of the S-layer and C-layer hierarchy and evaluate it on the MNIST digit dataset. The model extracts features through cascaded simple and complex layers and demonstrates translation-invariant classification performance.

## Framework
* Language: Python
* Libraries: PyTorch, TorchVision, NumPy, Matplotlib

## code Overview
* Implementation of a Neocognitron-inspired convolutional network
* Training and evaluation scripts for MNIST
* Visualization utilities for feature maps and translation-invariance tests
* Demo script for generating prediction images

## Attribution
* Inspired by Fukushima’s Neocognitron (1980)
* MNIST dataset from LeCun et al.
* Built entirely using PyTorch and standard computer vision tools

## Project Structure
```text
Neocognitron-Project/
│
├── src/
│   ├── models_neocognitron.py
│   ├── data_dataset_loader.py
│   ├── train.py
│   ├── eval.py
│   └── demo.py
│
├── results/
│   ├── graphs/
│   ├── feature_maps/
│   ├── invariance/
│   └── demo/
│
├── week1log.txt
├── week2log.txt
├── week3log.txt
├── week4log.txt
├── week5log.txt
│
└── README.md
```
