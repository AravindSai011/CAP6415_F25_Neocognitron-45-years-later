# Neocognitron: 45 Years Later  
# Rebuilding Fukushima’s 1980 Neocognitron using modern computer vision techniques

## Abstract
This project revisits Fukushima’s 1980 Neocognitron through a modern computer vision lens.  
We implement two models using PyTorch:  
(1) a modern CNN baseline, and  
(2) a Neocognitron-inspired architecture with S-cells, C-cells, and lateral inhibition.  
Both models are trained and evaluated on MNIST. Results include training curves,  
confusion matrix, sample predictions, and a comparison of biological vs. modern architectures.


## Overview
This project revisits the original 1980 Neocognitron and rebuilds it using modern computer vision methods.  
The implementation includes:

- **Improved Modern CNN**
- **Neocognitron with true S-cells, C-cells, and lateral inhibition**

Both models are trained and evaluated on the MNIST handwritten digit dataset using PyTorch.

---

## Features
- S-cells with excitatory convolution + fixed surround inhibition  
- C-cells for shift-invariant pooling  
- Biologically inspired lateral inhibition  
- Modern CNN baseline for comparison  
- Training curves, confusion matrix, and sample predictions  
- Clean and reproducible PyTorch pipeline (train + eval)

---

## Repository Structure
```text
src/
├── model.py            # Modern CNN + Neocognitron S/C-layer architecture
├── train.py            # Training loop
├── eval.py             # Evaluation script
├── dataset.py          # MNIST dataloaders
└── utils.py            # Plotting and helper utilities

results/
├── best_model.pth
├── training_loss_curve.png
├── training_accuracy_curve.png
├── confusion_matrix.png
└── sample_predictions.png

requirements.txt
week1log.txt
week2log.txt
week3log.txt
week4log.txt
week5log.txt
```


## Installation

Install all required packages:

```bash
pip install -r requirements.txt

```


## Training

Run the training script:

```bash
python src/train.py
```

All outputs (loss/accuracy curves, checkpoints) are saved in the ``` results/ ```directory.

## Evaluation

Evaluate the trained model:
```bash
python src/eval.py
```

This generates:

* Confusion matrix

* Sample prediction images

* Final test accuracy

All stored in ``` results/. ```

## Results

Test Accuracy: 98%+

Stable behavior with S/C layers + lateral inhibition

Clear visualizations for understanding model performance

## Summary

This project modernizes Fukushima’s Neocognitron using contemporary computer vision workflows.
It demonstrates how early biologically inspired concepts—S-cells, C-cells, and lateral inhibition—connect to today’s convolutional architectures.


