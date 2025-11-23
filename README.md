## Abstract
This project re-implements Kunihiko Fukushima's 1980 Neocognitron, a self-organizing neural network designed for position-invariant pattern recognition. The Neocognitron introduced hierarchical "S-layer" and "C-layer" modules, inspired by simple and complex cells of the visual cortex. These concepts eventually influenced modern Convolutional Neural Networks (CNNs).

The key challenge here is to determine how a neural network can learn to identify patterns, regardless of translations, distortions, or local noise, in a totally unsupervised fashion. For this the Neocognitron addresses the following:

using S-cells with modifiable synapses that self-organize through repeated presentations of input patterns,
by using C-cells that pool S-cell responses over space to provide translation-invariant representations,
organizing cells into planes that detect particular features across a spatial field.

In this work, we re-implement the Neocognitron with contemporary tools and strictly follow its original mathematical formulation. We implement multi-stage S/C modules, inhibitory mechanisms, self-organization rules, and feature-plane interactions. In experiments, we show position invariance and robustness against shape distortion, which were consistent with the results of the original paper.

Models, Tools & Code Attribution

Frameworks used:
Python 3.10
PyTorch: for tensor operation and model implementation
*NumPy, *Matplotlib (for visualization)

Jupyter Notebook (for training logs and experiments)
Code structure inspired by:
Original 1980 Neocognitron model equations

Layer-wise modular design patterns used in modern CNN implementations
Visualization approaches used in Stanford CS231n project examples

