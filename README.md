## Abstract
This project reimplements Kunihiko Fukushima’s 1980 Neocognitron, a
self-organizing neural network designed for position-invariant pattern recognition.
The Neocognitron introduced hierarchical “S-layer” and “C-layer” modules,
inspired by simple and complex cells of the visual cortex. These concepts
eventually influenced modern Convolutional Neural Networks (CNNs).

The main problem addressed in this project is **how a neural network can learn to
recognize patterns regardless of their location, distortions, or local noise**,
without supervision. The Neocognitron solves this by:
- using S-cells with modifiable synapses that self-organize through repeated
  presentations of input patterns,
- using C-cells that aggregate S-cell responses to form translation-invariant
  representations,
- organizing cells into planes that detect specific features across a spatial field.

In this project, we reconstruct the Neocognitron using modern tools while
remaining faithful to its original mathematical formulation. We implement
multi-stage S/C modules, inhibitory mechanisms, self-organization rules, and
feature-plane interactions. Experiments demonstrate position invariance and
robustness to shape distortion, replicating results from the original paper.


## Frameworks, Tools & Code Attribution
*Frameworks used:*
- *Python 3.10*
- *PyTorch* (for tensor operations and model implementation)
- *NumPy, **Matplotlib* (for visualization)
- *Jupyter Notebook* (for training logs and experiments)

*Code structure inspired by:*
- Original 1980 Neocognitron model equations
- Layer-wise modular design patterns used in modern CNN implementations
- Visualization approaches used in Stanford CS231n project examples

*Academic attribution:*
This project is based on the foundational research:

Fukushima, K.  
*Neocognitron: A self-organizing neural network model for a mechanism of  
pattern recognition unaffected by shift in position.*  
Biological Cybernetics, 1980.

PDF source included in this repository: Safari.pdf.

Conceptual lineage referenced from classical vision research:
- Hubel, D.H. & Wiesel, T.N. (1962, 1965): simple/complex cell structure
- Rosenblatt (1962): early neural recognition systems

This project is a *reinterpretation and reimplementation* and does not use
any original source code from the 1980 simulations.

