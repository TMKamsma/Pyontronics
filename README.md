# Optimization and Applications of Echo State Networks with Leaky-Integrator Neurons
badges met DOIs

## Authors
Tim Kamsma
Jelle Jasper Teijema

## Abstract
This repository provides an Echo State Network (ESN) framework designed for both continuous data streams and sets of shorter pulses, each with a single label. The ESN uses leaky-integrator neurons, which offer a controlled decay of past inputs and states. This design helps the network capture temporal dependencies across diverse applications such as signal classification, time-series prediction, and dynamic modeling.

We also introduce a functional operator $g_{\text{inf}}(V)$ that integrates an internal function $g(x, V)$ over the domain $[0,L]$:

$g_{\text{inf}}(V) = \int_0^L g(x, V) \, dx$.

This operator highlights how internal parameters can transform across a spatial or temporal domain, offering a robust way to capture complex input-output mappings in leaky networks.

## Unique Contribution
Our approach handles both:

1. Single time series (predicting at each step).
2. Multiple shorter series (pulses) (predicting a single label at the end of each pulse).

The leaky-integrator neurons and the integrated operator $g_{\text{inf}}$ allow for flexible state evolution that aligns with many practical tasks.

# usage

1. Clone or download this repository.
2. Install required libraries:
```console
pip install -r .\requirements.txt
```
3. Open and run the demo notebook included in the repository.

## EchoStateNetwork Parameters
Below are the main parameters you can set when creating an EchoStateNetwork instance:

- input_dim: Dimension of the input data.
- reservoir_size: Number of neurons in the reservoir.
- output_dim: Dimension of the output data (e.g., number of classes or regression targets).
- leaking_rate (a): Self-coupling constant, controlling the decay of the reservoir state.
- step_size (d): Discrete time step used in updates.
- time_scale (c): Scaling factor for the reservoir’s internal time evolution.
- spectral_radius: Desired spectral radius of the reservoir weight matrix.
- sparsity: Fraction of reservoir weights set to zero to encourage sparse connectivity.
- input_scaling: Scaling factor applied to input weights.
- regularization: Regularization coefficient (λ) used in ridge regression for output weights.
- activation: Nonlinear activation function (default is $tahn$).

# Figures

<img src="output/ginf_activator_plot.png" width=50% height=50%>
<img src="output/mg_prediction_plot.png" width=50% height=50%>
<img src="output/mg_phase_space_plot.png" width=50% height=50%>
<img src="output/sine_prediction_plot.png" width=50% height=50%>

# License
This extension is released under the MIT License.

# Contact

This work is part of [publication]. For questions, please contact t.m.kamsma@uu.nl