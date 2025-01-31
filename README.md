# Optimization and applications of echo state networks with leaky-integrator neurons
 
badges met DOIs

auteurs

practisch de abstract maar dan zonder plagiaat van jezelf

ook uitleg $g_{\text{inf}}(V) = \int_0^L g(x, V) \, dx$

## uitleg unique contribution

uitleg

# usage

installeer `pip install -r .\requirements.txt`

open de demo file notebook

filetree

## EchoStateNetwork Params

hier komt een uitleg van de parameters van de class

- input_dim: Dimension of input data
- reservoir_size: Number of neurons in the reservoir
- output_dim: Dimension of output data
- leaking_rate: (a) Self coupling constant
- step_size: (d) Time step size
- time_scale: (c) Scale of time evolution
- spectral_radius: Spectral radius of reservoir weight matrix
- sparsity: Proportion of recurrent weights set to zero
- input_scaling: Scaling factor for input weights
- regularization: Regularization coefficient for ridge regression
- activation: Activation function for ESN nodes

$latex formule inclusief A D C$

# Figures

<img src="output/ginf_activator_plot.png" width=50% height=50%>
<img src="output/mg_prediction_plot.png" width=50% height=50%>
<img src="output/mg_phase_space_plot.png" width=50% height=50%>
<img src="output/sine_prediction_plot.png" width=50% height=50%>

# License
This extension is published under the MIT license.

# Contact

This work is part of [publication]. For contact information email timkamsma