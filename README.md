# Pyontronics

[![DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.15113279.svg)](https://doi.org/10.5281/zenodo.15113279)

**Pyontronics** is a Python package for simulating iontronic neuromorphic devices and Echo State Networks (ESNs). It provides tools for physical reservoir computing, including leaky-integrator neurons, physical activation functions, and flexible network architectures for both time series and pulse-based data.

## Installation

Install the core package:
```bash
pip install .
```

To enable network visualization features, install with the optional `graph` dependency:
```bash
pip install .[graph]
```

## Features

- **Echo State Network (ESN)**: Flexible implementation with physical parameters and activation functions.
- **BandPassNetwork**: ESN variant with per-unit timescales for richer dynamics.
- **Physical Activation Functions**: Use conductance models derived from iontronic devices.
- **Linear Autoregression**: Simple AR models for benchmarking and comparison.
- **Visualization**: Visualize ESN architectures as directed graphs (requires `networkx`).
- **Pulse-based and Continuous Data Support**: Train and predict on both single time series and sets of pulses.

## Usage Example

```python
from pyontronics import EchoStateNetwork, visualize_reservoir

# Create an ESN
esn = EchoStateNetwork(
    input_dim=1,
    reservoir_size=100,
    output_dim=1,
)

# Fit to data
esn.fit(inputs, targets)

# Predict
outputs = esn.predict(inputs)

# Visualize (requires networkx)
visualize_reservoir(esn)
```

## Optional Visualization

To visualize network architectures, install with:
```bash
pip install .[graph]
```
and use `visualize_reservoir(esn)`.

## Functionality Overview

- **Reservoir Computing**: Simulate and train ESNs with physical or standard activation functions.
- **Physical Models**: Use `GinfActivator` and `NCNM_activator` for realistic conductance-based activations.
- **Pulse and Bandpass Networks**: Specialized classes for pulse-based data and variable timescales.
- **Linear Autoregression**: Benchmark with classic AR models.
- **Visualization**: Plot ESN graphs with node and edge coloring.

## API Highlights

- `EchoStateNetwork`: Core ESN class.
- `BandPassNetwork`: ESN with per-unit timescales.
- `PulseEchoStateNetwork`, `PulseBandPassNetwork`: For pulse-based tasks.
- `GinfActivator`, `NCNM_activator`: Physical activation functions.
- `LinearAutoregression`: Simple AR model.
- `visualize_reservoir`: Visualize ESN structure (optional).

## Development
Install the development dependencies with:
```bash
pip install .[graph,dev]
```

## License

MIT License

## Authors

Tim Kamsma  
Jelle Jasper Teijema

## Contact

For questions, please contact t.m.kamsma@uu.nl

Project homepage: [https://github.com/TMKamsma/Pyontronics](https://github.com/TMKamsma/Pyontronics)