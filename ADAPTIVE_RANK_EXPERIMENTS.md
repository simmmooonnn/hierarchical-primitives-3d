# Adaptive Rank Experiment Summary

This file summarizes the benchmark results for several Tucker rank selection methods tested on synthetic CAD-like TSDF inputs.

## Methods

- `fixed`
  - Uniform global rank for the full TSDF.
  - Simple and fastest, but not adaptive to local complexity.

- `adaptive`
  - Increases rank globally by fixed steps (e.g. +4) until a reconstruction threshold is met.
  - More accurate than fixed for some inputs, but can be slower and still ignores local spatial variations.

- `adaptive_complexity`
  - Chooses a narrower global rank search window based on an overall complexity estimate.
  - Saves some time versus full adaptive search, but still uses a single global rank.

- `complexity_only`
  - Picks one global rank from the complexity estimate and does one decomposition.
  - Fastest of the adaptive-like methods, but lower accuracy on more complex shapes.

- `blockwise`
  - Splits the input into 8 equal spatial blocks and decomposes each block separately.
  - Uses adaptive rank per block but starts from the same coarse candidate set.

- `blockwise_complexity`
  - Splits the input into 8 spatial blocks.
  - Uses each block’s complexity to select a starting rank, then increases until the error threshold is met.
  - Best middle ground in these tests.

- `regionwise_complexity`
  - Splits input recursively into smaller regions based on complexity.
  - Uses local complexity to set a starting rank, then adapts upward per region.
  - Highest fidelity in these experiments, but also the slowest.

## Complexity factors used

Complexity was estimated from the TSDF using:
- normalized variance,
- local gradient magnitude,
- and surface voxel density.

## Benchmark data

Relative error is the normalized reconstruction error; time is the end-to-end decomposition runtime.

Mode | Mean error | Mean time (s) | Mean rank
--- | --- | --- | ---
fixed | 3.4202e-02 | 0.58 | 16.00
adaptive | 3.4202e-02 | 2.14 | 11.00
adaptive_complexity | 4.7305e-02 | 1.31 | 8.00
complexity_only | 6.4251e-02 | 0.76 | 6.00
blockwise | 4.7305e-02 | 1.51 | 8.00
blockwise_complexity | 1.8946e-02 | 2.20 | 11.50
regionwise_complexity | 0.0000e+00 | 2.89 | 7.50

## Per-sample details

Sample | Shape | Complexity | Method | Error | Time (s) | Rank
--- | --- | --- | --- | --- | --- | ---
0 | simple_box | 0.20 | fixed | 2.3121e-07 | 0.50 | 16
0 | simple_box | 0.20 | adaptive | 2.7467e-07 | 0.36 | 6
0 | simple_box | 0.20 | adaptive_complexity | 2.7467e-07 | 0.47 | 6
0 | simple_box | 0.20 | complexity_only | 2.7467e-07 | 0.37 | 6
0 | simple_box | 0.20 | blockwise | 2.7467e-07 | 0.64 | 6
0 | simple_box | 0.20 | blockwise_complexity | 2.5762e-07 | 0.30 | 7
0 | simple_box | 0.20 | regionwise_complexity | 0.0000e+00 | 3.63 | 8
1 | gear | 0.40 | fixed | 7.2674e-02 | 0.38 | 16
1 | gear | 0.40 | adaptive | 7.2674e-02 | 2.13 | 16
1 | gear | 0.40 | adaptive_complexity | 9.5250e-02 | 1.08 | 10
1 | gear | 0.40 | complexity_only | 1.2206e-01 | 0.43 | 6
1 | gear | 0.40 | blockwise | 9.5250e-02 | 1.37 | 10
1 | gear | 0.40 | blockwise_complexity | 4.1535e-02 | 2.99 | 16
1 | gear | 0.40 | regionwise_complexity | 0.0000e+00 | 1.82 | 7
2 | simple_box | 0.60 | fixed | 2.3121e-07 | 0.99 | 16
2 | simple_box | 0.60 | adaptive | 2.7467e-07 | 0.96 | 6
2 | simple_box | 0.60 | adaptive_complexity | 2.7467e-07 | 0.94 | 6
2 | simple_box | 0.60 | complexity_only | 2.7467e-07 | 1.13 | 6
2 | simple_box | 0.60 | blockwise | 2.7467e-07 | 1.50 | 6
2 | simple_box | 0.60 | blockwise_complexity | 2.5762e-07 | 0.71 | 7
2 | simple_box | 0.60 | regionwise_complexity | 0.0000e+00 | 3.77 | 8
3 | gear | 0.80 | fixed | 6.4134e-02 | 0.44 | 16
3 | gear | 0.80 | adaptive | 6.4134e-02 | 5.10 | 16
3 | gear | 0.80 | adaptive_complexity | 9.3971e-02 | 2.75 | 10
3 | gear | 0.80 | complexity_only | 1.3495e-01 | 1.11 | 6
3 | gear | 0.80 | blockwise | 9.3971e-02 | 2.52 | 10
3 | gear | 0.80 | blockwise_complexity | 3.4248e-02 | 4.79 | 16
3 | gear | 0.80 | regionwise_complexity | 0.0000e+00 | 2.35 | 7
