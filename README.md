# SplineConv implementation for Paddle
This module implements the `SplineConv` operators from

> Matthias Fey, Jan Eric Lenssen, Frank Weichert, Heinrich Müller: SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels (CVPR 2018).

It is in early development, and may have problems. Feel free to open an issue if you find one.

## Requirements
It only needs `paddle`. It is tested on `paddle >= 2.1.0, <= 2.2.0rc1`, but should work for any recent paddle versions.

For development -- since we run tests against `torch-spline-conv`, you will need that.

## Installation
`pip install paddle-spline-conv`

## Usage
Here are some basic usage descriptions. See docstring in code for more detailed descriptions, types and shapes of parameters.

Currently only degree-1 splines are supported. But the basic operators have been ready, and adding more shouldn't be very hard. You are welcome to contribute for higher degree splines!

```python
import paddle_spline_conv

# Stacked SplineConv layers implemented in SConv
g = paddle_spline_conv.nn.GraphData(x, edge_index, edge_attr)
# Input n_features and output n_features
sconv = paddle_spline_conv.nn.SConv(10, 40)
sconv(g)

# Standalone SplineConv layer
paddle_spline_conv.nn.SplineConv(
    in_channels: int,
    out_channels: int,
    dim: int,
    kernel_size: int,
    is_open_spline: bool = True,
    degree: int = 1,
    aggr: str = 'mean',
    root_weight: bool = True,
    bias: bool = True
)

# Standalone SplineConv functional API
paddle_spline_conv.functional.spline_conv(
    x: paddle.Tensor, edge_index: paddle.Tensor,
    pseudo: paddle.Tensor, weight: paddle.Tensor,
    kernel_size: paddle.Tensor, is_open_spline: paddle.Tensor,
    degree: int = 1, aggr: str = 'mean',
    root_weight: Optional[paddle.Tensor] = None,
    bias: Optional[paddle.Tensor] = None
)

# SplineConv-specific operators
paddle_spline_conv.ops.spline_basis
paddle_spline_conv.ops.spline_weighting
paddle_spline_conv.ops.basis_kernel_1d
```
