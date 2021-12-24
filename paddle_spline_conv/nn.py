import paddle
import paddle.nn
from .functional import spline_conv


class SplineConv(paddle.nn.Layer):
    """
    SplineConv layer.
    See functional.spline_conv for parameters.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int,
        is_open_spline: bool = True,
        degree: int = 1,
        aggr: str = 'mean',
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_open = is_open_spline
        self.kernel_size = kernel_size
        self.ic = in_channels
        self.oc = out_channels
        self.dim = dim
        self.degree = degree
        assert degree == 1, "SplineConv: degree != 1 is not supported yet"
        self.aggr = aggr
        self.root_weight = root_weight
        if root_weight:
            self.root = self.create_parameter((self.ic, self.oc))
        else:
            self.root = None
        if bias:
            self.bias = self.create_parameter((out_channels,), is_bias=True)
        else:
            self.bias = None
        self.weight = self.create_parameter((kernel_size ** dim, in_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        is_open = paddle.full([self.dim], int(self.is_open), 'int64')
        return spline_conv(
            x, edge_index, edge_attr, self.weight, self.kernel_size, is_open,
            self.degree, self.aggr,
            self.root,
            self.bias
        )


class SConv(paddle.nn.Layer):
    """
    SConv layer formed by stacking multiple spline convs.
    """
    def __init__(self, input_features, output_features, num_layers=2):
        super(SConv, self).__init__()

        self.in_channels = input_features
        self.num_layers = num_layers
        self.convs = paddle.nn.LayerList()

        for _ in range(self.num_layers):
            conv = SplineConv(input_features, output_features, dim=2, kernel_size=5, aggr="max")
            self.convs.append(conv)
            input_features = output_features

        input_features = output_features
        self.out_channels = input_features

    def forward(self, data: 'GraphData'):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = [x]

        for conv in self.convs[:-1]:
            xs += [paddle.nn.functional.relu(conv(xs[-1], edge_index, edge_attr))]

        xs += [self.convs[-1](xs[-1], edge_index, edge_attr)]
        return xs[-1]


class GraphData:
    """
    GraphData helper class for SConv layer.

    Input attributes are node features, edge indices and edge pseudo-coordinates, resp.
    The attributes should be `paddle.Tensor`

    x: (N, d), node features
    edge_index: (2, e), [out_nodes, in_nodes] edges
    edge_attr: (e, dim), edge pseudo-coordinates
    """
    def __init__(self, x, edge_index, edge_attr) -> None:
        self.x = x
        self.edge_index = edge_index[1], edge_index[0]
        self.edge_attr = edge_attr
