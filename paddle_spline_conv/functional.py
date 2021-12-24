import paddle
from typing import Optional
from .ops import spline_weighting, spline_basis


def scatter_max(inputs, idx, val):
    idx_expanded = idx.unsqueeze(-1).expand_as(val)
    cidx = paddle.cast(idx.unsqueeze(-1), val.dtype)
    twosort = paddle.argsort(cidx * (val.max() - val.min() + 2) + val, axis=0)
    gaidx = paddle.stack([
        twosort, paddle.arange(twosort.shape[-1]).unsqueeze(0).expand_as(twosort)
    ], axis=-1)
    gidx = paddle.gather_nd(idx_expanded, gaidx)
    mask = paddle.cast(
        gidx != paddle.concat([gidx[1:], paddle.full_like(gidx[:1], -1)]), 'int64'
    )
    return inputs.scatter(
        paddle.sort(idx),
        paddle.gather_nd(val, gaidx) * mask, overwrite=False
    )


def spline_conv(x: paddle.Tensor, edge_index: paddle.Tensor,
                pseudo: paddle.Tensor, weight: paddle.Tensor,
                kernel_size: paddle.Tensor, is_open_spline: paddle.Tensor,
                degree: int = 1, aggr: str = 'mean',
                root_weight: Optional[paddle.Tensor] = None,
                bias: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    r"""Applies the spline-based convolution operator :math:`(f \star g)(i) =
    \frac{1}{|\mathcal{N}(i)|} \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(i)}
    f_l(j) \cdot g_l(u(i, j))` over several node features of an input graph.
    The kernel function :math:`g_l` is defined over the weighted B-spline
    tensor product basis for a single input feature map :math:`l`.

    x: (number_of_nodes, in_channels), input node features.
    edge_index: (2, number_of_edges)-int, graph edges, given by source and
        target indices, of shape (2 x number_of_edges) in the fixed
        interval [0, 1]
    pseudo: (number_of_edges, dim),
        edge attributes, ie. pseudo coordinates in the fixed interval [0, 1]
    weight: (kernel_size ** dim, in_channels, out_channels),
        trainable weight parameters
    kernel_size: int, number of trainable weight parameters in each edge dimension
    is_open_spline: (dim)-int, {0, 1}, whether to use open or closed B-spline bases
        for each edge dimension
    degree: int, B-spline basis degree (only 1 the default is supported by now)
    aggr: 'mean', 'max' or 'sum', aggregation function of output (default: mean)
    root_weight: (in_channels, out_channels), additional shared trainable
        parameters for each feature of the root node of shape (default: None)
    bias: (out_channels), Optional bias of shape (default: None)
    """
    assert aggr in ['sum', 'mean', 'max'], "Unsupported aggregation"
    x = x.unsqueeze(-1) if x.dim() == 1 else x
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

    row, col = edge_index[0], edge_index[1]
    N, E, M_out = x.shape[0], row.shape[0], weight.shape[2]

    # Weight each node.
    basis, weight_index = spline_basis(pseudo, kernel_size, is_open_spline, degree)

    out = spline_weighting(x[col], weight, basis, weight_index)

    # Convert E x M_out to N x M_out features.
    # row_expanded = row.unsqueeze(-1).expand_as(out).reshape([-1])
    if aggr in ['sum', 'mean']:
        out = paddle.zeros(
            (N, M_out), dtype=out.dtype
        ).scatter(row.reshape([-1]), out, overwrite=False)
    if aggr == 'max':
        out = scatter_max(paddle.zeros(
            (N, M_out), dtype=out.dtype
        ), row, out)

    # Normalize out by node degree (if wished).
    if aggr == 'mean':
        ones = paddle.ones((E,), dtype=x.dtype)
        deg = paddle.zeros(
            (N,), dtype=x.dtype
        ).scatter(row.reshape([-1]), ones.reshape([-1]), overwrite=False)
        out = out / paddle.clip(deg.unsqueeze(-1), min=1)

    # Weight root node separately (if wished).
    if root_weight is not None:
        out = out + paddle.matmul(x, root_weight)

    # Add bias (if wished).
    if bias is not None:
        out = out + bias

    return out
