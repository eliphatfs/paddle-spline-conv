import paddle


def basis_kernel_1d(v, k_mod):
    return 1 - v + (2 * v - 1) * k_mod


def mod_c(a, b):
    return a % b - paddle.cast(
        paddle.logical_and(a.detach() < 0, (a % b).detach() != 0), 'int64'
    ) * b


def spline_basis(pseudo, kernel_size, is_open_spline, degree):
    """
    Computes spline basis.
    pseudo: (E, D), edge pseudo coordinates
    is_open_spline: (D)-int, per-dim open spline flag
    kernel_size: int, kernel size
    degree: int, degree of spline

    returns: 2-tuple of (E, S), basis and weight indices
    """
    _, D = pseudo.shape
    S = (degree + 1) ** D
    s = paddle.arange(0, S, dtype='int64')
    d = paddle.arange(0, D, dtype='int64')
    k_divisor = (degree + 1) ** d
    k_u = s.unsqueeze(-1) // k_divisor  # S, D
    k_mod = k_u % (degree + 1)
    v = pseudo * (kernel_size - degree * is_open_spline)
    # E, S, D
    cprod = kernel_size ** d
    idx = (paddle.cast(v.detach(), 'int64').unsqueeze(-2) + k_mod.unsqueeze(-3))
    idx = (mod_c(idx, kernel_size) * cprod).sum(-1)
    v = v - paddle.floor(v)  # E, D
    # [E, 1, D] broadcast [1, S, D]
    basis_data = basis_kernel_1d(v.unsqueeze(-2), k_mod.unsqueeze(-3))
    basis_data = paddle.prod(basis_data, -1)  # E, S
    return basis_data, idx


def spline_weighting(x, weight, basis, weight_index):
    """
    Computes spline weighting.
    x: (E, I), edge data (typically indexed from node attributes)
    weight: (kernel_size ** ndim, I, out_channel), Kernel weights
    basis/weight_index: (E, S), `spline_basis` outputs
    """
    # [ESIO -> ESOI, EI -> E1I1] -> ESO1
    x = paddle.matmul(weight[weight_index].transpose((0, 1, 3, 2)), x.unsqueeze(1).unsqueeze(-1))
    x = paddle.squeeze(x, -1)  # ESO
    # [ESO -> EOS, ES -> ES1] -> EO1
    x = paddle.matmul(x.transpose((0, 2, 1)), basis.unsqueeze(-1))
    return x.squeeze(-1)
