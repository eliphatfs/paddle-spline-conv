import paddle_spline_conv
import torch_spline_conv
import paddle
import torch
import numpy
import unittest


class TestSplineConv(unittest.TestCase):
    def test_spline_basis(self):
        ks = 5
        for d in range(2, 5):
            u = numpy.random.rand(32, d).astype(numpy.float64)
            is_open = numpy.random.randint(2, size=[d])
            basis, wi = torch_spline_conv.spline_basis(
                torch.tensor(u),
                torch.full([d], ks),
                torch.tensor(is_open).byte(),
                1
            )
            basis_pdl, wi_pdl = paddle_spline_conv.spline_basis(
                paddle.Tensor(u),
                ks,
                paddle.Tensor(is_open),
                1
            )
            basis = basis.detach().numpy()
            basis_pdl = basis_pdl.detach().numpy()
            wi = wi.detach().numpy()
            wi_pdl = wi_pdl.detach().numpy()
            self.assertTrue(
                numpy.allclose(basis, basis_pdl),
                "Dim %d" % (d)
            )
            self.assertTrue(numpy.allclose(wi, wi_pdl),
                "Dim %d Good/Err %d/%d" % (d, (wi == wi_pdl).sum(), (wi != wi_pdl).sum())
            )

    def test_spline_weighting_simple(self):
        x = paddle.Tensor(numpy.array([[1., 2], [3, 4]]))
        w = paddle.Tensor(numpy.array([[[1.], [2]], [[3], [4]], [[5], [6]], [[7], [8]]]))
        basis = paddle.Tensor(numpy.array([[0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5]]))
        wi = paddle.Tensor(numpy.array([[0, 1, 2, 3], [0, 1, 2, 3]]))
        expected = paddle.Tensor(numpy.array([
            [0.5 * ((1 * (1 + 5)) + (2 * (2 + 6)))],
            [0.5 * ((3 * (5 + 7)) + (4 * (6 + 8)))],
        ]))
        w_pdl = paddle_spline_conv.spline_weighting(x, w, basis, wi).detach().numpy()
        self.assertTrue(numpy.allclose(expected, w_pdl))

    def test_spline_weighting(self):
        ks = 5
        for d in range(2, 5):
            x = numpy.random.randn(32, 11).astype(numpy.float64)
            kernel = numpy.random.randn(ks ** d, 11, 17).astype(numpy.float64)
            u = numpy.random.rand(32, d).astype(numpy.float64)
            is_open = numpy.random.randint(2, size=[d])
            basis, wi = torch_spline_conv.spline_basis(
                torch.tensor(u),
                torch.full([d], ks),
                torch.tensor(is_open).byte(),
                1
            )
            basis_pdl, wi_pdl = paddle_spline_conv.spline_basis(
                paddle.Tensor(u),
                ks,
                paddle.Tensor(is_open),
                1
            )
            w = torch_spline_conv.spline_weighting(
                torch.tensor(x),
                torch.tensor(kernel),
                basis, wi
            ).detach().numpy()
            w_pdl = paddle_spline_conv.spline_weighting(
                paddle.Tensor(x),
                paddle.Tensor(kernel),
                basis_pdl, wi_pdl
            ).detach().numpy()
            self.assertTrue(numpy.allclose(w, w_pdl))

    def test_functional(self):
        ks = 5
        for _ in range(20):
            x = numpy.random.randn(23, 10)
            e = numpy.random.randint(23, size=[2, 69])
            dim = numpy.random.randint(2, 5)
            u = numpy.random.rand(69, dim)
            kernel = numpy.random.randn(ks ** dim, 10, 17)
            is_open = numpy.random.randint(2, size=[dim])
            root = numpy.random.randn(10, 17)
            y_pdl = paddle_spline_conv.functional.spline_conv(
                paddle.Tensor(x), paddle.Tensor(e),
                paddle.Tensor(u), paddle.Tensor(kernel),
                ks,
                paddle.Tensor(is_open),
                root_weight=paddle.Tensor(root)
            ).detach().numpy()
            y = torch_spline_conv.spline_conv(
                torch.tensor(x), torch.tensor(e).long(),
                torch.tensor(u), torch.tensor(kernel),
                torch.full([dim], ks),
                torch.tensor(is_open).byte(),
                root_weight=torch.tensor(root)
            ).detach().numpy()
            self.assertTrue(numpy.allclose(y, y_pdl))

    def test_scatter_max(self):
        for n in [1, 5, 25, 50, 100]:
            a = paddle.zeros((n, 5))
            idx = paddle.randint(0, n, shape=[n * 3])
            val = paddle.randn([n * 3, 5])
            output = paddle_spline_conv.functional.scatter_max(a, idx, val).detach().numpy()
            altern = numpy.full((n, 5), -numpy.inf)
            for q, v in zip(idx.detach().numpy(), val.detach().numpy()):
                altern[q] = numpy.maximum(altern[q], v)
            altern[altern == -numpy.inf] = 0
            self.assertTrue(numpy.allclose(output, altern))

    def test_nn_shapes(self):
        mod = paddle_spline_conv.nn.SConv(20, 20)
        g = paddle_spline_conv.nn.GraphData(
            paddle.rand((29, 20)),
            paddle.randint(0, 29, shape=[2, 43]),
            paddle.rand((43, 2))
        )
        self.assertEqual(mod(g).shape, g.x.shape)

    def test_zsconv_results(self):
        data = torch.load("paddle_spline_conv/test_sconv.pt", "cpu")
        mod = paddle_spline_conv.nn.SConv(2, 10)
        mod.convs[0].weight.set_value(paddle.Tensor(data['sconv']['convs.0.weight'].detach().numpy()))
        mod.convs[0].bias.set_value(paddle.Tensor(data['sconv']['convs.0.bias'].detach().numpy()))
        mod.convs[0].root.set_value(paddle.Tensor(data['sconv']['convs.0.root'].detach().numpy()))
        mod.convs[1].weight.set_value(paddle.Tensor(data['sconv']['convs.1.weight'].detach().numpy()))
        mod.convs[1].bias.set_value(paddle.Tensor(data['sconv']['convs.1.bias'].detach().numpy()))
        mod.convs[1].root.set_value(paddle.Tensor(data['sconv']['convs.1.root'].detach().numpy()))
        graph = paddle_spline_conv.nn.GraphData(
            x=paddle.Tensor(data['graph']['x'].astype(numpy.float32)),
            edge_index=paddle.Tensor(data['graph']['edge_index']),
            edge_attr=paddle.Tensor(data['graph']['edge_attr'].astype(numpy.float32)),
        )
        self.assertTrue(numpy.allclose(
            mod(graph).detach().numpy(),
            data['output'].detach().numpy()
        ))



if __name__ == "__main__":
    unittest.main()
