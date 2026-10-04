"""INT4 packing, reconstruction, and native-kernel checks."""
import platform
import unittest

import numpy as np
import torch

from litespark_inference.nemotron_int4 import quantize, expand


class QuantizationChecks(unittest.TestCase):
    def test_zero_odd_shapes_and_group_tails(self):
        for rows, cols, group in [(3, 37, 16), (5, 128, 128), (1, 3, 2)]:
            weight = torch.zeros(rows, cols, dtype=torch.bfloat16)
            packed, scales, error = quantize(weight, group)
            self.assertEqual(packed.shape, (rows, (cols+1)//2))
            self.assertEqual(scales.shape, (rows, (cols+group-1)//group))
            self.assertTrue(torch.equal(expand(packed, scales, weight.shape, group), weight))
            self.assertEqual(error["relative_rmse"], 0.)

    def test_mse_search_improves_absmax_for_outliers(self):
        torch.manual_seed(615)
        weight = torch.randn(32, 512).bfloat16()
        weight[:, ::128] *= 5
        packed, scales, _ = quantize(weight, 128)
        reconstructed = expand(packed, scales, weight.shape, 128).float()
        x = weight.float().reshape(32, 4, 128)
        scale = (x.abs().amax(-1, keepdim=True)/7).half().float()
        plain = ((x/scale).round().clamp(-7, 7)*scale).bfloat16().float().reshape_as(weight)
        self.assertLess((reconstructed-weight.float()).square().sum(), (plain-weight.float()).square().sum())
        self.assertTrue(torch.isfinite(reconstructed).all())


@unittest.skipUnless(platform.machine().lower() in ("arm64", "aarch64"), "NEON required")
class NativeInt4Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from litespark_inference.nemotron_cpu import NativeKernels
        torch.set_num_threads(10)
        cls.kernels = NativeKernels(10)

    @torch.inference_mode()
    def test_decode_and_prefill_against_expanded_bf16(self):
        torch.manual_seed(48)
        for rows, cols, group in [(17, 37, 18), (127, 1024, 128), (192, 4096, 128)]:
            weight = (torch.randn(rows, cols)*.03).bfloat16()
            packed, scales, _ = quantize(weight, group)
            expanded = expand(packed, scales, weight.shape, group).float()
            for tokens in (1, 5):
                x = torch.randn(tokens, cols).bfloat16().float()
                output = self.kernels.dense_int4(packed.numpy(), scales.view(torch.uint16).numpy(),
                                                  x.numpy(), cols, group)
                np.testing.assert_allclose(output, (x@expanded.T).numpy(), atol=2e-5, rtol=2e-5)
            x = torch.randn(1, cols)
            output = self.kernels.dense_int4(packed.numpy(), scales.view(torch.uint16).numpy(),
                                              x.numpy(), cols, group)
            np.testing.assert_allclose(output, (x@expanded.T).numpy(), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
