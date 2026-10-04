"""Numerical checks against independently expanded BF16 matrices."""
import platform
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from litespark_inference.nemotron_cpu import NativeExperts, NativeKernels, NativeLinear, _native_convolution


class FixtureCheckpoint:
    def __init__(self, dim, intermediate, group, experts=3):
        self.manifest = {"group_size": group}
        self.tensors, self.dense = {}, {}
        rng = np.random.default_rng(119)
        for expert in range(experts):
            for projection, rows, cols in [("up_proj", intermediate, dim),
                                            ("down_proj", dim, intermediate)]:
                q = rng.integers(-1, 2, (rows, cols), dtype=np.int8)
                scales = rng.uniform(.01, .1, (rows, (cols+group-1)//group)).astype(np.float16)
                planes = (rows+3)//4
                packed = np.zeros((planes, cols), dtype=np.uint8)
                for row in range(rows):
                    packed[row % planes] |= (q[row]+1).astype(np.uint8) << (2*(row//planes))
                key = f"language_model.backbone.layers.0.mixer.experts.{expert}.{projection}.weight"
                self.tensors[key] = torch.from_numpy(packed)
                self.tensors[key+"_scale"] = torch.from_numpy(scales)
                expanded = q.astype(np.float32) * np.repeat(scales.astype(np.float32), group, axis=1)[:, :cols]
                self.dense[expert, projection] = torch.from_numpy(expanded).bfloat16().float()

    def raw(self, name):
        return self.tensors[name]


@unittest.skipUnless(platform.machine().lower() in ("arm64", "aarch64"), "ARM64 NEON required")
class NativeNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(10)
        cls.kernels = NativeKernels(10)

    @torch.inference_mode()
    def test_dense_decode_and_prefill_with_bias(self):
        torch.manual_seed(27)
        for rows, cols in [(17, 23), (128, 512)]:
            weight = torch.randn(rows, cols).bfloat16()
            bias = torch.randn(rows).bfloat16()
            native = NativeLinear(weight, self.kernels, bias)
            for tokens in (1, 7):
                x = torch.randn(1, tokens, cols).bfloat16()
                expected = (x.float() @ weight.float().T + bias.float()).bfloat16()
                torch.testing.assert_close(native(x), expected, atol=.002, rtol=.008)

    @torch.inference_mode()
    def check_experts(self, dim, intermediate, group, tokens, dtype=torch.bfloat16):
        checkpoint = FixtureCheckpoint(dim, intermediate, group)
        config = SimpleNamespace(moe_latent_size=dim, moe_intermediate_size=intermediate, n_routed_experts=3)
        native = NativeExperts(checkpoint, 0, config, self.kernels)
        torch.manual_seed(78)
        x = torch.randn(tokens, dim).to(dtype)
        ids = torch.tensor([[2, 0], [1, 2], [0, 1]])[:tokens]
        route = torch.rand(tokens, 2).softmax(-1)
        expected = torch.zeros(tokens, dim)
        for token in range(tokens):
            for slot in range(2):
                expert = ids[token, slot].item()
                hidden = (x[token].float() @ checkpoint.dense[expert, "up_proj"].T).bfloat16()
                hidden = hidden.relu().square().bfloat16()
                out = (hidden.float() @ checkpoint.dense[expert, "down_proj"].T).bfloat16()
                expected[token] += out.float()*route[token, slot]
        actual = native(x, ids, route)
        # Float reduction order can change a BF16 tie. Require high aggregate
        # agreement as well as a bounded maximum absolute error.
        error = (actual.float()-expected.bfloat16().float()).abs()
        self.assertLess(error.max().item(), max(.005, expected.abs().max().item()*.009))
        self.assertLess(error.norm().item()/expected.norm().item(), .003)

    def test_odd_dimensions_partial_groups(self):
        self.check_experts(37, 71, 19, 3)

    def test_production_expert_shape_decode(self):
        self.check_experts(1024, 2688, 128, 1)

    def test_production_expert_shape_prefill(self):
        self.check_experts(1024, 2688, 128, 3)

    def test_experts_fp32_input_fallback(self):
        self.check_experts(37, 71, 19, 3, dtype=torch.float32)

    @torch.inference_mode()
    def test_causal_convolution_prefill_and_update(self):
        torch.manual_seed(58)
        for channels, tokens, output_tokens in [(19, 3, 3), (10240, 32, 32), (10240, 5, 1)]:
            x = torch.randn(1, channels, tokens).bfloat16()
            weight = torch.randn(channels, 4).bfloat16()
            for bias in (None, torch.randn(channels).bfloat16()):
                expected = torch.nn.functional.conv1d(x, weight[:, None], bias, padding=3, groups=channels)
                expected = torch.nn.functional.silu(expected[:, :, :tokens][:, :, -output_tokens:])
                actual = self.kernels.causal_conv(x, weight, bias, output_tokens)
                torch.testing.assert_close(actual, expected, rtol=.008, atol=.002)

    @torch.inference_mode()
    def test_in_place_convolution_update_and_cache(self):
        from transformers.models.nemotron_h import modeling_nemotron_h as mh
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        torch.manual_seed(811)
        config = NemotronHConfig(hidden_size=16, mamba_num_heads=2, mamba_head_dim=8,
            n_groups=1, ssm_state_size=4, layers_block_type=["linear_attention"])
        mixer = mh.NemotronHMamba2Mixer(config, layer_idx=0).bfloat16().eval()
        _native_convolution(mixer, self.kernels)
        native = mixer.forward.__func__.__globals__["causal_conv1d_update"]
        for batches, channels, width in [(1, 10240, 4), (2, 19, 3), (1, 7, 1)]:
            weight = torch.randn(channels, width).bfloat16()
            bias = torch.randn(channels).bfloat16()
            cache = torch.randn(batches, channels, width).bfloat16()
            for _ in range(3):
                x = torch.randn(batches, channels, 1).bfloat16()
                expected_cache = torch.cat((cache[:, :, 1:], x), dim=-1)
                expected = torch.nn.functional.conv1d(expected_cache, weight[:, None], bias, groups=channels)
                expected = torch.nn.functional.silu(expected)
                actual = native(x, cache, weight, bias)
                torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.008)

    @torch.inference_mode()
    def test_mamba_cache_prefill_and_repeated_decode(self):
        from transformers.cache_utils import DynamicCache
        from transformers.models.nemotron_h import modeling_nemotron_h as mh
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        torch.manual_seed(91)
        config = NemotronHConfig(hidden_size=16, mamba_num_heads=2, mamba_head_dim=8,
            n_groups=1, ssm_state_size=4, layers_block_type=["linear_attention"], chunk_size=8)
        reference = mh.NemotronHMamba2Mixer(config, layer_idx=0).bfloat16().eval()
        reference.norm.weight.fill_(1)
        native = copy.deepcopy(reference)
        original_conv = mh.causal_conv1d_fn
        original_update = mh.causal_conv1d_update
        _native_convolution(native, self.kernels)
        self.assertIs(mh.causal_conv1d_fn, original_conv)
        self.assertIs(mh.causal_conv1d_update, original_update)
        caches = [DynamicCache(config=config), DynamicCache(config=config)]
        for tokens in (5, 1, 1, 1):
            x = torch.randn(1, tokens, 16).bfloat16()
            expected = reference(x, cache_params=caches[0])
            actual = native(x, cache_params=caches[1])
            torch.testing.assert_close(actual, expected, atol=.002, rtol=.008)
            for field in ("conv_states", "recurrent_states"):
                torch.testing.assert_close(getattr(caches[0].layers[0], field)[0],
                    getattr(caches[1].layers[0], field)[0], atol=.002, rtol=.008)

    @torch.inference_mode()
    def test_fused_recurrent_state_production_and_tail_shapes(self):
        from transformers.models.nemotron_h import modeling_nemotron_h as mh
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        torch.manual_seed(491)
        for batches, heads, dim, size, groups in [(1, 128, 64, 128, 8), (2, 6, 7, 19, 3)]:
            config = NemotronHConfig(hidden_size=dim*heads, mamba_num_heads=heads,
                mamba_head_dim=dim, n_groups=groups, ssm_state_size=size,
                layers_block_type=["linear_attention"])
            mixer = mh.NemotronHMamba2Mixer(config, layer_idx=0).bfloat16().eval()
            _native_convolution(mixer, self.kernels)
            native = mixer.forward.__func__.__globals__["mamba2_selective_state_update"]
            reference = mh.mamba2_selective_state_update
            initial = torch.randn(batches, heads, dim, size)*.1
            states = [initial.clone(), initial.clone()]
            A = -torch.rand(heads).add(.1)[:, None, None].expand(-1, dim, size)
            D = torch.randn(heads).bfloat16()[:, None].expand(-1, dim)
            bias = torch.randn(heads).bfloat16()[:, None].expand(-1, dim)
            for _ in range(5):
                x = torch.randn(batches, heads, dim).bfloat16()
                dt = torch.randn(batches, heads).bfloat16()[:, :, None].expand(-1, -1, dim)
                B = torch.randn(batches, groups, size).bfloat16()
                C = torch.randn(batches, groups, size).bfloat16()
                expected = reference(states[0], x, dt, A, B, C, D, dt_bias=bias, dt_softplus=True)
                actual = native(states[1], x, dt, A, B, C, D, dt_bias=bias, dt_softplus=True)
                torch.testing.assert_close(states[1], states[0], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.008)


if __name__ == "__main__":
    unittest.main()
