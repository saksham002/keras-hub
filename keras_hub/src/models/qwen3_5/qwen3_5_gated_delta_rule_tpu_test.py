"""Tests for the TPU-oriented chunked gated delta rule (JAX backend only)."""

import os
from unittest import mock

import keras
import numpy as np
import pytest

from keras_hub.src.tests.test_case import TestCase

pytestmark = pytest.mark.skipif(
    keras.config.backend() != "jax",
    reason = "The TPU gated delta rule is implemented in JAX.",
)

if keras.config.backend() == "jax":
    import jax
    import jax.numpy as jnp

    from keras_hub.src.models.qwen3_5 import qwen3_5_gated_delta_net as gdn
    from keras_hub.src.models.qwen3_5 import (
        qwen3_5_gated_delta_rule_tpu as gdn_tpu,
    )
    from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone


def _recurrence_float64(
    query, key, value, g, beta, initial_state = None, padding_mask = None
):
    """Token-by-token gated delta rule in float64 numpy, written from the
    definition (independent of keras-hub's code):

        S = exp(g) S;  v_old = S^T k;  S += k (beta (v - v_old))^T;  o = S^T q

    with L2-normalized q and k, q scaled by 1 / sqrt(k_dim), and masked
    tokens leaving S unchanged. State layout (B, H, k_dim, v_dim).
    """
    q, k, v, g, beta = (
        np.asarray(x, np.float64) for x in (query, key, value, g, beta)
    )
    q = q / np.sqrt(np.sum(q * q, -1, keepdims = True) + 1e-6)
    k = k / np.sqrt(np.sum(k * k, -1, keepdims = True) + 1e-6)
    q = q / np.sqrt(q.shape[-1])
    batch, seq, heads, k_dim = k.shape
    v_dim = v.shape[-1]
    state = (
        np.zeros((batch, heads, k_dim, v_dim))
        if initial_state is None
        else np.asarray(initial_state, np.float64).copy()
    )
    out = np.zeros((batch, seq, heads, v_dim))
    for b in range(batch):
        for t in range(seq):
            keep = padding_mask is None or bool(padding_mask[b, t])
            for h in range(heads):
                decay = np.exp(g[b, t, h]) if keep else 1.0
                write = beta[b, t, h] if keep else 0.0
                s = state[b, h] * decay
                v_old = s.T @ k[b, t, h]
                s = s + np.outer(k[b, t, h], write * (v[b, t, h] - v_old))
                state[b, h] = s
                out[b, t, h] = s.T @ q[b, t, h]
    return out, state


def _random_inputs(
    seed = 86, batch = 2, seq = 150, heads = 3, k_dim = 16, v_dim = 8
):
    rng = np.random.default_rng(seed)
    return dict(
        query = rng.normal(size = (batch, seq, heads, k_dim)).astype("float32"),
        key = rng.normal(size = (batch, seq, heads, k_dim)).astype("float32"),
        value = rng.normal(size = (batch, seq, heads, v_dim)).astype("float32"),
        g = -rng.uniform(0.0, 0.5, size = (batch, seq, heads)).astype(
            "float32"
        ),
        beta = rng.uniform(0.0, 1.0, size = (batch, seq, heads)).astype(
            "float32"
        ),
        initial_state = rng.normal(size = (batch, heads, k_dim, v_dim)).astype(
            "float32"
        ),
    )


def _stress_inputs(seed = 86, batch = 1, seq = 128, heads = 2, k_dim = 16):
    """Nearly parallel keys, write gates near 1 and no decay: large
    intra-chunk interactions, the hardest case for the inverse."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size = (1, 1, heads, k_dim))
    key = base + 1e-2 * rng.normal(size = (batch, seq, heads, k_dim))
    inputs = _random_inputs(seed, batch, seq, heads, k_dim, 8)
    inputs["key"] = key.astype("float32")
    inputs["g"] = np.zeros((batch, seq, heads), "float32")
    inputs["beta"] = np.full((batch, seq, heads), 0.99, "float32")
    return inputs


def _relative_error(actual, expected):
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


class GatedDeltaRuleTPUTest(TestCase):
    def test_inverse_matches_exact_inverse(self):
        rng = np.random.default_rng(86)
        for size in (1, 2, 4, 8, 16, 32, 64):
            a = np.tril(rng.normal(size = (3, size, size)) * 0.3, k = -1)
            expected = np.linalg.inv(np.eye(size) - a)
            actual = gdn_tpu._inverse_by_masked_doubling(
                jnp.asarray(a, jnp.float32)
            )
            self.assertLess(_relative_error(actual, expected), 1e-5)

    def test_inverse_stress(self):
        # The entries a chunk sees with parallel keys and beta near 1.
        size = 64
        a = -0.99 * np.tril(np.ones((size, size)), k = -1)
        expected = np.linalg.inv(np.eye(size) - a)
        actual = gdn_tpu._inverse_by_masked_doubling(
            jnp.asarray(a, jnp.float32)
        )
        self.assertLess(_relative_error(actual, expected), 1e-5)

    def test_inverse_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            gdn_tpu._inverse_by_masked_doubling(jnp.zeros((48, 48)))

    def test_matches_float64_recurrence(self):
        inputs = _random_inputs()
        mask = np.ones((2, 150), bool)
        mask[0, 140:] = False  # right padding
        mask[1, 70] = False  # a masked token inside a chunk
        expected_out, expected_state = _recurrence_float64(
            **inputs, padding_mask = mask
        )
        out, state = gdn_tpu.chunk_gated_delta_rule_tpu(
            **inputs, output_final_state = True, padding_mask = mask
        )
        self.assertLess(_relative_error(out, expected_out), 1e-5)
        self.assertLess(_relative_error(state, expected_state), 1e-5)

    def test_matches_float64_recurrence_without_state_or_mask(self):
        inputs = _random_inputs(seq = 64)
        inputs.pop("initial_state")
        expected_out, _ = _recurrence_float64(**inputs)
        out, state = gdn_tpu.chunk_gated_delta_rule_tpu(**inputs)
        self.assertIsNone(state)
        self.assertLess(_relative_error(out, expected_out), 1e-5)

    def test_stress_matches_float64_recurrence(self):
        inputs = _stress_inputs()
        expected_out, expected_state = _recurrence_float64(**inputs)
        out, state = gdn_tpu.chunk_gated_delta_rule_tpu(
            **inputs, output_final_state = True
        )
        self.assertLess(_relative_error(out, expected_out), 1e-4)
        self.assertLess(_relative_error(state, expected_state), 1e-4)

    def test_matches_reference_rule_forward_and_gradients(self):
        inputs = {k: jnp.asarray(v) for k, v in _random_inputs().items()}
        mask = np.ones((2, 150), bool)
        mask[0, 120:] = False
        rng = np.random.default_rng(1)
        out_weights = jnp.asarray(rng.normal(size = (2, 150, 3, 8)), "float32")
        state_weights = jnp.asarray(
            rng.normal(size = (2, 3, 16, 8)), "float32"
        )

        def loss(rule, args):
            out, state = rule(
                **args, output_final_state = True, padding_mask = mask
            )
            return jnp.sum(out * out_weights) + jnp.sum(state * state_weights)

        with mock.patch.dict(os.environ, {gdn.GDN_IMPL_ENV_VAR: "reference"}):
            ref_value, ref_grads = jax.value_and_grad(
                lambda args: loss(gdn._chunk_gated_delta_rule, args)
            )(inputs)
        tpu_value, tpu_grads = jax.value_and_grad(
            lambda args: loss(gdn_tpu.chunk_gated_delta_rule_tpu, args)
        )(inputs)
        self.assertLess(_relative_error(tpu_value, ref_value), 1e-5)
        for name in inputs:
            self.assertLess(
                _relative_error(tpu_grads[name], ref_grads[name]), 1e-4, name
            )

    def test_jit_and_bfloat16_inputs(self):
        inputs = _random_inputs(seq = 100)
        rule = jax.jit(
            lambda **kw: gdn_tpu.chunk_gated_delta_rule_tpu(
                **kw, output_final_state = True
            )
        )
        out32, _ = rule(**inputs)
        bf16 = {k: jnp.asarray(v, jnp.bfloat16) for k, v in inputs.items()}
        out16, state16 = rule(**bf16)
        self.assertEqual(out16.dtype, jnp.bfloat16)
        self.assertEqual(state16.dtype, jnp.float32)
        self.assertLess(_relative_error(out16, out32), 5e-2)

    def test_env_var_selects_implementation(self):
        inputs = _random_inputs(seq = 64)
        calls = []
        original = gdn_tpu.chunk_gated_delta_rule_tpu

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(gdn_tpu, "chunk_gated_delta_rule_tpu", spy):
            with mock.patch.dict(os.environ, {gdn.GDN_IMPL_ENV_VAR: "tpu"}):
                tpu_out, _ = gdn._chunk_gated_delta_rule(**inputs)
            self.assertEqual(len(calls), 1)
            with mock.patch.dict(
                os.environ, {gdn.GDN_IMPL_ENV_VAR: "reference"}
            ):
                ref_out, _ = gdn._chunk_gated_delta_rule(**inputs)
            self.assertEqual(len(calls), 1)
            # Unset means the reference implementation.
            with mock.patch.dict(os.environ, {}, clear = True):
                gdn._chunk_gated_delta_rule(**inputs)
            self.assertEqual(len(calls), 1)
        self.assertLess(_relative_error(tpu_out, ref_out), 1e-5)

        with mock.patch.dict(os.environ, {gdn.GDN_IMPL_ENV_VAR: "fast"}):
            with self.assertRaises(ValueError):
                gdn._chunk_gated_delta_rule(**inputs)

    def test_backbone_outputs_and_gradients_match(self):
        keras.utils.set_random_seed(86)
        backbone = Qwen3_5Backbone(
            vocabulary_size = 50,
            num_layers = 4,
            num_query_heads = 2,
            num_key_value_heads = 1,
            head_dim = 8,
            hidden_dim = 16,
            intermediate_dim = 32,
            layer_types = ["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads = 2,
            linear_num_value_heads = 4,
            linear_key_head_dim = 8,
            linear_value_head_dim = 8,
        )
        rng = np.random.default_rng(86)
        token_ids = rng.integers(1, 50, size = (2, 150)).astype("int32")
        padding_mask = np.ones((2, 150), "int32")
        padding_mask[1, 100:] = 0
        inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        trainable = [v.value for v in backbone.trainable_variables]
        non_trainable = [v.value for v in backbone.non_trainable_variables]

        def loss(params):
            out, _ = backbone.stateless_call(params, non_trainable, inputs)
            return jnp.sum(jnp.square(out) * padding_mask[..., None])

        results = {}
        for impl in ("reference", "tpu"):
            with mock.patch.dict(os.environ, {gdn.GDN_IMPL_ENV_VAR: impl}):
                results[impl] = jax.value_and_grad(loss)(trainable)
        ref_value, ref_grads = results["reference"]
        tpu_value, tpu_grads = results["tpu"]
        self.assertLess(_relative_error(tpu_value, ref_value), 1e-5)
        for ref_grad, tpu_grad in zip(ref_grads, tpu_grads):
            if np.linalg.norm(np.asarray(ref_grad)) > 0:
                self.assertLess(_relative_error(tpu_grad, ref_grad), 1e-4)
