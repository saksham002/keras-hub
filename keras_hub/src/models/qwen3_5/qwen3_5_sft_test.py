"""Tests for image + text supervised fine-tuning with Qwen3.5."""

import keras
import numpy as np
import pytest

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm import Qwen3_5CausalLM
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm_preprocessor import (
    Qwen3_5CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5.qwen3_5_image_converter import (
    Qwen3_5ImageConverter,
)
from keras_hub.src.models.qwen3_5.qwen3_5_layers import (
    Qwen3_5InterleaveEmbeddings,
)
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase

VISION = "<|vision_start|><|image_pad|><|vision_end|>"


def _tokenizer():
    merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i"]
    merges += ["p l", "n e", "Ġa t", "p o", "r t", "Ġt h"]
    merges += ["ai r", "pl a", "po rt", "Ġai r", "Ġa i"]
    merges += ["pla ne"]
    vocab = []
    for merge in merges:
        a, b = merge.split(" ")
        vocab.extend([a, b, a + b])
    vocab += ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
    vocab += ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
    vocab += ["<|video_pad|>", "!"]
    vocab = sorted(set(vocab))
    vocab = dict([(token, i) for i, token in enumerate(vocab)])
    return Qwen3_5Tokenizer(vocabulary = vocab, merges = merges)


def _image_converter():
    # 16x16 images -> 4x4 patches of 4px -> 2x2 = 4 tokens after merging.
    return Qwen3_5ImageConverter(
        patch_size = 4,
        temporal_patch_size = 2,
        spatial_merge_size = 2,
        min_pixels = 16 * 16,
        max_pixels = 64 * 64,
        interpolation = "nearest",
    )


def _vision_encoder(fixed_image_size = None):
    return Qwen3_5VisionEncoder(
        depth = 2,
        hidden_size = 16,
        num_heads = 2,
        intermediate_size = 32,
        patch_size = 4,
        temporal_patch_size = 2,
        spatial_merge_size = 2,
        out_hidden_size = 16,
        num_position_embeddings = 16,
        fixed_image_size = fixed_image_size,
    )


class Qwen3_5SFTTest(TestCase):
    def test_image_converter_patch_order(self):
        # Paint each 4x4 patch with its row-major index; the converter must
        # emit patches grouped by 2x2 merge block, as HF's processor does.
        image = np.zeros((16, 16, 3), dtype = "float32")
        for r in range(4):
            for c in range(4):
                image[r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4] = r * 4 + c
        out = _image_converter()(image)
        patches = keras.ops.convert_to_numpy(out["patches"])
        self.assertAllEqual(
            patches[:, 0, 0, 0, 0],
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
        )

    def test_fixed_image_size_matches_dynamic(self):
        dynamic = _vision_encoder()
        fixed = _vision_encoder(fixed_image_size = (16, 16))
        dynamic.build()
        fixed.build()
        fixed.set_weights(dynamic.get_weights())
        rng = np.random.default_rng(86)
        pixel_values = rng.random((2 * 16, 2, 4, 4, 3)).astype("float32")
        grid_thw = np.array([[1, 4, 4], [1, 4, 4]], dtype = "int32")
        expected = dynamic(pixel_values, grid_thw)
        self.assertAllClose(fixed(pixel_values, grid_thw), expected)
        if keras.config.backend() == "jax":
            import jax

            jitted = jax.jit(lambda pv, g: fixed(pv, g))
            self.assertAllClose(jitted(pixel_values, grid_thw), expected)

    def test_interleave_with_per_row_indices(self):
        layer = Qwen3_5InterleaveEmbeddings(hidden_dim = 2)
        text = np.zeros((2, 6, 2), dtype = "float32")
        image = np.arange(1, 9, dtype = "float32").reshape(1, 4, 2)
        out = layer(
            image_embeddings = image,
            text_embeddings = text,
            vision_indices = np.array([[1, 2], [3, 4]], dtype = "int32"),
        )
        out = keras.ops.convert_to_numpy(out)
        self.assertAllEqual(out[0, 1], [1, 2])
        self.assertAllEqual(out[0, 2], [3, 4])
        self.assertAllEqual(out[1, 3], [5, 6])
        self.assertAllEqual(out[1, 4], [7, 8])
        self.assertAllEqual(out[0, 3], [0, 0])

    def test_sft_preprocessing(self):
        preprocessor = Qwen3_5CausalLMPreprocessor(
            tokenizer = _tokenizer(),
            image_converter = _image_converter(),
            sequence_length = 16,
        )
        rng = np.random.default_rng(86)
        images = rng.integers(0, 256, (2, 2, 16, 16, 3)).astype("uint8")
        prompts = [f"{VISION}{VISION} airplane", f"{VISION}{VISION} at"]
        responses = [" airport", " airplane"]
        x, y, sw = preprocessor(
            {"prompts": prompts, "responses": responses, "images": images}
        )
        self.assertEqual(x["token_ids"].shape, (2, 16))
        self.assertEqual(x["position_ids"].shape, (2, 4, 16))
        self.assertEqual(x["pixel_values"].shape, (2, 32, 2, 4, 4, 3))
        self.assertEqual(x["image_grid_thw"].shape, (2, 2, 3))
        self.assertEqual(x["vision_indices"].shape, (2, 8))

        image_id = preprocessor.image_token_id
        for b in range(2):
            self.assertAllEqual(
                x["token_ids"][b][x["vision_indices"][b]], [image_id] * 8
            )
            # Loss only on response tokens and the end token.
            prompt_len = len(
                preprocessor._tokenize_with_special_tokens(
                    prompts[b], [4, 4], []
                )
            )
            weighted = np.where(sw[b] > 0)[0]
            self.assertEqual(weighted[0], prompt_len - 1)
            self.assertAllEqual(
                y[b][weighted[-1]], preprocessor.tokenizer.end_token_id
            )

        # First image (after <|vision_start|> at 0): t = 1, (h, w) in a 2x2
        # grid offset by 1; text resumes at 1 + max(2, 2) = 3.
        self.assertAllEqual(x["position_ids"][0, 1, 1:5], [1, 1, 1, 1])
        self.assertAllEqual(x["position_ids"][0, 2, 1:5], [1, 1, 2, 2])
        self.assertAllEqual(x["position_ids"][0, 3, 1:5], [1, 2, 1, 2])
        self.assertAllEqual(x["position_ids"][0, :, 5], [3, 3, 3, 3])

        # One sample at a time, then stacked, gives the same batch.
        single = preprocessor(
            {
                "prompts": prompts[1],
                "responses": responses[1],
                "images": images[1],
            }
        )
        for key in x:
            self.assertAllEqual(single[0][key], x[key][1])

    @pytest.mark.skipif(
        keras.config.backend() == "tensorflow",
        reason = "The vision encoder branches on the patch count, which is "
        "symbolic in TF graph mode (pre-existing limitation).",
    )
    def test_fit_with_images(self):
        preprocessor = Qwen3_5CausalLMPreprocessor(
            tokenizer = _tokenizer(),
            image_converter = _image_converter(),
            sequence_length = 16,
        )
        backbone = Qwen3_5Backbone(
            vocabulary_size = preprocessor.tokenizer.vocabulary_size(),
            num_layers = 4,
            num_query_heads = 2,
            num_key_value_heads = 1,
            head_dim = 8,
            hidden_dim = 16,
            intermediate_dim = 32,
            layer_types = ["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads = 2,
            linear_num_value_heads = 2,
            linear_key_head_dim = 4,
            linear_value_head_dim = 4,
            mrope_section = [1, 0, 0],
            vision_encoder = _vision_encoder(fixed_image_size = (16, 16)),
        )
        model = Qwen3_5CausalLM(backbone = backbone, preprocessor = None)
        rng = np.random.default_rng(86)
        images = rng.integers(0, 256, (2, 1, 16, 16, 3)).astype("uint8")
        x, y, sw = preprocessor(
            {
                "prompts": [f"{VISION} airplane", f"{VISION} at"],
                "responses": [" airport", " airplane"],
                "images": images,
            }
        )
        model.compile(
            optimizer = keras.optimizers.Adam(1e-2),
            loss = keras.losses.SparseCategoricalCrossentropy(
                from_logits = True
            ),
        )
        history = model.fit(x, y, sample_weight = sw, epochs = 3, verbose = 0)
        losses = history.history["loss"]
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertLess(losses[-1], losses[0])
