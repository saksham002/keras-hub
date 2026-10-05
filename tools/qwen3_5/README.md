# Qwen3.5 image + text SFT (fork notes)

This fork makes `Qwen3_5CausalLM` trainable on image + text data with `fit()`
on the JAX backend (and therefore on TPU), with outputs that match HF
transformers.

## What changed

| File | Change |
| --- | --- |
| `qwen3_5_image_converter.py`, `qwen3_5_video_converter.py` | **Bug fix.** Patches were emitted row by row; the vision encoder and HF expect them grouped by 2x2 merge block. Images fed through keras-hub previously reached the model scrambled. |
| `qwen3_5_backbone.py` | New optional `position_ids` input `(batch, 4, seq_len)`, passed to every decoder layer. Previously the training graph never received M-RoPE positions and silently used 1D RoPE for image tokens. When omitted, 1D RoPE is used (exact for text-only). |
| `qwen3_5_causal_lm.py` | `__call__`/`call` fill in omitted optional inputs, so text-only batches keep working (the JAX trainer calls `call` directly). |
| `qwen3_5_vision_encoder.py` | New `fixed_image_size=(H, W)`: position tables and per-image attention windows are built from static shapes instead of reading `grid_thw` values, so the encoder can be traced by `jit`. |
| `qwen3_5_layers.py` | A 2D `vision_indices` `(batch, n)` now holds positions within each row (the row offset is added in the layer), so samples can be preprocessed one at a time and stacked. |
| `qwen3_5_causal_lm_preprocessor.py` | New training path for `{"prompts", "responses", "images"}`: returns `(x, y, sample_weight)` with `position_ids`, `pixel_values`, `image_grid_thw`, `vision_indices`, and loss only on the response. Pure numpy; no TensorFlow needed for images. |
| `qwen3_5_gated_delta_net.py` | **Bug fix.** The delta rule accepted `padding_mask` but ignored it, so padding tokens still decayed (and, through the short convolution, wrote into) the recurrent state. Masked tokens are now no-ops (`beta = 0`, `g = 0`). Affects text-only generation too. |
| `qwen3_5_causal_lm.py` (`generate_step`) | **Bug fixes.** (1) The prefill consumed the whole prompt and the first decode step re-fed the last prompt token, processing it twice in the cumulative linear-attention state; the prefill now stops before it. (2) Decode steps after an image now add the M-RoPE position offset (HF's `rope_deltas`) instead of using the raw token index. |
| `qwen3_5_sft_test.py` | Unit tests for all of the above, including cached greedy decoding vs. an uncached forward pass. |
| `tools/qwen3_5/check_sft_parity.py` | End-to-end check against HF (tokens, pixels, positions, logits under jit, `fit()`). |
| `tools/qwen3_5/check_generate_photo.py` | Greedy image + text generation on a real photo with Qwen3.5-0.8B, keras-hub vs HF. |

## TPU-oriented gated delta rule

Set `KERAS_HUB_QWEN3_5_GDN_IMPL` to choose the chunked gated delta rule used by
the linear-attention layers (training forward pass and the decode prefill):

| Value | Implementation |
| --- | --- |
| `reference` (default, or unset) | The backend-agnostic rule in `qwen3_5_gated_delta_net.py`: each chunk's `(I - A)^-1` by a 63-step row loop, chunks as an unrolled Python loop. |
| `tpu` (JAX only) | `qwen3_5_gated_delta_rule_tpu.py`: the inverse by six levels of masked block doubling (two full chunk x chunk matmuls per level), all chunks' intra-chunk work batched, and only the chunk-to-chunk state recurrence as a `lax.scan`. |

```bash
KERAS_HUB_QWEN3_5_GDN_IMPL=tpu python train.py ...
```

The variable is read when the layer is traced, so set it before the first
compiled call. Neither implementation sets matmul precision; both follow JAX's
global setting (e.g. `JAX_DEFAULT_MATMUL_PRECISION=highest` for full float32
on TPU, whose default is a single bfloat16 pass). Same math as the reference: the tests in
`qwen3_5_gated_delta_rule_tpu_test.py` check it against an independent float64
token-by-token recurrence (including stress inputs), against the reference rule
(forward and gradients), and a backbone's outputs and parameter gradients under
both settings. Ported from the `gdn_fast_doubling` latency patch
(`vla_archs_latency`, report of 2026-10-05: 0.984 s to 0.696 s per train step
on a v6e-8 and 464 s to 95 s compile, measured there, not in this repo).

## Verified so far (JAX backend, CPU, float32)

- Tiny random Qwen3.5 VLM (2 samples x 2 cameras, random-noise images): token
  ids, pixel values and M-RoPE positions identical to HF; logits of the
  jitted functional model within ~1e-6; `fit()` runs with a decreasing loss
  (overfitting one batch; loss values and gradients were not compared to HF).
- `Qwen/Qwen3.5-0.8B` real weights, one image + text sample: logits within
  2.9e-3 (max |logit| 35.5), identical argmax on all response positions.
- `Qwen/Qwen3.5-0.8B`, real COCO photo + question, greedy `generate()` for 80
  tokens: text identical to HF's greedy output.
- Unit tests pass on JAX and TensorFlow; on torch one pre-existing video test
  fails on macOS (MPS has no float64), unrelated to these changes.

Not verified: the 2B/4B checkpoints, TPU, bfloat16, training loss/gradients
against HF, batched image generation with different prompt lengths, and
image `fit()` on the TensorFlow backend (pre-existing graph-mode limitation).

## Usage

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import keras_hub

preset = "hf://Qwen/Qwen3.5-2B"  # or "qwen3_5_2b" / "qwen3_5_4b" from Kaggle
preprocessor = keras_hub.models.Qwen3_5CausalLMPreprocessor.from_preset(
    preset, sequence_length = 512
)
lm = keras_hub.models.Qwen3_5CausalLM.from_preset(preset, preprocessor = None)
lm.backbone.vision_encoder.fixed_image_size = (256, 256)

vision = "<|vision_start|><|image_pad|><|vision_end|>"
x, y, sample_weight = preprocessor({
    "prompts": [f"<|im_start|>user\n{vision}{vision}Pick up the block.<|im_end|>\n<|im_start|>assistant\n"],
    "responses": ["512 498 501 499 500 512 1000<|im_end|>"],
    "images": images,  # (batch, num_cameras, 256, 256, 3), uint8
})

lm.compile(
    optimizer = keras.optimizers.AdamW(5e-6),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
)
lm.fit(x, y, sample_weight = sample_weight, batch_size = 8)
```

Notes:

- The preprocessing path runs eagerly in Python (not inside `tf.data`). Call it
  per sample in a Grain / Python loader and stack, or on batches where every
  sample has the same number and size of images.
- `fixed_image_size` must match the size the image converter produces. Images
  of 256x256 (the processor's minimum pixel count) pass through unchanged.
- Keras's default loss reduction divides the weighted sum by *all* positions,
  not by the number of response tokens, so the effective loss scale depends on
  the prompt/response length ratio.
