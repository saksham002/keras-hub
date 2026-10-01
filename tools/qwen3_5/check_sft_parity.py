# ruff: noqa: E501
"""End-to-end check of the keras-hub Qwen3.5 SFT path against HF transformers.

Builds a tiny random Qwen3.5 VLM (real vocabulary size and special-token ids)
in transformers, loads it through keras-hub's HF converter, and uses the real
Qwen3.5 tokenizer and image processor. For two samples with two 256x256 camera
images each it checks, against HF:

- token ids, pixel values and M-RoPE position ids from the
  `{"prompts", "responses", "images"}` preprocessing path,
- that the loss mask covers only the response,
- logits of the functional model run under jit with `fixed_image_size`,
- and that a few `fit()` steps run under JAX with a decreasing loss.

Requires `transformers>=5` and `torch`; downloads only the tokenizer and
preprocessor files of `Qwen/Qwen3.5-0.8B`.

Usage:
    KERAS_BACKEND=jax python tools/qwen3_5/check_sft_parity.py
"""

import json
import os
import tempfile

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import torch
from transformers import AutoProcessor
from transformers import Qwen3_5Config
from transformers import Qwen3_5ForConditionalGeneration

import keras_hub

HF_ID = "Qwen/Qwen3.5-0.8B"
IMG = 256
SEQ_LEN = 256


def tiny_hf_model(seed = 86):
    torch.manual_seed(seed)
    config = Qwen3_5Config(
        text_config = dict(
            vocab_size = 248320,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 32,
            linear_num_key_heads = 2,
            linear_num_value_heads = 4,
            linear_key_head_dim = 16,
            linear_value_head_dim = 16,
            linear_conv_kernel_dim = 4,
            layer_types = ["linear_attention"] * 3 + ["full_attention"],
            rope_parameters = dict(
                rope_type = "default",
                rope_theta = 1e7,
                partial_rotary_factor = 0.25,
                mrope_section = [2, 1, 1],
                mrope_interleaved = True,
            ),
            tie_word_embeddings = True,
        ),
        vision_config = dict(
            depth = 2,
            hidden_size = 64,
            num_heads = 4,
            intermediate_size = 128,
            out_hidden_size = 64,
            patch_size = 16,
            temporal_patch_size = 2,
            spatial_merge_size = 2,
            num_position_embeddings = 2304,
            deepstack_visual_indexes = [],
        ),
        image_token_id = 248056,
        video_token_id = 248057,
        vision_start_token_id = 248053,
        vision_end_token_id = 248054,
        tie_word_embeddings = True,
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return model


def report(name, a, b, tol):
    err = float(np.abs(np.asarray(a, "float64") - np.asarray(b, "float64")).max())
    status = "OK " if err <= tol else "FAIL"
    print(f"[{status}] {name}: max|diff| = {err:.3e} (tol {tol})")
    return err <= tol


def main():
    ok = True
    hf = tiny_hf_model()
    hf_proc = AutoProcessor.from_pretrained(HF_ID)
    tmp = tempfile.mkdtemp()
    hf.save_pretrained(tmp)
    with open(os.path.join(tmp, "preprocessor_config.json"), "w") as f:
        json.dump(hf_proc.image_processor.to_dict(), f)

    rng = np.random.default_rng(86)
    images = rng.integers(0, 256, size = (2, 2, IMG, IMG, 3), dtype = np.uint8)
    vis = "<|vision_start|><|image_pad|><|vision_end|>"
    prompts = [
        f"<|im_start|>user\n{vis}{vis}Pick up the red block.<|im_end|>\n<|im_start|>assistant\n",
        f"<|im_start|>user\n{vis}{vis}Open the drawer.<|im_end|>\n<|im_start|>assistant\n",
    ]
    responses = ["512 498 501 499 500 512 1000<|im_end|>", "7 0 999 250 500 750 1<|im_end|>"]

    # === Keras preprocessing (new SFT path) ===
    pre = keras_hub.models.Qwen3_5CausalLMPreprocessor.from_preset(
        f"hf://{HF_ID}", sequence_length = SEQ_LEN, add_end_token = False
    )
    x, y, sw = pre({"prompts": prompts, "responses": responses, "images": images})
    print({k: (v.shape, v.dtype) for k, v in x.items()})

    # Per-sample (unbatched) calls stacked afterwards must give the same arrays.
    singles = [pre({"prompts": prompts[b], "responses": responses[b], "images": images[b]}) for b in range(2)]
    for k in x:
        ok &= report(f"unbatched-then-stacked == batched [{k}]", np.stack([s[0][k] for s in singles]), x[k], 0)

    # === HF reference ===
    hf_all = []
    for b in range(2):
        hf_in = hf_proc(text = [prompts[b] + responses[b]], images = list(images[b]), return_tensors = "pt")
        n = hf_in["input_ids"].shape[1]
        ok &= report(f"sample {b} token_ids vs HF", x["token_ids"][b, :n], hf_in["input_ids"][0].numpy(), 0)
        ok &= report(f"sample {b} padding_mask covers HF length", x["padding_mask"][b].sum(), n, 0)
        ok &= report(f"sample {b} pixel_values vs HF",
                     x["pixel_values"][b].transpose(0, 4, 1, 2, 3).reshape(x["pixel_values"].shape[1], -1),
                     hf_in["pixel_values"].numpy(), 1e-5)
        mm = hf_in["mm_token_type_ids"] if "mm_token_type_ids" in hf_in else (hf_in["input_ids"] == 248056).int()
        pos, _ = hf.model.get_rope_index(hf_in["input_ids"], mm, image_grid_thw = hf_in["image_grid_thw"])
        kh_pos = x["position_ids"][b][:, :n]
        ok &= report(f"sample {b} position_ids [t,h,w] vs HF", kh_pos[1:], pos[:, 0].numpy(), 0)
        with torch.no_grad():
            hf_all.append(hf(**hf_in).logits[0].numpy())
        resp_start = len(pre._tokenize_with_special_tokens(prompts[b], [64, 64], []))
        ok &= report(f"sample {b} sample_weight on response only",
                     sw[b][: n - 1], (np.arange(1, n) >= resp_start).astype("float32"), 0)

    # === Keras model: functional graph under jit, fixed image size ===
    lm = keras_hub.models.Qwen3_5CausalLM(
        backbone = keras_hub.models.Qwen3_5Backbone.from_preset(tmp), preprocessor = None
    )
    lm.backbone.vision_encoder.fixed_image_size = (IMG, IMG)
    kh_logits = lm.predict(x, batch_size = 2, verbose = 0)  # predict() runs jitted
    for b in range(2):
        n = hf_all[b].shape[0]
        ok &= report(f"sample {b} logits (jit, fixed_image_size) vs HF", kh_logits[b, :n], hf_all[b], 5e-5)

    # Without position_ids the full-attention layers fall back to 1D RoPE: logits must differ.
    x_no_pos = {k: v for k, v in x.items() if k != "position_ids"}
    kh_no_pos = lm.predict(x_no_pos, batch_size = 2, verbose = 0)
    diff = np.abs(kh_no_pos[0, : hf_all[0].shape[0]] - hf_all[0]).max()
    print(f"[info] logits without position_ids differ from HF by {diff:.3e} (expected > 0)")

    # === Training: a few fit() steps under JAX jit ===
    lm.compile(
        optimizer = keras.optimizers.Adam(1e-3),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        weighted_metrics = [],
    )
    hist = lm.fit(x, y, sample_weight = sw, batch_size = 2, epochs = 5, verbose = 0)
    losses = hist.history["loss"]
    print("[info] fit losses:", [round(v, 4) for v in losses])
    ok &= bool(np.all(np.isfinite(losses)) and losses[-1] < losses[0])
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
