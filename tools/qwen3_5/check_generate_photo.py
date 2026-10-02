# ruff: noqa: E501
"""Greedy image + text generation with Qwen3.5-0.8B: keras-hub vs HF.

Downloads a COCO validation photo (two cats on a couch with two remotes),
asks Qwen/Qwen3.5-0.8B about it with HF's chat template (thinking off), and
decodes greedily with HF transformers and with keras-hub's `generate()` on the
JAX backend (`fixed_image_size` set to the photo size). The two texts should be
identical. Requires `transformers>=5`, `torch` and `pillow`.

Usage:
    KERAS_BACKEND=jax python tools/qwen3_5/check_generate_photo.py
"""

import io
import os
import sys
import time
import urllib.request

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
from PIL import Image

HF_ID = "Qwen/Qwen3.5-0.8B"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
QUESTION = "Describe this photo. How many cats are there, and what objects are lying next to them?"
MAX_NEW = 80


def build_prompt(processor):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    return processor.apply_chat_template(
        messages, add_generation_prompt = True, tokenize = False, enable_thinking = False
    )


def run_hf(image):
    import torch
    from transformers import AutoProcessor
    from transformers import Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(HF_ID)
    prompt = build_prompt(processor)
    inputs = processor(text = [prompt], images = [image], return_tensors = "pt")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(HF_ID, dtype = torch.float32).eval()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens = MAX_NEW, do_sample = False)
    new = out[0, inputs["input_ids"].shape[1]:]
    return prompt, inputs["input_ids"].shape[1], new.tolist(), processor.decode(new, skip_special_tokens = True)


def run_keras(prompt, prompt_len, image):
    import keras_hub

    lm = keras_hub.models.Qwen3_5CausalLM.from_preset(f"hf://{HF_ID}", dtype = "float32")
    lm.backbone.vision_encoder.fixed_image_size = (image.shape[0], image.shape[1])
    lm.compile(sampler = "greedy")
    t0 = time.time()
    out = lm.generate({"prompts": prompt, "images": image}, max_length = prompt_len + MAX_NEW, strip_prompt = True)
    return out, time.time() - t0


def main():
    image = np.array(Image.open(io.BytesIO(urllib.request.urlopen(IMAGE_URL).read())).convert("RGB"))
    print("image", image.shape)
    prompt, prompt_len, hf_ids, hf_text = run_hf(image)
    print("prompt:", repr(prompt))
    print("prompt tokens (HF):", prompt_len)
    print("\n=== HF greedy ===\n", hf_text)
    sys.stdout.flush()
    kh_text, secs = run_keras(prompt, prompt_len, image)
    print(f"\n=== keras-hub fork (JAX) greedy, {secs:.0f}s ===\n", kh_text)
    kh_text = kh_text[0] if isinstance(kh_text, list) else kh_text
    print("\nidentical text:", kh_text.strip() == hf_text.strip())


if __name__ == "__main__":
    main()
