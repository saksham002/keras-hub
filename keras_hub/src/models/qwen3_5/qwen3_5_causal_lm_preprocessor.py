import re

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_image_converter import (
    Qwen3_5ImageConverter,
)
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer
from keras_hub.src.models.qwen3_5.qwen3_5_video_converter import (
    Qwen3_5VideoConverter,
)
from keras_hub.src.utils.tensor_utils import assert_tf_installed
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Qwen3_5CausalLMPreprocessor")
class Qwen3_5CausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen3.5 Causal LM preprocessor with multimodal support.

    For text-only usage this behaves identically to the base
    ``CausalLMPreprocessor``.  When an ``image_converter`` is provided,
    the preprocessor also:

    1. Converts images to patch tensors via ``Qwen3_5ImageConverter``.
    2. Replaces ``<|image_pad|>`` and ``<|video_pad|>`` placeholder tokens
       in the token sequence with the correct number of vision tokens.
    3. Computes flat ``vision_indices`` for scattering visual embeddings
       into the text sequence.
    4. Builds 4-channel M-RoPE ``position_ids`` for spatial awareness.

    Args:
        tokenizer: A ``Qwen3_5Tokenizer`` instance. Vision special token
            IDs (``image_token``, ``video_token``, etc.) are resolved
            from the tokenizer's vocabulary automatically.
        image_converter: A ``Qwen3_5ImageConverter`` instance, or ``None``
            for text-only mode.
        video_converter: A ``Qwen3_5VideoConverter`` instance, or ``None``.
        sequence_length: int. Total padded sequence length. Default 1024.
        add_start_token: bool. Prepend BOS token. Default ``False``.
        add_end_token: bool. Append EOS token. Default ``True``.
        video_fps: float. Default video sampling rate for timestamp
            computation. Default ``2.0``.
    """

    backbone_cls = Qwen3_5Backbone
    tokenizer_cls = Qwen3_5Tokenizer
    image_converter_cls = Qwen3_5ImageConverter
    video_converter_cls = Qwen3_5VideoConverter

    _SPECIAL_TOKEN_ATTRS = [
        "im_start_token",
        "end_token",
        "vision_start_token",
        "vision_end_token",
        "image_token",
        "video_token",
    ]

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=True,
        video_fps=2.0,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter
        self.video_converter = video_converter
        self.video_fps = video_fps

        # Token strings — these are static.
        self.image_token = getattr(
            self.tokenizer, "image_token", "<|image_pad|>"
        )
        self.video_token = getattr(
            self.tokenizer, "video_token", "<|video_pad|>"
        )

        # Lazily built after the tokenizer's vocabulary is loaded.
        self._cached_special_token_map = None
        self._cached_special_token_pattern = None

    @property
    def image_token_id(self):
        """Image pad token ID, resolved from the tokenizer."""
        return getattr(self.tokenizer, "image_token_id", None)

    @property
    def video_token_id(self):
        """Video pad token ID, resolved from the tokenizer."""
        return getattr(self.tokenizer, "video_token_id", None)

    @property
    def _special_token_map(self):
        """Lazily build token-string → token-ID map."""
        if self._cached_special_token_map is None:
            self._cached_special_token_map = {}
            for attr in self._SPECIAL_TOKEN_ATTRS:
                tok_str = getattr(self.tokenizer, attr, None)
                tok_id = getattr(self.tokenizer, f"{attr}_id", None)
                if tok_str is not None and tok_id is not None:
                    self._cached_special_token_map[tok_str] = tok_id
        return self._cached_special_token_map

    @property
    def _special_token_pattern(self):
        """Lazily build regex for splitting at special tokens."""
        if self._cached_special_token_pattern is None:
            self._cached_special_token_pattern = re.compile(
                "("
                + "|".join(re.escape(t) for t in self._special_token_map)
                + ")"
            )
        return self._cached_special_token_pattern

    def _tokenize_with_special_tokens(
        self, text, num_image_tokens, num_video_tokens
    ):
        """Tokenize text while correctly handling special tokens.

        The KerasHub BPE tokenizer may not encode Qwen3.5's added special
        tokens (``<|image_pad|>``, ``<|vision_start|>``, etc.) as single
        tokens — it can break them into sub-word pieces. This method
        splits the input by known special tokens, tokenizes only the
        text segments, and manually inserts the correct token IDs.

        For ``<|image_pad|>`` and ``<|video_pad|>`` tokens, each occurrence is
        expanded to ``N`` copies.

        Args:
            text: str. The prompt string.
            num_image_tokens: list[int].
            num_video_tokens: list[int].
        Returns:
            list[int]. The complete token ID sequence.
        """
        parts = self._special_token_pattern.split(text)

        all_ids = []
        img_idx = 0
        vid_idx = 0
        for part in parts:
            if part in self._special_token_map:
                if part == self.image_token:
                    # Expand image placeholder to N copies.
                    if img_idx < len(num_image_tokens):
                        n = num_image_tokens[img_idx]
                        img_idx += 1
                    else:
                        n = 1
                    all_ids.extend([self.image_token_id] * n)
                elif part == self.video_token:
                    if vid_idx < len(num_video_tokens):
                        n = num_video_tokens[vid_idx]
                        vid_idx += 1
                    else:
                        n = 1
                    all_ids.extend([self.video_token_id] * n)
                else:
                    all_ids.append(self._special_token_map[part])
            elif part:
                tokenized = self.tokenizer(part)
                if hasattr(tokenized, "numpy"):
                    all_ids.extend(tokenized.numpy().tolist())
                else:
                    all_ids.extend(list(tokenized))
        return all_ids

    def _compute_vision_indices(self, token_ids):
        """Return indices where token_ids matches image or video token IDs.

        Indices are strictly ordered: all image token indices followed by all
        video indices. This matches the concatenated order of `pixel_values`.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
        Returns:
            int32 tensor ``(total_vision_tokens,)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)
        img_mask = (token_ids_np == self.image_token_id).reshape(-1)
        img_indices = np.where(img_mask)[0].astype(np.int32)

        vid_mask = (token_ids_np == self.video_token_id).reshape(-1)
        vid_indices = np.where(vid_mask)[0].astype(np.int32)

        return tf.constant(np.concatenate([img_indices, vid_indices], axis=0))

    @staticmethod
    def _calculate_timestamps(indices, video_fps, merge_size=2):
        """Compute per-temporal-patch timestamps from frame indices.

        Matches HF's ``Qwen3VLProcessor._calculate_timestamps``
        exactly: timestamps are the average real time (in seconds)
        of the first and last frame within each temporal patch.

        Args:
            indices: list[int]. Raw frame indices from the
                source video.
            video_fps: float. Original video frame rate.
            merge_size: int. Temporal patch size (frames per
                patch).
        Returns:
            list[float]. One timestamp per temporal patch.
        """
        if not isinstance(indices, list):
            indices = list(indices)
        # Pad to a multiple of merge_size.
        while len(indices) % merge_size != 0:
            indices.append(indices[-1])
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def _expand_video_prompt(
        self,
        prompt,
        video_grid_thws,
        temporal_patch_size=2,
        video_metadata=None,
    ):
        """Expand ``<|vision_start|><|video_pad|><|vision_end|>``
        into per-frame sections with timestamps.

        Produces the same token structure as HF's processor::

            <0.5 seconds><|vision_start|><|video_pad|>×N<|vision_end|>
            <1.0 seconds><|vision_start|><|video_pad|>×N<|vision_end|>

        Args:
            prompt: str. The raw prompt string.
            video_grid_thws: list of (T, H, W) tuples.
            temporal_patch_size: int.
            video_metadata: optional list of dicts with
                ``frames_indices`` (list[int]) and ``fps``
                (float) per video. When provided, timestamps
                are computed from real video metadata matching
                HF exactly. Otherwise a fallback formula is
                used.
        Returns:
            tuple of (expanded_prompt,
            num_video_tokens_per_frame).
        """
        merge_size = getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )
        video_marker = "<|vision_start|><|video_pad|><|vision_end|>"
        num_video_tokens_per_frame = []

        for vid_i, grid_thw in enumerate(video_grid_thws):
            t_grid = int(grid_thw[0])
            h_grid = int(grid_thw[1])
            w_grid = int(grid_thw[2])
            frame_seqlen = (h_grid // merge_size) * (w_grid // merge_size)

            # Compute timestamps.
            if (
                video_metadata
                and vid_i < len(video_metadata)
                and video_metadata[vid_i] is not None
            ):
                meta = video_metadata[vid_i]
                timestamps = self._calculate_timestamps(
                    meta["frames_indices"],
                    meta["fps"],
                    temporal_patch_size,
                )
            else:
                # Fallback: evenly spaced at self.video_fps.
                n_raw = t_grid * temporal_patch_size
                indices = list(range(n_raw))
                timestamps = self._calculate_timestamps(
                    indices, self.video_fps, temporal_patch_size
                )

            # Build per-frame block.
            video_block = ""
            for frame_idx in range(t_grid):
                ts = timestamps[frame_idx]
                video_block += f"<{ts:.1f} seconds>"
                video_block += "<|vision_start|><|video_pad|><|vision_end|>"
                num_video_tokens_per_frame.append(frame_seqlen)

            prompt = prompt.replace(video_marker, video_block, 1)

        return prompt, num_video_tokens_per_frame

    def _compute_position_ids(self, token_ids, image_grid_thw, video_grid_thw):
        """Tensor-returning wrapper around ``_compute_position_ids_np``."""
        return tf.constant(
            self._compute_position_ids_np(
                token_ids, image_grid_thw, video_grid_thw
            ),
            dtype = "int32",
        )

    def _compute_position_ids_np(
        self, token_ids, image_grid_thw, video_grid_thw
    ):
        """Build 4-channel M-RoPE position IDs matching HF's algorithm.

        For text tokens all 4 channels have the same sequential position.
        For vision tokens channels 1-3 encode (temporal, height, width)
        grid coordinates. Channel 0 mirrors channel 1 (temporal).

        This matches HF's ``get_rope_index`` / ``get_vision_position_ids``:
        - temporal: ``full(n_tokens, start_pos)``
        - height: ``arange(start, start + h_m).repeat_interleave(w_m * t_eff)``
        - width: ``arange(start, start + w_m).repeat(h_m * t_eff)``
        - text_pos advances by ``max(H, W) // merge_size`` after vision span.

        For **video**, grids are kept as full ``[T, H, W]`` entries (no
        per-frame splitting). The method auto-detects whether video tokens
        are contiguous (HF format) or split per-frame with timestamps
        (KerasHub format) by counting consecutive video tokens.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
            image_grid_thw: int32 tensor ``(num_images, 3)``.
            video_grid_thw: int32 tensor ``(num_videos, 3)``.
        Returns:
            int32 numpy array ``(batch, 4, seq_len)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)
        if hasattr(image_grid_thw, "numpy"):
            image_grid_np = ops.convert_to_numpy(image_grid_thw)
        elif image_grid_thw is not None:
            image_grid_np = np.array(image_grid_thw)
        else:
            image_grid_np = np.zeros((0, 3), dtype=np.int32)

        if hasattr(video_grid_thw, "numpy"):
            video_grid_np = ops.convert_to_numpy(video_grid_thw)
        elif video_grid_thw is not None:
            video_grid_np = np.array(video_grid_thw)
        else:
            video_grid_np = np.zeros((0, 3), dtype=np.int32)

        # Video grids use the full [T, H, W] directly
        # matching HF's get_rope_index.

        batch_size, seq_len = token_ids_np.shape
        merge_size = getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )

        all_pos = np.zeros((batch_size, 4, seq_len), dtype=np.int32)

        for b in range(batch_size):
            ids = token_ids_np[b]
            t_pos = np.zeros(seq_len, dtype=np.int32)
            h_pos = np.zeros(seq_len, dtype=np.int32)
            w_pos = np.zeros(seq_len, dtype=np.int32)

            current_pos = 0
            img_idx = 0
            vid_idx = 0
            # Track how many frames have been consumed from the current
            # video grid entry (for KerasHub per-frame token format).
            vid_frames_consumed = 0
            i = 0
            while i < seq_len:
                is_image = (
                    ids[i] == self.image_token_id
                    and img_idx < image_grid_np.shape[0]
                )
                is_video = (
                    ids[i] == self.video_token_id
                    and vid_idx < video_grid_np.shape[0]
                )

                if is_image or is_video:
                    if is_image:
                        t_grid = int(image_grid_np[img_idx, 0])
                        h_grid = int(image_grid_np[img_idx, 1])
                        w_grid = int(image_grid_np[img_idx, 2])
                        img_idx += 1
                    else:
                        t_grid = int(video_grid_np[vid_idx, 0])
                        h_grid = int(video_grid_np[vid_idx, 1])
                        w_grid = int(video_grid_np[vid_idx, 2])

                    llm_grid_h = h_grid // merge_size
                    llm_grid_w = w_grid // merge_size
                    frame_tokens = llm_grid_h * llm_grid_w

                    # Count consecutive vision tokens of the same type.
                    tok_id = ids[i]
                    j = i
                    while j < seq_len and ids[j] == tok_id:
                        j += 1
                    n_consecutive = j - i

                    if is_video:
                        full_tokens = t_grid * frame_tokens
                        remaining = (
                            full_tokens - vid_frames_consumed * frame_tokens
                        )
                        # t_eff: temporal extent of this contiguous block.
                        t_eff = min(n_consecutive, remaining) // frame_tokens
                        t_eff = max(t_eff, 1)
                        n_tokens = t_eff * frame_tokens
                    else:
                        # Images always have T=1 in practice.
                        t_eff = t_grid
                        n_tokens = t_eff * frame_tokens

                    span_end = min(i + n_tokens, seq_len)
                    actual_n = span_end - i

                    for vi in range(actual_n):
                        t_pos[i + vi] = current_pos
                        h_idx = vi // (llm_grid_w * t_eff)
                        w_idx = vi % llm_grid_w
                        h_pos[i + vi] = current_pos + h_idx
                        w_pos[i + vi] = current_pos + w_idx

                    # Advance by max(h_merged, w_merged) — HF convention.
                    current_pos += max(llm_grid_h, llm_grid_w)
                    i = span_end

                    # Track video grid consumption.
                    if is_video:
                        vid_frames_consumed += t_eff
                        if vid_frames_consumed >= t_grid:
                            vid_idx += 1
                            vid_frames_consumed = 0
                else:
                    t_pos[i] = current_pos
                    h_pos[i] = current_pos
                    w_pos[i] = current_pos
                    current_pos += 1
                    i += 1

            # Channel layout: [text, temporal, height, width].
            all_pos[b, 0] = t_pos
            all_pos[b, 1] = t_pos
            all_pos[b, 2] = h_pos
            all_pos[b, 3] = w_pos

        return all_pos

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        """Preprocess a training batch.

        Plain strings take the base text-only path (loss on every token).

        A dict with ``"prompts"`` and ``"responses"`` (and, for a
        multimodal model, optional ``"images"``) is turned into supervised
        fine-tuning inputs with the loss on the response tokens only:

        - ``"prompts"``: str or list of str. Each image is referenced by a
          ``<|vision_start|><|image_pad|><|vision_end|>`` placeholder, in
          the same order as the images.
        - ``"responses"``: str or list of str, the target text.
        - ``"images"``: for a single sample, a ``(num_images, H, W, 3)``
          array or a list of ``(H, W, 3)`` images; for a batch, a
          ``(batch, num_images, H, W, 3)`` array or a list (one entry per
          sample) of those. Pixel values in ``[0, 255]``.

        Returns ``(x, y, sample_weight)`` where ``x`` holds ``token_ids``,
        ``padding_mask`` and ``position_ids`` (M-RoPE, ``(4, seq_len)`` per
        sample), plus ``pixel_values``, ``image_grid_thw`` and
        ``vision_indices`` (positions within the sample's sequence) for a
        multimodal model. ``y`` is the next token and ``sample_weight`` is
        1 on response tokens (and the end token) and 0 elsewhere.

        This path runs eagerly in Python (it cannot be traced inside
        ``tf.data``). Preprocess samples one at a time and stack them, or
        pass batches in which every sample has the same number and size of
        images. Use the result with ``Qwen3_5CausalLM(..., preprocessor=
        None)`` or ``model.fit(x, y, sample_weight=...)`` on the arrays.
        """
        if not (isinstance(x, dict) and "responses" in x):
            return super().call(
                x,
                y = y,
                sample_weight = sample_weight,
                sequence_length = sequence_length,
            )
        if in_tf_function():
            raise ValueError(
                "`Qwen3_5CausalLMPreprocessor` with `prompts`/`responses` "
                "inputs runs eagerly and cannot be traced inside `tf.data` "
                "or a `tf.function`. Call it in Python (e.g. in a Grain or "
                "plain Python data loader) and feed the resulting arrays."
            )
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length

        prompts, batched = self._as_string_list(x["prompts"])
        responses, _ = self._as_string_list(x["responses"])
        if len(prompts) != len(responses):
            raise ValueError(
                f"Got {len(prompts)} prompts but {len(responses)} responses."
            )
        images = self._as_per_sample_images(
            x.get("images", None), len(prompts), batched
        )
        if images is not None and self.image_converter is None:
            raise ValueError(
                "`images` were passed, but this preprocessor has no "
                "`image_converter` (text-only model)."
            )

        samples = [
            self._sft_sample(
                prompts[b],
                responses[b],
                images[b] if images is not None else [],
                sequence_length,
            )
            for b in range(len(prompts))
        ]

        token_ids = np.stack([s["token_ids"] for s in samples])
        padding_mask = np.stack([s["padding_mask"] for s in samples])
        loss_mask = np.stack([s["loss_mask"] for s in samples])
        position_ids = np.stack([s["position_ids"] for s in samples])

        # The last token has no next token, so it is dropped from `x`.
        x_out = {
            "token_ids": token_ids[:, :-1],
            "padding_mask": padding_mask[:, :-1],
            "position_ids": position_ids,
        }
        y_out = token_ids[:, 1:]
        sample_weight_out = loss_mask[:, 1:].astype("float32")

        if self.image_converter is not None:
            for key in ("pixel_values", "image_grid_thw", "vision_indices"):
                shapes = {s[key].shape for s in samples}
                if len(shapes) > 1:
                    raise ValueError(
                        f"Samples in a batch produced `{key}` of different "
                        f"shapes {sorted(shapes)}. Every sample in a batch "
                        "needs the same number and size of images. "
                        "Preprocess samples one at a time instead."
                    )
                x_out[key] = np.stack([s[key] for s in samples])

        if not batched:
            x_out = {k: v[0] for k, v in x_out.items()}
            y_out = y_out[0]
            sample_weight_out = sample_weight_out[0]
        return keras.utils.pack_x_y_sample_weight(
            x_out, y_out, sample_weight_out
        )

    def _sft_sample(self, prompt, response, sample_images, sequence_length):
        """Tokenize, pack and build vision inputs for one SFT sample.

        Returns numpy arrays of length ``sequence_length + 1`` for the token
        fields (one extra token for the shift into ``y``) and
        ``sequence_length`` for ``position_ids``.
        """
        num_placeholders = prompt.count(self.image_token)
        if num_placeholders != len(sample_images):
            raise ValueError(
                f"The prompt has {num_placeholders} `{self.image_token}` "
                f"placeholders but {len(sample_images)} images were given. "
                f"Prompt: {prompt!r}"
            )

        patches, grids = [], []
        for image in sample_images:
            converted = self.image_converter(image)
            patches.append(convert_to_numpy(converted["patches"]))
            grids.append(convert_to_numpy(converted["grid_thw"]))
        merge_size = getattr(self.image_converter, "spatial_merge_size", 2)
        num_image_tokens = [
            int(g[0]) * (int(g[1]) // merge_size) * (int(g[2]) // merge_size)
            for g in grids
        ]

        prompt_ids = self._tokenize_with_special_tokens(
            prompt, num_image_tokens, []
        )
        response_ids = self._tokenize_with_special_tokens(response, [], [])
        start_ids = (
            [self.tokenizer.start_token_id] if self.add_start_token else []
        )
        end_ids = [self.tokenizer.end_token_id] if self.add_end_token else []

        ids = start_ids + prompt_ids + response_ids + end_ids
        loss_mask = [0] * (len(start_ids) + len(prompt_ids)) + [1] * (
            len(response_ids) + len(end_ids)
        )

        # Truncate from the end, but never through an image.
        total_length = sequence_length + 1
        if len(ids) > total_length:
            if self.image_token_id in ids[total_length - 1 :]:
                raise ValueError(
                    f"`sequence_length={sequence_length}` is too short to "
                    f"fit the image tokens of a sample that needs "
                    f"{len(ids) - 1} tokens. Increase `sequence_length`."
                )
            ids = ids[:total_length]
            loss_mask = loss_mask[:total_length]
        num_pad = total_length - len(ids)
        padding_mask = [True] * len(ids) + [False] * num_pad
        ids = ids + [self.tokenizer.pad_token_id] * num_pad
        loss_mask = loss_mask + [0] * num_pad

        token_ids = np.array(ids, dtype = "int32")
        grid_array = (
            np.stack(grids).astype("int32")
            if grids
            else np.zeros((0, 3), dtype = "int32")
        )
        position_ids = self._compute_position_ids_np(
            token_ids[None, :-1],
            grid_array if grids else None,
            None,
        )[0]

        sample = {
            "token_ids": token_ids,
            "padding_mask": np.array(padding_mask, dtype = "bool"),
            "loss_mask": np.array(loss_mask, dtype = "int32"),
            "position_ids": position_ids.astype("int32"),
        }
        if self.image_converter is not None:
            ic = self.image_converter
            patch_shape = (
                ic.temporal_patch_size,
                ic.patch_size,
                ic.patch_size,
                3,
            )
            sample["pixel_values"] = (
                np.concatenate(patches).astype("float32")
                if patches
                else np.zeros((0, *patch_shape), dtype = "float32")
            )
            sample["image_grid_thw"] = grid_array
            sample["vision_indices"] = np.where(
                token_ids[:-1] == self.image_token_id
            )[0].astype("int32")
        return sample

    @staticmethod
    def _as_string_list(value):
        """Return ``(list_of_str, batched)`` for str/bytes/array/list input."""

        def _decode(v):
            return v.decode("utf-8") if isinstance(v, bytes) else str(v)

        if isinstance(value, (str, bytes)):
            return [_decode(value)], False
        if not isinstance(value, (list, tuple)):
            array = convert_to_numpy(value)
            if array.ndim == 0:
                return [_decode(array.item())], False
            return [_decode(v) for v in array.tolist()], True
        return [_decode(v) for v in value], True

    @staticmethod
    def _as_per_sample_images(images, batch_size, batched):
        """Normalize ``images`` to a list (per sample) of ``(H, W, 3)``
        numpy arrays, or ``None``."""
        if images is None:
            return None
        if not batched:
            images = [images]
        if not isinstance(images, (list, tuple)):
            images = list(convert_to_numpy(images))
        if len(images) != batch_size:
            raise ValueError(
                f"Got images for {len(images)} samples but {batch_size} "
                "prompts."
            )
        per_sample = []
        for sample_images in images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = convert_to_numpy(sample_images)
                if sample_images.ndim == 3:
                    sample_images = sample_images[None]
                sample_images = list(sample_images)
            per_sample.append([convert_to_numpy(i) for i in sample_images])
        return per_sample

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        """Preprocess inputs for generation (prompt-only, no labels).

        Accepts either:
        - A plain string / list of strings (text-only).
        - A dict with ``"prompts"`` and optional ``"images"``,
          ``"videos"``, and ``"video_metadata"`` keys.

        When ``"video_metadata"`` is provided (a list of dicts
        with ``"frames_indices"`` and ``"fps"`` per video),
        timestamps in the video prompt will match HF exactly.

        Returns:
            dict with ``token_ids``, ``padding_mask``, and
            optionally ``pixel_values``, ``image_grid_thw``,
            ``vision_indices``, ``position_ids``.
        """
        # Check whether the input has images/videos.
        images = None
        videos = None
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)

        # video_metadata is read from self._video_metadata
        video_metadata = getattr(self, "_video_metadata", None)

        # Text-only: delegate to the base class entirely.
        if images is None and videos is None:
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        # Multimodal path.
        assert_tf_installed("Qwen3_5CausalLMPreprocessor with images or videos")
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts = x["prompts"]

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, 0)

        # 1. Process images and videos
        vision_out_images = None
        vision_out_videos = None

        if images is not None and self.image_converter is not None:
            vision_out_images = self._preprocess_images(images, batched)
        if videos is not None and self.video_converter is not None:
            vision_out_videos = self._preprocess_videos(videos, batched)

        # 2. Compute token counts for images.
        merge_size = getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )

        num_image_tokens = []
        if vision_out_images is not None:
            grid_np = (
                vision_out_images["image_grid_thw"].numpy()
                if hasattr(vision_out_images["image_grid_thw"], "numpy")
                else np.array(vision_out_images["image_grid_thw"])
            )
            for i in range(grid_np.shape[0]):
                t = int(grid_np[i, 0])
                h = int(grid_np[i, 1])
                w = int(grid_np[i, 2])
                num_image_tokens.append(
                    t * (h // merge_size) * (w // merge_size)
                )

        # 3. Build prompt strings and expand video tokens per-frame.
        if isinstance(prompts, tf.Tensor):
            prompts_list = [p.numpy().decode("utf-8") for p in prompts]
        elif isinstance(prompts, (list, tuple)):
            prompts_list = [
                p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
                for p in prompts
            ]
        else:
            prompts_list = [str(prompts)]

        # Expand video prompts
        num_video_tokens = []
        if vision_out_videos is not None:
            grid_np = (
                vision_out_videos["grid_thw"].numpy()
                if hasattr(vision_out_videos["grid_thw"], "numpy")
                else np.array(vision_out_videos["grid_thw"])
            )
            video_grid_thws = [
                (int(grid_np[i, 0]), int(grid_np[i, 1]), int(grid_np[i, 2]))
                for i in range(grid_np.shape[0])
            ]
            temporal_patch_size = getattr(
                self.video_converter, "temporal_patch_size", 2
            )
            for idx in range(len(prompts_list)):
                prompts_list[idx], per_frame_tokens = self._expand_video_prompt(
                    prompts_list[idx],
                    video_grid_thws,
                    temporal_patch_size,
                    video_metadata=video_metadata,
                )
                num_video_tokens.extend(per_frame_tokens)

        # 4. Tokenize with special-token-aware splitting.
        expanded_sequences = []
        for prompt_str in prompts_list:
            ids = self._tokenize_with_special_tokens(
                prompt_str, num_image_tokens, num_video_tokens
            )
            expanded_sequences.append(ids)

        # 5. Pack to fixed length.
        token_ids_ragged = tf.ragged.constant(expanded_sequences, dtype="int32")
        token_ids, padding_mask = self.packer(
            token_ids_ragged,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        # 6. Compute vision indices & M-RoPE position IDs.
        vision_indices = self._compute_vision_indices(token_ids)

        img_grid = (
            vision_out_images["image_grid_thw"] if vision_out_images else None
        )
        vid_grid = vision_out_videos["grid_thw"] if vision_out_videos else None
        pos_ids = self._compute_position_ids(token_ids, img_grid, vid_grid)

        # 7. Build combined pixel_values / image_grid_thw for vision encoder.
        pixel_values_list = []
        grid_list = []
        if vision_out_images is not None:
            pixel_values_list.append(vision_out_images["pixel_values"])
            grid_list.append(vision_out_images["image_grid_thw"])
        if vision_out_videos is not None:
            pixel_values_list.append(vision_out_videos["patches"])
            grid_list.append(vision_out_videos["grid_thw"])

        if pixel_values_list:
            combined_pixel_values = tf.concat(pixel_values_list, axis=0)
            combined_grid_thw = tf.concat(grid_list, axis=0)
        else:
            # Text-only: empty tensors so no data flows through
            # the vision encoder.
            combined_pixel_values = tf.zeros((0,), dtype="float32")
            combined_grid_thw = tf.zeros((0, 3), dtype="int32")

        result = {
            "token_ids": token_ids if batched else tf.squeeze(token_ids, 0),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, 0)
            ),
            "pixel_values": combined_pixel_values,
            "image_grid_thw": combined_grid_thw,
            "vision_indices": vision_indices,
            "position_ids": pos_ids if batched else tf.squeeze(pos_ids, 0),
        }
        return result

    def _preprocess_images(self, images, batched):
        """Convert raw images to patch tensors using the image converter.

        Args:
            images: A single numpy image, a list of images, or a
                batched tensor.
            batched: bool. Whether the input is already batched.
        Returns:
            dict with ``pixel_values`` and ``image_grid_thw``.
        """
        # Normalize to a flat list of individual 3-D images.
        if isinstance(images, (list, tuple)):
            flat_images = []
            for img in images:
                if hasattr(img, "shape") and len(img.shape) == 4:
                    for i in range(img.shape[0]):
                        flat_images.append(img[i])
                else:
                    flat_images.append(img)
        elif hasattr(images, "shape") and len(images.shape) == 4:
            flat_images = [images[i] for i in range(images.shape[0])]
        elif hasattr(images, "shape") and len(images.shape) == 3:
            flat_images = [images]
        else:
            flat_images = [images]

        all_patches = []
        all_grid_thw = []
        for img in flat_images:
            if isinstance(img, np.ndarray) and img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            result = self.image_converter(img)
            patches = result["patches"]
            grid_thw = result["grid_thw"]

            if not isinstance(patches, tf.Tensor):
                if hasattr(patches, "cpu"):
                    patches = patches.cpu().detach().numpy()
                patches = tf.constant(patches)
            if not isinstance(grid_thw, tf.Tensor):
                if hasattr(grid_thw, "cpu"):
                    grid_thw = grid_thw.cpu().detach().numpy()
                grid_thw = tf.constant(grid_thw)

            all_patches.append(patches)
            all_grid_thw.append(grid_thw)

        return {
            "pixel_values": tf.concat(all_patches, axis=0),
            "image_grid_thw": tf.stack(all_grid_thw, axis=0),
        }

    def _preprocess_videos(self, videos, batched):
        """Convert raw videos to patch tensors using the video converter."""
        if isinstance(videos, (list, tuple)):
            flat_videos = []
            for vid in videos:
                if hasattr(vid, "shape") and len(vid.shape) == 5:
                    for i in range(vid.shape[0]):
                        flat_videos.append(vid[i])
                else:
                    flat_videos.append(vid)
        elif hasattr(videos, "shape") and len(videos.shape) == 5:
            flat_videos = [videos[i] for i in range(videos.shape[0])]
        elif hasattr(videos, "shape") and len(videos.shape) == 4:
            flat_videos = [videos]
        else:
            flat_videos = [videos]

        all_patches = []
        all_grid_thw = []
        for vid in flat_videos:
            result = self.video_converter(vid)
            patches = result["patches"]
            grid_thw = result["grid_thw"]

            if not isinstance(patches, tf.Tensor):
                if hasattr(patches, "cpu"):
                    patches = patches.cpu().detach().numpy()
                patches = tf.constant(patches)
            if not isinstance(grid_thw, tf.Tensor):
                if hasattr(grid_thw, "cpu"):
                    grid_thw = grid_thw.cpu().detach().numpy()
                grid_thw = tf.constant(grid_thw)

            all_patches.append(patches)
            all_grid_thw.append(grid_thw)

        return {
            "patches": tf.concat(all_patches, axis=0),
            "grid_thw": tf.stack(all_grid_thw, axis=0),
        }

    def _generate_postprocess(self, x):
        if not self.built:
            self.build(None)

        def _strip_to_ragged(token_ids, masks, ids_to_strip):
            """Remove masked and special tokens from a sequence."""
            for id in ids_to_strip:
                masks = masks & (token_ids != id)
            if token_ids.ndim == 1:
                token_ids = token_ids[masks].tolist()
            else:
                ragged_ids = []
                for i in range(token_ids.shape[0]):
                    ragged_ids.append(token_ids[i][masks[i]].tolist())
                token_ids = ragged_ids
            return token_ids

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        token_ids = keras.ops.convert_to_numpy(token_ids).astype("int32")
        padding_mask = keras.ops.convert_to_numpy(padding_mask).astype("bool")

        # Collect all IDs to strip: base special tokens + vision tokens.
        ids_to_strip = list(self.tokenizer.special_token_ids)
        for tok_id in self._special_token_map.values():
            if tok_id not in ids_to_strip:
                ids_to_strip.append(tok_id)

        token_ids = _strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

    @preprocessing_function
    def _generate_postprocess_tf(self, x):
        if not self.built:
            self.build(None)

        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        padding_mask = keras.ops.convert_to_numpy(x["padding_mask"])

        # Collect all IDs to strip: base special tokens + vision tokens.
        ids_to_strip = list(self.tokenizer.special_token_ids)
        for tok_id in self._special_token_map.values():
            if tok_id not in ids_to_strip:
                ids_to_strip.append(tok_id)

        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        output = self.tokenizer.detokenize(token_ids)

        # Safety net: strip residual special token strings that may
        # survive if the BPE model encodes them as byte-fallback pieces.
        for tok_str in self._special_token_map:
            output = tf.strings.regex_replace(output, re.escape(tok_str), "")
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "video_fps": self.video_fps,
            }
        )
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        if self.video_converter is not None:
            config["video_converter"] = keras.layers.serialize(
                self.video_converter
            )
        return config
