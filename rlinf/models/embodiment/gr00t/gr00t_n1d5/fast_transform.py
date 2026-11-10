# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched host path of ``GR00TTransform`` (gr00t/model/transforms.py) for the rollout.

The reference ``GR00TTransform.apply_batch`` processes every sample on its own:
per image a PIL image (``Image.fromarray`` of a transposed array, then
``convert("RGB")``), the Eagle image processor (tiling, rescale + normalize of a
1-tile image), the chat template, the media-placeholder regex, and a collate. Here
the same outputs are built for the whole batch at once:

* pixel values: for an image of exactly one tile of the processor's size (no
  resize, no crop, no thumbnail: what ``_get_image_patches`` yields for it), the
  processor computes ``rescale_and_normalize`` (``normalize(uint8.to(float32),
  mean, std)``, elementwise) of the uint8 CHW image: the same call on the stacked
  images of the batch;
* text: the processor's own chat template per sample, the placeholders replaced by
  the exact strings ``replace_media_placeholder`` builds (1 tile per image), and the
  processor's own tokenizer call with the processor's merged kwargs;
* state / state_mask / embodiment_id: ``_prepare_state`` + collate, vectorized.

Any input outside this case (training mode, other image sizes / tile counts,
placeholders in the instruction, missing state, ...) returns None and the caller
runs the original transform.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Optional

import numpy as np
import torch

FAST_HOST_TRANSFORM = os.environ.get("RLINF_GR00T_FAST_TRANSFORM", "1") != "0"


def _one_tile_kwargs(ip, height: int, width: int) -> Optional[tuple]:
    """``rescale_and_normalize`` arguments of the processor when an image of
    (height, width) is processed as one plain tile (no resize, crop, padding or
    thumbnail), else None."""
    cache = ip.__dict__.setdefault("_rlinf_one_tile_cache", {})
    key = (height, width)
    if key in cache:
        return cache[key]
    res = None
    try:
        size = ip.size or {}
        tile = size.get("height")
        ok = (
            type(ip).__name__ == "Eagle2_5_VLImageProcessorFast"
            and not ip.do_resize
            and not ip.do_center_crop
            and not ip.pad_during_tiling
            and size.get("height") == tile == size.get("width")
            and (not ip.crop_size or not ip.crop_size.get("height"))
            and height == tile
            and width == tile
        )
        if ok:
            min_num, max_num = ip.min_dynamic_tiles, ip.max_dynamic_tiles
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            ratio = ip.find_closest_aspect_ratio(width / height, target_ratios, width, height, tile)
            ok = tuple(ratio) == (1, 1)
        if ok:
            mean = ip.image_mean
            std = ip.image_std
            mean = tuple(mean) if isinstance(mean, list) else mean
            std = tuple(std) if isinstance(std, list) else std
            res = (ip.do_rescale, ip.rescale_factor, ip.do_normalize, mean, std)
    except Exception:  # noqa: BLE001
        res = None
    cache[key] = res
    return res


def _processor_kwargs_class(proc):
    module = sys.modules.get(type(proc).__module__)
    return getattr(module, "Eagle2_5_VLProcessorKwargs", None)


def fast_groot_apply(t, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """``GR00TTransform.apply`` for a batched eval-mode rollout input, or None."""
    if t.training:
        return None
    video = data.get("video")
    if not isinstance(video, np.ndarray) or video.ndim != 6 or video.dtype != np.uint8:
        return None
    # check_keys_and_batch_size (also sets t._language_key, as the reference does)
    is_batched, batch_size = t.check_keys_and_batch_size(data)
    if not is_batched:
        return None
    B, T, V, H, W, C = video.shape
    if B != batch_size or C != 3 or B == 0:
        return None
    for k, v in data.items():
        if k in ("video", "state"):
            continue
        if k == t._language_key:
            if not isinstance(v, np.ndarray) or v.shape[:1] != (B,):
                return None
            continue
        return None  # any other key (action, ...): reference path
    state = data.get("state")
    if not torch.is_tensor(state) or state.dim() != 3 or state.shape[0] != B:
        return None
    if state.shape[1] != t.state_horizon or state.dtype != torch.float32:
        return None
    proc = t.eagle_processor
    ip = proc.image_processor
    norm_args = _one_tile_kwargs(ip, H, W)
    if norm_args is None:
        return None
    kwargs_cls = _processor_kwargs_class(proc)
    if kwargs_cls is None:
        return None

    # 1) language and chat template (GR00TTransform._prepare_language / _apply_vlm_processing)
    n_img = T * V
    texts = []
    for i in range(B):
        if t._language_key is not None:
            lang = data[t._language_key][i]
            if isinstance(lang, list):
                lang = lang[0]
        else:
            lang = t.default_instruction
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "image": None} for _ in range(n_img)]
                + [{"type": "text", "text": lang}],
            }
        ]
        texts.append(
            proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        )

    # 2) Eagle2_5_VLProcessor.__call__ / replace_media_placeholder
    output_kwargs = proc._merge_kwargs(
        kwargs_cls,
        tokenizer_init_kwargs=proc.tokenizer.init_kwargs,
        return_tensors="pt",
        padding=True,
    )
    pattern = re.compile(rf"<({proc.image_placeholder}|{proc.video_placeholder})-(\d+)>")
    expected = [(proc.image_placeholder, str(k + 1)) for k in range(n_img)]
    image_body = proc.image_start_token + proc.image_token * proc.tokens_per_tile + proc.image_end_token
    new_texts = []
    for text in texts:
        if pattern.findall(text) != expected:
            return None

        def repl(match):
            idx_in_list = int(match.group(2)) - 1
            return f"<image {idx_in_list + 1}>{image_body}"

        new_texts.append(pattern.sub(repl, text))
    text_inputs = proc.tokenizer(new_texts, **output_kwargs["text_kwargs"])

    # pixel values: per sample the images in (t v) order, CHW, through the table
    # (the processor's rescale_and_normalize is elementwise: on the stacked batch it
    # gives every image the values it gives that image alone)
    chw = torch.from_numpy(video).permute(0, 1, 2, 5, 3, 4).reshape(B * n_img, C, H, W)
    pixel_values = ip.rescale_and_normalize(chw, *norm_args)
    if pixel_values.dtype != torch.float32 or not pixel_values.is_contiguous():
        return None
    image_sizes = torch.tensor([[H, W]] * (B * n_img))

    # 3) state (GR00TTransform._prepare_state + collate)
    D = state.shape[-1]
    max_d = t.max_state_dim
    state_np = state.numpy()
    if D > max_d:
        padded = state_np[:, :, :max_d]
        n_dims = max_d
    else:
        padded = np.pad(state_np, ((0, 0), (0, 0), (0, max_d - D)), "constant")
        n_dims = D
    state_mask = np.zeros(padded.shape, dtype=bool)
    state_mask[:, :, :n_dims] = True

    batch = {
        "state": torch.from_numpy(np.ascontiguousarray(padded)),
        "state_mask": torch.from_numpy(state_mask),
    }
    for k, v in text_inputs.items():
        batch["eagle_" + k] = v
    batch["eagle_pixel_values"] = pixel_values
    batch["eagle_image_sizes"] = image_sizes
    batch["embodiment_id"] = torch.from_numpy(np.stack([t.get_embodiment_tag()] * B))
    return batch
