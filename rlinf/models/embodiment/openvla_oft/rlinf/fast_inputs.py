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

"""Bitwise-identical, cheaper equivalent of ``MultiInputPrismaticProcessor`` + ``.to(device, dtype)`` for the rollout.

The reference (``PrismaticImageProcessor.apply_transform``) runs, for every vision backbone ``idx`` (SigLIP, DINOv2):
``resize(uint8) -> center_crop -> / 255.0 -> normalize(mean_idx, std_idx)`` on the CPU in fp32, concatenates the
results along the channels and converts them to the model dtype. Here:

* the resize + crop is computed once per distinct (resize, crop) parameter set (both backbones use the same one,
  so the second bicubic antialiased resize of the reference is the same deterministic op on the same input);
* the rest is a function of (backbone, channel, uint8 value) only: it is tabulated once by running exactly the
  reference ops (``x / 255.0``, ``TVF.normalize``, ``.to(dtype)``) on a uint8 tensor holding all 256 values per
  channel, and applied on the GPU by a gather on the uploaded uint8 image (1/4 of the bytes of the bf16 pixels).

Only used for uint8 images and the plain resize-naive transform (no letterbox); everything else takes the processor.
"""

import torch
import torchvision.transforms.functional as TVF


def _freeze(d):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items()))


class FastInputs:
    def __init__(self, processor):
        self.processor = processor
        self._tables = {}

    def eligible(self, images):
        ip = getattr(self.processor, "image_processor", None)
        if ip is None or type(ip).__name__ != "PrismaticImageProcessor":
            return False
        if type(self.processor).__name__ != "MultiInputPrismaticProcessor":
            return False
        if getattr(ip, "tvf_do_letterbox", True):
            return False
        n = len(ip.input_sizes)
        if not (len(ip.tvf_resize_params) == len(ip.tvf_crop_params) == len(ip.tvf_normalize_params) == n):
            return False
        for p in ip.tvf_normalize_params:
            if len(p["mean"]) != 3 or len(p["std"]) != 3:
                return False
        for img in images:
            if not (torch.is_tensor(img) and img.dtype == torch.uint8 and img.dim() == 5 and img.shape[2] == 3):
                return False
            if img.device.type != "cpu":
                return False
        return True

    def _table(self, idx, dtype, device):
        ip = self.processor.image_processor
        key = (idx, _freeze(ip.tvf_normalize_params[idx]), dtype, str(device))
        t = self._tables.get(key)
        if t is None:
            # all 256 uint8 values per channel, through the reference's own ops
            v = torch.arange(256, dtype=torch.uint8).view(1, 1, 1, 256).expand(1, 3, 1, 256).contiguous()
            x = v / 255.0
            x = TVF.normalize(x, **ip.tvf_normalize_params[idx])
            t = x.to(dtype).reshape(3 * 256).to(device)
            self._tables[key] = t
        return t

    def pixel_values(self, images, dtype, device):
        """images: list of uint8 CPU tensors [B, N, 3, H, W] (one per processor call of the reference).
        Returns the reference's ``pixel_values.to(device, dtype).reshape(B, -1, H', W')``."""
        ip = self.processor.image_processor
        n_backbones = len(ip.input_sizes)
        out = []
        offs = (torch.arange(3, device=device, dtype=torch.long) * 256).view(1, 3, 1, 1)
        for img in images:
            batch_size = img.shape[0]
            flat = img.reshape(-1, *img.shape[2:])
            resized = {}
            per_backbone = []
            for idx in range(n_backbones):
                k = (_freeze(ip.tvf_resize_params[idx]), _freeze(ip.tvf_crop_params[idx]))
                r = resized.get(k)
                if r is None:
                    r = TVF.resize(flat, **ip.tvf_resize_params[idx])
                    r = TVF.center_crop(r, **ip.tvf_crop_params[idx])
                    assert r.dtype == torch.uint8
                    r = r.contiguous().to(device, non_blocking=False).long() + offs
                    resized[k] = r
                per_backbone.append(torch.take(self._table(idx, dtype, device), r))
            pv = torch.cat(per_backbone, dim=1)  # [B*N, 3 * n_backbones, H', W']
            out.append(pv.reshape(batch_size, -1, *pv.shape[2:]))
        pv = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        return pv.contiguous()

    def text(self, text, padding, max_length):
        """The text part of MultiInputPrismaticProcessor.__call__ (CPU tensors)."""
        tok = self.processor.tokenizer
        assert tok.padding_side == "left", "Required: Init tokenizer with padding_side='left'"
        text_inputs = tok(text, return_tensors="pt", padding=padding, truncation=None, max_length=max_length)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        first_nonzero_indices = torch.argmax(attention_mask, dim=1).unsqueeze(1)
        assert torch.all(input_ids.gather(1, first_nonzero_indices) == tok.bos_token_id)
        assert torch.all(input_ids[:, -1] != tok.pad_token_id)
        input_ids.scatter_(1, first_nonzero_indices, tok.pad_token_id)
        attention_mask.scatter_(1, first_nonzero_indices, 0)
        input_ids[:, 0] = tok.bos_token_id
        attention_mask[:, 0] = 1
        return input_ids, attention_mask
