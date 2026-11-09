# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``openpi.transforms.ResizeImages`` without resizing masked-out camera slots."""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms
from openpi_client import image_tools


@dataclasses.dataclass(frozen=True)
class ResizeImagesUnmasked(transforms.DataTransformFn):
    """Resize (with pad) every image whose ``image_mask`` is not False.

    A slot whose sample-level ``image_mask`` is False (a padding camera) gets a
    zero image of the target size instead of a PIL resize of its input. Its
    content cannot reach any output: SigLIP runs per slot, and every token of a
    masked slot is masked out of the prefix attention (weight exactly 0), of
    the value head and of the suffix. Unmasked slots are resized exactly as by
    ``ResizeImages``.
    """

    height: int
    width: int

    def __call__(self, data: dict) -> dict:
        masks = data.get("image_mask") or {}
        images = {}
        for key, image in data["image"].items():
            mask = masks.get(key)
            if mask is not None and np.ndim(mask) == 0 and not bool(mask):
                image = np.asarray(image)
                if image.shape[-3:-1] != (self.height, self.width):
                    image = np.zeros(
                        (*image.shape[:-3], self.height, self.width, image.shape[-1]),
                        dtype=image.dtype,
                    )
                images[key] = image
            else:
                images[key] = image_tools.resize_with_pad(image, self.height, self.width)
        data["image"] = images
        return data
