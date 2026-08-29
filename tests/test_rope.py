# SPDX-License-Identifier: Apache-2.0

import torch

from sparsecache.rope import apply_rope, remove_rope


def test_rope_inverse():
    generator = torch.Generator().manual_seed(7)
    tensor = torch.randn(1, 2, 5, 8, generator=generator)
    angles = torch.randn(5, 4, generator=generator)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotated = apply_rope(tensor, cos, sin, sequence_dim=2)
    restored = remove_rope(rotated, cos, sin, sequence_dim=2)
    torch.testing.assert_close(restored, tensor, atol=1e-5, rtol=1e-5)
