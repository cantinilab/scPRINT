"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Optional

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from torch import Tensor, int32
from torch.nn import Module

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)
        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        assert z.shape[-1] == self.dim, (
            f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"
        )
        small = self.project_in(z)

        z = rearrange(small, "b (c d) -> b c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b c d -> b (c d)")

        out = self.project_out(codes)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return out, indices, small


if __name__ == "__main__":
    levels = [8, 5, 5, 5]  # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels)

    x = torch.randn(1, 4, 16, 16)  # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    print(xhat.shape)  # (1, 1024, 4) - (batch, seq, dim)
    # print(indices.shape) # (1, 1024)    - (batch, seq)
