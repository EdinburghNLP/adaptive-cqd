# -*- coding: utf-8 -*-

import pytest
import torch

from torch import Tensor


def reorder(logits: Tensor, idxs: Tensor) -> Tensor:
    logits_ = torch.zeros_like(logits, device=logits.device)
    logits_[idxs] = logits
    return logits_


def reorder_v2(logits: Tensor, idxs: Tensor) -> Tensor:
    return logits[torch.argsort(idxs)]


def test_reorder():
    for _ in range(8192):
        with torch.inference_mode():
            k = torch.randint(low=2, high=65536, size=(1,))
            perm = torch.randperm(k.item())

            logits = torch.randn(k, 1)
            r1 = reorder(logits, perm)
            r2 = reorder_v2(logits, perm)

            assert torch.abs(r1 - r2).sum() < 1e-8


if __name__ == '__main__':
    pytest.main([__file__])
