# -*- coding: utf-8 -*-

import pytest
import torch
from cqda.cqd.tensor_operators import Norms, Negations


norms_inputs = [
    ([torch.rand(100, 100, 3), torch.rand(100, 100, 3), i], True)
    for i in range(len(Norms.NORM_FAMILIES)) if "Additive" not in Norms.NORM_FAMILIES[i] and "LAF" not in Norms.NORM_FAMILIES[i]
]

negations_inputs = [
    ([torch.rand(100, 100, 3), i], 1.0)
    for i in range(len(Negations.NEGATION_FAMILIES))
]


@pytest.mark.parametrize("test_inputs, expected", norms_inputs)
def test_norms(test_inputs, expected):

    a , b, norm_index = test_inputs
    norm_families = Norms()
    norm_type = norm_families.NORM_FAMILIES[norm_index]

    t_norm, t_conorm = norm_families.func_dict[norm_type]

    print(f"Testing {norm_type} T-(co)norms")

    res_1 = 1.0 - t_norm(1-a, 1-b)
    res_2 = t_conorm(a,b)

    assert torch.allclose(res_1, res_2, rtol=1e-05, atol=1e-06) == expected


@pytest.mark.parametrize("test_inputs, expected", negations_inputs)
def test_negations(test_inputs, expected):

    a, norm_index = test_inputs
    norm_families = Negations()
    norm_type = norm_families.NEGATION_FAMILIES[norm_index]

    negation = norm_families.func_dict[norm_type]

    print(f"Testing {norm_type} T-(co)norms")

    res = negation(a)

    assert torch.max(res) <= expected


if __name__ == '__main__':
    pytest.main([__file__])
