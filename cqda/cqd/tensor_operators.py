# -*- coding: utf-8 -*-

from torch import Tensor

from typing import Tuple, Callable, Dict

import torchnorms.negations as tneg
import torchnorms.tnorms as tnorm
import torchnorms.tconorms as tconorm


class Norms:
    MIN_NORM = 'min'
    PROD_NORM = 'prod'
    LUKA_NORM = 'Lukasiewicz'
    DRASTIC_NORM = 'drastic'
    EINSTEIN_NORM = 'Einstein'
    BOUNDED_NORM = 'bounded'
    HAMACHER_NORM = 'Hamacher'
    HAMACHER_SIMPLE_NORM = 'Hamacher-simple'
    FRANK_NORM = 'Frank'
    YAGER_NORM = 'Yager'
    SCHWEIZER_NORM = 'Schweizer'
    DOMBI_NORM = 'Dombi'
    WEBER_NORM = 'Weber'
    DUBOIS_NORM = 'Dubois'
    ACZEL_NORM = 'Aczel'
    LAF = 'LAF'
    SIAM_LAF = 'Siam-LAF'
    ADDITIVE_LINEAR_NORM = 'Additive_linear'
    ADDITIVE_NORM = 'Additive'
    ADDITIVE_LI = 'Additive_learnable_inverse'
    ADDITIVE_POLY = 'Additive_polynomial'

    NORM_FAMILIES = [MIN_NORM, PROD_NORM, LUKA_NORM, DRASTIC_NORM, EINSTEIN_NORM, BOUNDED_NORM, HAMACHER_NORM,
                     HAMACHER_SIMPLE_NORM, FRANK_NORM, YAGER_NORM, SCHWEIZER_NORM, DOMBI_NORM, WEBER_NORM, ADDITIVE_NORM,
                     ADDITIVE_LINEAR_NORM, ADDITIVE_LI, ADDITIVE_POLY, ACZEL_NORM, LAF, SIAM_LAF]

    def __init__(self, type: str = MIN_NORM):
        self.type = type
        self.func_dict = self.pair()

    def pair(self) -> Dict[str, Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, Tensor], Tensor]]]:
        func_dict = {}

        func_dict[self.MIN_NORM] = tnorm.classic.MinimumTNorm(), tconorm.classic.MinimumCoNorm()
        func_dict[self.PROD_NORM] = tnorm.classic.ProductTNorm(), tconorm.classic.ProductTCoNorm()
        func_dict[self.LUKA_NORM] = tnorm.classic.LukasiewiczTNorm(), tconorm.classic.LukasiewiczTCoNorm()
        func_dict[self.DRASTIC_NORM] = tnorm.classic.DrasticTNorm(), tnorm.classic.DrasticTNorm().conorm
        func_dict[self.EINSTEIN_NORM] = tnorm.classic.EinsteinTNorm(), tconorm.classic.EinsteinTCoNorm()
        func_dict[self.BOUNDED_NORM] = tnorm.classic.BoundedTNorm(), tconorm.classic.BoundedTCoNorm()

        tnorm_family = tnorm.classic.HamacherSimpleTNorm()
        func_dict[self.HAMACHER_SIMPLE_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.hamacher.HamacherTNorm()
        func_dict[self.HAMACHER_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.yager.YagerTNorm()
        func_dict[self.YAGER_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.ss.SchweizerSklarTNorm()
        func_dict[self.SCHWEIZER_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.dombi.DombiTNorm()
        func_dict[self.DOMBI_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.weber.WeberTNorm()
        func_dict[self.WEBER_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.dubois.DuboisTNorm()
        func_dict[self.DUBOIS_NORM] = tnorm_family, tnorm_family.conorm

        # func_dict[self.HAMACHER_NORM] = tnorm.hamacher.HamacherTNorm(), tconorm.hamacher.HamacherTCoNorm()
        # func_dict[self.YAGER_NORM] = tnorm.yager.YagerTNorm(), tconorm.yager.YagerTCoNorm()
        # func_dict[self.SCHWEIZER_NORM] = tnorm.ss.SchweizerSklarTNorm(), tconorm.ss.SchweizerSklarTCoNorm()
        # func_dict[self.DOMBI_NORM] = tnorm.dombi.DombiTNorm(), tconorm.dombi.DombiTCoNorm()
        # func_dict[self.WEBER_NORM] = tnorm.weber.WeberTNorm(), tconorm.weber.WeberTCoNorm()
        # func_dict[self.DUBOIS_NORM] = tnorm.dubois.DuboisTNorm(), tconorm.dubois.DuboisTCoNorm()

        tnorm_family = tnorm.frank.FrankTNorm()
        func_dict[self.FRANK_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.aa.AczelAlsinaTNorm()
        func_dict[self.ACZEL_NORM] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.laf.LearnableLatentTNorm()
        func_dict[self.LAF] = tnorm_family, tnorm_family.conorm

        tnorm_family = tnorm.siamese_laf.LearnableSiameseLatentTNorm
        func_dict[self.SIAM_LAF] = tnorm_family

        additive_lin = tnorm.additive_generator.AdditiveLinearGenerator()
        func_dict[self.ADDITIVE_LINEAR_NORM] = additive_lin, additive_lin.conorm

        additive = tnorm.additive_generator.AdditiveGenerator()
        func_dict[self.ADDITIVE_NORM] = additive, additive.conorm

        additive_learnable_inverse = tnorm.additive_generator.AdditiveLearnableInverseGenerator()
        func_dict[self.ADDITIVE_LI] = additive_learnable_inverse, additive_learnable_inverse.conorm

        additive_poly = tnorm.additive_generator.AdditivePolynomialGenerator()
        func_dict[self.ADDITIVE_POLY] = additive_poly, additive_poly.conorm

        return func_dict


class Negations:
    STANDARD_NEGATION = 'standard'
    STRICT_NEGATION = 'strict'
    WEBER_NEGATION = 'Weber'
    YAGER_NEGATION = 'Yager'
    STRICT_COSINE_NEGATION = 'strict_cos'
    AFFINE_NEGATION = 'affine'
    NEGATION_FAMILIES = [STANDARD_NEGATION, STRICT_NEGATION, WEBER_NEGATION, YAGER_NEGATION, STRICT_COSINE_NEGATION, AFFINE_NEGATION]

    def __init__(self, type: str = STANDARD_NEGATION):
        self.type = type
        self.func_dict = self.pair()

    def pair(self) -> Dict[str, Callable[[Tensor, Tensor], Tensor]]:
        func_dict = {}

        func_dict[self.STANDARD_NEGATION] = tneg.classic.StandardNegation()
        func_dict[self.STRICT_NEGATION] = tneg.classic.StrictNegation()
        func_dict[self.STRICT_COSINE_NEGATION] = tneg.classic.StrictCosNegation()
        func_dict[self.AFFINE_NEGATION] = tneg.classic.AffineNegation()

        func_dict[self.WEBER_NEGATION] = tneg.weber.WeberNegation()
        func_dict[self.YAGER_NEGATION] = tneg.yager.YagerNegation()

        return func_dict
