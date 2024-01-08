# -*- coding: utf-8 -*-

import torch
from torch import nn, optim, Tensor
import math

from cqda.cqd.util import query_to_atoms
from cqda.cqd.tensor_operators import Norms, Negations
from cqda.cqd import discrete as d2

from typing import Tuple, List, Optional, Dict
from copy import deepcopy


class N3:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQD(nn.Module):
    def __init__(self,
                 nentity: int,
                 nrelation: int,
                 rank: int,
                 init_size: float = 1e-3,
                 reg_weight: float = 1e-2,
                 test_batch_size: int = 1,
                 method: str = 'discrete',
                 t_norm_name: str = 'prod',
                 negation_name: str = 'standard',
                 k: int = 5,
                 test_k: int = None,
                 query_name_dict: Optional[Dict] = None,
                 do_sigmoid_normalize: bool = False,
                 do_minmax_normalize: bool = False,
                 do_softmax_normalize: bool = False,
                 use_cuda: bool = False,
                 freeze_embeddings: bool = False,
                 use_logits: bool = False,
                 use_per_relation_params: bool = False,
                 aim_run=None,
                 score_transform: Optional[str] = None,
                 score_transform_input: Optional[str] = None,
                 score_transform_hidden: Optional[int] = None,
                 projection_type: str = 'replace',
                 project_entity_emb: Optional[int] = None,
                 project_predicate_emb: Optional[int] = None):
        super(CQD, self).__init__()

        self.rank = rank
        self.nentity = nentity
        self.nrelation = nrelation
        self.method = method
        self.t_norm_name = t_norm_name
        self.k = k
        self.test_k = self.k if test_k is None else test_k
        self.query_name_dict = query_name_dict
        self.aim_run = aim_run
        self.negation_name = negation_name

        self.score_transform = score_transform
        self.score_transform_input = score_transform_input
        self.score_transform_hidden = score_transform_hidden

        self.projection_type = projection_type
        self.entity_projection_size = project_entity_emb
        self.predicate_projection_size = project_predicate_emb

        sizes = (nentity, nrelation)
        self.embeddings = nn.ModuleList([nn.Embedding(s, 2 * rank, sparse=False) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.entity_projection_size = project_entity_emb
        self.predicate_projection_size = project_predicate_emb

        self.entity_projection_fun = None
        self.predicate_projection_fun = None
        if self.entity_projection_size is not None:
            emb_size = 2 * rank
            self.entity_projection_fun = nn.Linear(in_features=emb_size, out_features=self.entity_projection_size, bias=True)

        if self.predicate_projection_size is not None:
            emb_size = 2 * rank
            self.predicate_projection_fun = nn.Linear(in_features=emb_size, out_features=self.predicate_projection_size, bias=True)

        if freeze_embeddings is True:
            self.embeddings[0].weight.requires_grad = False
            self.embeddings[1].weight.requires_grad = False

        self.score_transform_fun = None
        self.score_transform_subj_fun = None
        self.score_transform_pred_fun = None
        self.score_transform_obj_fun = None

        self.score_transform_use_subj: bool = ('s' in self.score_transform_input) if self.score_transform_input is not None else False
        self.score_transform_use_pred: bool = ('p' in self.score_transform_input) if self.score_transform_input is not None else False
        self.score_transform_use_obj: bool = ('o' in self.score_transform_input) if self.score_transform_input is not None else False

        if self.score_transform in {'none', None}:
            pass
        elif self.score_transform in {'affine', 'deep'}:
            # x = score
            # y = x * a + b

            nb_out_features = 2
            if self.score_transform in {'deep'}:
                assert self.score_transform_hidden is not None
                nb_out_features = self.score_transform_hidden

                nb_deep_inputs = 0
                if self.score_transform_use_subj is True:
                    nb_deep_inputs += nb_out_features
                if self.score_transform_use_pred is True:
                    nb_deep_inputs += nb_out_features
                if self.score_transform_use_obj is True:
                    nb_deep_inputs += nb_out_features
                if nb_deep_inputs == 0:
                    nb_deep_inputs = 1
                self.score_deep_transform_fun = nn.Linear(in_features=nb_deep_inputs, out_features=nb_out_features, bias=True)

            if not (self.score_transform_use_subj or self.score_transform_use_pred or self.score_transform_use_obj):
                self.score_transform_fun = nn.Linear(in_features=1, out_features=nb_out_features, bias=True)
                self.score_transform_fun.weight.data *= init_size
                self.score_transform_fun.bias.data *= init_size

            if self.score_transform_use_subj:
                input_size = 2 * rank if self.entity_projection_size is None else self.entity_projection_size
                self.score_transform_subj_fun = nn.Linear(in_features=input_size, out_features=nb_out_features, bias=True)
                self.score_transform_subj_fun.weight.data *= init_size
                self.score_transform_subj_fun.bias.data *= init_size

            if self.score_transform_use_pred:
                input_size = 2 * rank if self.predicate_projection_size is None else self.predicate_projection_size
                self.score_transform_pred_fun = nn.Linear(in_features=input_size, out_features=nb_out_features, bias=True)
                self.score_transform_pred_fun.weight.data *= init_size
                self.score_transform_pred_fun.bias.data *= init_size

            if self.score_transform_use_obj:
                input_size = 2 * rank if self.entity_projection_size is None else self.entity_projection_size
                self.score_transform_obj_fun = nn.Linear(in_features=input_size, out_features=nb_out_features, bias=True)
                self.score_transform_obj_fun.weight.data *= init_size
                self.score_transform_obj_fun.bias.data *= init_size
        else:
            assert False, f'Unknown score transform: {self.score_transform}'

        self.use_logits = use_logits
        self.use_per_relation_params = use_per_relation_params

        self.init_size = init_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

        self.do_sigmoid_normalize = do_sigmoid_normalize
        self.do_minmax_normalize = do_minmax_normalize
        self.do_softmax_normalize = do_softmax_normalize

        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)
        if self.use_cuda is True:
            self.batch_entity_range = self.batch_entity_range.cuda()

        self.norms = Norms()
        self.negations = Negations()
        self.norm_families = self.norms.NORM_FAMILIES
        self.negation_families = self.negations.NEGATION_FAMILIES

        if 'siam-laf' in self.t_norm_name.lower():
            tnorm_family = self.norms.func_dict[self.t_norm_name](input_size = nentity, hidden_dim = rank)
            self.norms.func_dict[self.t_norm_name] = tnorm_family, tnorm_family.conorm

        self.t_norm, self.t_conorm = self.norms.func_dict[self.t_norm_name]

        if use_per_relation_params:
            self.negations = [deepcopy(self.negations.func_dict[self.negation_name]) for i in range(self.nrelation)]
            self.negations = nn.ModuleDict([[f"negation_{idx}", neg] for idx,neg in enumerate(self.negations)])
        else:
            self.negations = [self.negations.func_dict[self.negation_name]]
            self.negations = nn.ModuleDict([[f"negation_{idx}", neg] for idx,neg in enumerate(self.negations)])

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        # rank = self.rank
        assert lhs_emb.shape[-1] == rel_emb.shape[-1] == rhs_emb.shape[-1]
        rank = lhs_emb.shape[-1] // 2

        lhs = lhs_emb[..., :rank], lhs_emb[..., rank:]
        rel = rel_emb[..., :rank], rel_emb[..., rank:]
        rhs = rhs_emb[..., :rank], rhs_emb[..., rank:]
        return lhs, rel, rhs

    def loss(self,
             triples: Tensor) -> Tensor:
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)
        return l_fit + l_reg

    def score_candidates(self,
                         triples: Tensor) -> Tuple[Tuple[Tensor, Tensor], Optional[List[Tensor]]]:
        lhs_emb = self.embeddings[0](triples[:, 0])
        rel_emb = self.embeddings[1](triples[:, 1])
        rhs_emb = self.embeddings[0](triples[:, 2])
        to_score = self.embeddings[0].weight
        scores_o, _ = self.score_o(lhs_emb, rel_emb, to_score)
        scores_s, _ = self.score_s(to_score, rel_emb, rhs_emb)
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        factors = self.get_factors(lhs, rel, rhs)
        return (scores_o, scores_s), factors

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(-1, -2)
        score_2 = (lhs[1] * rel[0] + lhs[0] * rel[1]) @ rhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def score_s(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (rhs[0] * rel[0] + rhs[1] * rel[1]) @ lhs[0].transpose(-1, -2)
        score_2 = (rhs[1] * rel[0] - rhs[0] * rel[1]) @ lhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def get_factors(self,
                    lhs: Tuple[Tensor, Tensor],
                    rel: Tuple[Tensor, Tensor],
                    rhs: Tuple[Tensor, Tensor]) -> List[Tensor]:
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def get_full_embeddings(self, queries: Tensor) \
            -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        lhs = rel = rhs = None
        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        if torch.sum(queries[:, 1]).item() > 0:
            rel = self.embeddings[1](queries[:, 1])
        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        return lhs, rel, rhs

    def batch_t_norm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQD.MIN_NORM:
            scores = torch.min(scores, dim=1)[0]
        elif self.t_norm_name == CQD.PROD_NORM:
            scores = torch.prod(scores, dim=1)
        else:
            raise ValueError(f't_norm must be one of {self.norm_families}, got {self.t_norm_name}')

        return scores

    def batch_t_conorm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQD.MIN_NORM:
            scores = torch.max(scores, dim=1, keepdim=True)[0]
        elif self.t_norm_name == CQD.PROD_NORM:
            scores = torch.sum(scores, dim=1, keepdim=True) - torch.prod(scores, dim=1, keepdim=True)
        else:
            raise ValueError(f't_norm must be one of {self.norm_families}, got {self.t_norm_name}')

        return scores

    def reduce_query_score(self, atom_scores, conjunction_mask, negation_mask):
        batch_size, num_atoms, *extra_dims = atom_scores.shape

        atom_scores = torch.sigmoid(atom_scores)
        scores = atom_scores.clone()
        scores[negation_mask] = 1 - atom_scores[negation_mask]

        disjunctions = scores[~conjunction_mask].reshape(batch_size, -1, *extra_dims)
        conjunctions = scores[conjunction_mask].reshape(batch_size, -1, *extra_dims)

        if disjunctions.shape[1] > 0:
            disjunctions = self.batch_t_conorm(disjunctions)

        conjunctions = torch.cat([disjunctions, conjunctions], dim=1)

        t_norm = self.batch_t_norm(conjunctions)
        return t_norm

    def forward(self,
                positive_sample: Optional[Tensor],
                negative_sample: Optional[Tensor],
                subsampling_weight: Optional[Tensor],
                batch_queries_dict: Dict[Tuple, Tensor],
                batch_idxs_dict: Dict[Tuple, List[int]]
                ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], List[int]]:
        all_idxs = []

        positive_scores_lst = [] if positive_sample is not None else None
        negative_scores_lst = [] if negative_sample is not None else None

        for query_structure, queries in batch_queries_dict.items():
            batch_size = queries.shape[0]
            atoms, num_variables, conjunction_mask, negation_mask = query_to_atoms(query_structure, queries)

            all_idxs.extend(batch_idxs_dict[query_structure])

            query_idxs = torch.LongTensor(batch_idxs_dict[query_structure]).to(queries.device)
            assert batch_size == query_idxs.shape[0]

            if negative_sample is not None:
                negative_sample = negative_sample.to(queries.device)

            query_positive_sample = positive_sample[query_idxs] if positive_sample is not None else None
            query_negative_sample = negative_sample[query_idxs] if negative_sample is not None else None

            # [False, True]
            target_mask = torch.sum(atoms == -num_variables, dim=-1) > 0

            # Offsets identify variables across different batches
            var_id_offsets = torch.arange(batch_size, device=atoms.device) * num_variables
            var_id_offsets = var_id_offsets.reshape(-1, 1, 1)

            # Replace negative variable IDs with valid identifiers
            vars_mask = atoms < 0
            atoms_offset_vars = -atoms + var_id_offsets

            atoms[vars_mask] = atoms_offset_vars[vars_mask]

            head, rel, tail = atoms[..., 0], atoms[..., 1], atoms[..., 2]
            head_vars_mask = vars_mask[..., 0]

            with torch.inference_mode():
                h_emb_constants = self.embeddings[0](head)
                r_emb = self.embeddings[1](rel)

            if 'continuous' in self.method:
                h_emb = h_emb_constants
                if num_variables > 1:
                    # var embedding for ID 0 is unused for ease of implementation
                    var_embs = nn.Embedding((num_variables * batch_size) + 1, self.rank * 2)
                    var_embs.weight.data *= self.init_size

                    var_embs.to(atoms.device)
                    optimizer = optim.Adam(var_embs.parameters(), lr=0.1)
                    prev_loss_value = 1000
                    loss_value = 999
                    i = 0

                    # CQD-CO optimization loop
                    while i < 1000 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        h_emb = h_emb_constants.clone()
                        # Fill variable positions with optimizable embeddings
                        h_emb[head_vars_mask] = var_embs(head[head_vars_mask])

                        t_emb = var_embs(tail)
                        scores, factors = self.score_o(h_emb.unsqueeze(-2),
                                                       r_emb.unsqueeze(-2),
                                                       t_emb.unsqueeze(-2),
                                                       return_factors=True)

                        query_score = self.reduce_query_score(scores,
                                                              conjunction_mask,
                                                              negation_mask)

                        loss = - query_score.mean() + self.regularizer.forward(factors)
                        loss_value = loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        i += 1

                with torch.inference_mode():
                    # Select predicates involving target variable only
                    conjunction_mask = conjunction_mask[target_mask].reshape(batch_size, -1)
                    negation_mask = negation_mask[target_mask].reshape(batch_size, -1)

                    target_mask = target_mask.unsqueeze(-1).expand_as(h_emb)
                    emb_size = h_emb.shape[-1]
                    h_emb = h_emb[target_mask].reshape(batch_size, -1, emb_size)
                    r_emb = r_emb[target_mask].reshape(batch_size, -1, emb_size)
                    to_score = self.embeddings[0].weight

                    scores, factors = self.score_o(h_emb, r_emb, to_score)
                    query_score = self.reduce_query_score(scores,
                                                          conjunction_mask,
                                                          negation_mask)
                    positive_scores_lst.append(query_score)

                # XXX: this variable "scores" is never used
                scores = torch.cat(positive_scores_lst, dim=0)

            elif 'discrete' in self.method:
                graph_type = self.query_name_dict[query_structure]
                t_norm, t_conorm = self.t_norm, self.t_conorm
                negation_fn = self.negation_fn

                if self.do_softmax_normalize is True:
                    negation_fn = self.negation_softmax

                if graph_type == "1p":
                    scores = d2.query_1p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function)
                elif graph_type == "2p":
                    scores = d2.query_2p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "3p":
                    scores = d2.query_3p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2i":
                    scores = d2.query_2i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function, t_norm=t_norm)
                elif graph_type == "3i":
                    scores = d2.query_3i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function, t_norm=t_norm)
                elif graph_type == "pi":
                    scores = d2.query_pi(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "ip":
                    scores = d2.query_ip(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=self.scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2u-DNF":
                    scores = d2.query_2u_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=self.scoring_function,
                                             t_conorm=t_conorm)
                elif graph_type == "up-DNF":
                    scores = d2.query_up_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=self.scoring_function,
                                             k=self.k, t_norm=t_norm, t_conorm=t_conorm)
                elif graph_type == "2in":
                    scores = d2.query_2in(entity_embeddings=self.embeddings[0],
                                          predicate_embeddings=self.embeddings[1],
                                          queries=queries,
                                          scoring_function=self.scoring_function,
                                          t_norm=t_norm, negation=negation_fn, aim_run = self.aim_run)
                elif graph_type == "3in":
                    scores = d2.query_3in(entity_embeddings=self.embeddings[0],
                                          predicate_embeddings=self.embeddings[1],
                                          queries=queries,
                                          scoring_function=self.scoring_function,
                                          t_norm=t_norm, negation=negation_fn, aim_run = self.aim_run)
                elif graph_type == "pin":
                    scores = d2.query_pin(entity_embeddings=self.embeddings[0],
                                          predicate_embeddings=self.embeddings[1],
                                          queries=queries,
                                          scoring_function=self.scoring_function,
                                          k=self.k, t_norm=t_norm, negation=negation_fn)
                elif graph_type == "pni":
                    scores = d2.query_pni(entity_embeddings=self.embeddings[0],
                                          predicate_embeddings=self.embeddings[1],
                                          queries=queries,
                                          scoring_function=self.scoring_function,
                                          k=self.k, t_norm=t_norm, negation=negation_fn)
                elif graph_type == "inp":
                    scores = d2.query_inp(entity_embeddings=self.embeddings[0],
                                          predicate_embeddings=self.embeddings[1],
                                          queries=queries,
                                          scoring_function=self.scoring_function,
                                          k=self.k, t_norm=t_norm, negation=negation_fn)
                else:
                    raise ValueError(f'Unknown query type: {graph_type}')

                if positive_scores_lst is not None:
                    pos_scores = self.select_scores(query_positive_sample, scores)
                    positive_scores_lst.append(pos_scores)

                if negative_scores_lst is not None:
                    neg_scores = self.select_scores(query_negative_sample, scores)
                    negative_scores_lst.append(neg_scores)

            positive_scores = negative_scores = None

            if positive_scores_lst is not None:
                positive_scores = torch.cat(positive_scores_lst, dim=0)

            if negative_scores_lst is not None:
                negative_scores = torch.cat(negative_scores_lst, dim=0)

        return positive_scores, negative_scores, subsampling_weight, all_idxs

    def select_scores(self, sample: Tensor, scores: Tensor) -> Tensor:
        if len(sample.shape) == 1:
            sample = sample.view(-1, 1)

        batch_size = scores.shape[0]
        nb_samples = sample.shape[1]
        res = scores
        if nb_samples != self.nentity:
            dim_0 = torch.arange(batch_size).view(-1, 1).repeat(1, nb_samples).view(-1)
            dim_1 = sample[:batch_size].view(-1)
            res = scores[dim_0, dim_1].view(batch_size, nb_samples)
        return res

    def minmax_normalize(self, scores_: Tensor) -> Tensor:
        scores_ = scores_ - scores_.min(1, keepdim=True)[0]
        scores_ = scores_ / scores_.max(1, keepdim=True)[0]

        # Avoiding complete 0 scores
        scores_ = scores_.clamp(min=1e-2)
        return scores_

    def find_indices(self, embeddings: Tensor, is_predicate: bool = True) -> Tensor:
        index = 1 if is_predicate else 0
        # [B, NR]
        rel_distances = torch.cdist(embeddings, self.embeddings[index].weight)
        # [B, 1] -- knn.values should be zero, check
        knn = rel_distances.topk(k=1, largest=False)

        torch.testing.assert_close(knn.values, torch.zeros_like(knn.values), atol=1e-1, rtol=0.0)
        indices = knn.indices.view(-1)
        # [B]
        return indices

    def scoring_function(self, lhs_: Tensor, rel_: Tensor, rhs_: Tensor) -> Tensor:
        if self.projection_type in {'replace'}:
            if self.entity_projection_fun is not None:
                lhs_ = self.entity_projection_fun(lhs_)
                rhs_ = self.entity_projection_fun(rhs_)

            if self.predicate_projection_fun is not None:
                rel_ = self.predicate_projection_fun(rel_)
        elif self.projection_type in {'concat'}:
            assert (self.entity_projection_fun is None) == (self.predicate_projection_fun is None)
            if self.entity_projection_fun is not None and self.predicate_projection_fun is not None:
                lhs_proj = self.entity_projection_fun(lhs_)
                rhs_proj = self.entity_projection_fun(rhs_)
                rel_proj = self.predicate_projection_fun(rel_)

                lhs_split, rel_split, rhs_split = self.split(lhs_, rel_, rhs_)
                lhs_proj_split, rel_proj_split, rhs_proj_split = self.split(lhs_proj, rel_proj, rhs_proj)

                lhs_real = torch.concat((lhs_split[0], lhs_proj_split[0]), dim=-1)
                rel_real = torch.concat((rel_split[0], rel_proj_split[0]), dim=-1)
                rhs_real = torch.concat((rhs_split[0], rhs_proj_split[0]), dim=-1)

                lhs_im = torch.concat((lhs_split[1], lhs_proj_split[1]), dim=-1)
                rel_im = torch.concat((rel_split[1], rel_proj_split[1]), dim=-1)
                rhs_im = torch.concat((rhs_split[1], rhs_proj_split[1]), dim=-1)

                lhs_ = torch.concat((lhs_real, lhs_im), dim=-1)
                rel_ = torch.concat((rel_real, rel_im), dim=-1)
                rhs_ = torch.concat((rhs_real, rhs_im), dim=-1)
        else:
            assert False, f'Unknown projection type: {self.projection_type}'

        # [B, N]
        res, _ = self.score_o(lhs_, rel_, rhs_)

        assert lhs_.shape[0] == rel_.shape[0]
        assert res.shape[0] == lhs_.shape[0]
        assert res.shape[1] == rhs_.shape[0]

        alpha, beta = 0.0, 0.0

        if self.score_transform in {'affine'}:

            if self.score_transform_fun is not None:
                # res is [B, N]
                batch_size_ = res.shape[0]
                nb_entities_ = res.shape[1]
                coeffs_ = self.score_transform_fun(res.view(-1, 1))
                # [B, N], [B, N]
                alpha = alpha + coeffs_[:, 0].view(batch_size_, nb_entities_)
                beta = beta + coeffs_[:, 1].view(batch_size_, nb_entities_)

            # [B, 1]
            if self.score_transform_subj_fun is not None:
                coeffs_ = self.score_transform_subj_fun(lhs_)
                # [B, 1], [B, 1]
                alpha = alpha + coeffs_[:, 0].view(-1, 1)
                beta = beta + coeffs_[:, 1].view(-1, 1)

            # [B, 1]
            if self.score_transform_pred_fun is not None:
                coeffs_ = self.score_transform_pred_fun(rel_)
                # [B, 1], [B, 1]
                alpha = alpha + coeffs_[:, 0].view(-1, 1)
                beta = beta + coeffs_[:, 1].view(-1, 1)

            # [1 or B, N]
            if self.score_transform_obj_fun is not None:
                coeffs_ = self.score_transform_obj_fun(rhs_)
                # [1, N], [1, N] -> [B, N], [B, N]
                alpha = alpha + coeffs_[:, 0].view(1, -1)
                beta = beta + coeffs_[:, 1].view(1, -1)

            # [B, N] = [B, N] * [B, N] + [B, N] (some broadcasting may happen here, if B or N is 1)
            res = res * (1.0 + alpha) + beta

        elif self.score_transform in {'deep'}:

            lhs_reps = rel_reps = rhs_reps = None
            batch_size_ = lhs_.shape[0]
            nb_entities_ = rhs_.shape[0]

            if self.score_transform_fun is not None:
                # res is [B, N]
                coeffs_ = self.score_transform_fun(res.view(-1, 1))
                # [B, N], [B, N]
                alpha = alpha + coeffs_[:, 0].view(batch_size_, nb_entities_)
                beta = beta + coeffs_[:, 1].view(batch_size_, nb_entities_)

            # [B, K]
            if self.score_transform_subj_fun is not None:
                lhs_reps = self.score_transform_subj_fun(lhs_)

            # [B, K]
            if self.score_transform_pred_fun is not None:
                rel_reps = self.score_transform_pred_fun(rel_)

            # [N, K]
            if self.score_transform_obj_fun is not None:
                rhs_reps = self.score_transform_obj_fun(rhs_)

            input_lst_ = []

            # [B, N, K]
            if lhs_reps is not None:
                lhs_reps = lhs_reps.view(batch_size_, 1, -1).repeat(1, nb_entities_, 1)
                input_lst_ += [lhs_reps]
            if rel_reps is not None:
                rel_reps = rel_reps.view(batch_size_, 1, -1).repeat(1, nb_entities_, 1)
                input_lst_ += [rel_reps]
            if rhs_reps is not None:
                rhs_reps = rhs_reps.view(1, nb_entities_, -1).repeat(batch_size_, 1, 1)
                input_lst_ += [rhs_reps]

            if len(input_lst_) == 0:
                input_lst_ = [res.view(batch_size_, nb_entities_, 1)]

            # [B, N, K] -> [B, N, 2]
            input_reps = torch.cat(input_lst_, dim=2)
            input_reps = torch.relu(input_reps)
            coeffs_ = self.score_deep_transform_fun(input_reps.view(batch_size_ * nb_entities_, -1))

            # [B, N], [B, N]
            alpha = alpha + coeffs_[:, 0].view(batch_size_, nb_entities_)
            beta = beta + coeffs_[:, 1].view(batch_size_, nb_entities_)

            # [B, N] = [B, N] * [B, N] + [B, N]
            res = res * (1.0 + alpha) + beta

        assert res.shape[0] == lhs_.shape[0]
        assert res.shape[1] == rhs_.shape[0]

        if self.do_sigmoid_normalize is True:
            res = torch.sigmoid(res)

        if self.do_minmax_normalize is True:
            res = self.minmax_normalize(res)

        if self.do_softmax_normalize is True:
            res = torch.softmax(res, dim=-1)

        return res

    def negation_fn(self, a: Tensor,rel_: Tensor = None) -> Tensor:
        negations = self.negations
        if len(negations) > 1:
            with torch.inference_mode():
                # [B, NR]
                rel_distances = torch.cdist(rel_, self.embeddings[1].weight)
                # [B, 1] -- knn.values should be zero, check
                knn = rel_distances.topk(1, largest=False)
                rel_indices = knn.indices.view(-1)

            rel_indices = rel_indices.clone()


            res = []
            for ind, rel_ind in enumerate(rel_indices):
                res.append(self.negations[f"negation_{rel_ind}"](a[ind]))

            res = torch.stack(res)

        else:
            res = self.negations[f"negation_{0}"](a)

        return res

    def negation_softmax(self, a: Tensor, rel_: Tensor = None) -> Tensor:
        res = - torch.log(a)
        return torch.softmax(res, dim=-1)
