# -*- coding: utf-8 -*-

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from cqda.util import get_query_name_dict
from torch import Tensor
from cqda.cqd import CQD
from tqdm import tqdm

import math

from typing import Tuple

from aim.pytorch import track_params_dists

import aim


def Identity(x):
    return x


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, aim_run=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.aim_run = aim_run
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # centor for entities
            activation, cen = box_mode
            self.cen = cen  # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # center for entities
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2,
                                                 self.relation_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(query_structure),
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure],
                                                                           query_structure,
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                      query_structure),
                                                           self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def to_logits(positive_logit: Tensor, negative_logit: Tensor, eps: float = 1e-4) -> Tuple[Tensor, Tensor]:
        # XXX consider using lop1p instead! more accurate for small values
        # positive_logit is [B, 1], negative_logit is [B, S]
        assert len(positive_logit.shape) == len(negative_logit.shape) == 2
        batch_size = positive_logit.shape[0]
        assert negative_logit.shape[0] == batch_size

        # [B]
        bias_pos, _ = torch.min(positive_logit, dim=-1)
        bias_neg, _ = torch.min(negative_logit, dim=-1)
        bias = torch.min(bias_pos, bias_neg)
        bias = torch.clip(bias, max=0.0)

        # Here bias.view(B, 1) is needed for broadcasting
        positive_logit = torch.log(eps - bias.view(batch_size, 1) + positive_logit)
        negative_logit = torch.log(eps - bias.view(batch_size, 1) + negative_logit)

        return positive_logit, negative_logit

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):

        # s = time.time()

        model.train()
        # optimizer.zero_grad()

        bce_loss = nn.BCELoss(reduction='mean')
        bcel_loss = nn.BCEWithLogitsLoss(reduction='mean')
        mr_loss = nn.MarginRankingLoss(reduction='sum' if args.loss in {'mr-sum', 'amr-sum'} else 'mean')
        hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
        ce_loss = nn.CrossEntropyLoss(reduction='mean')
        nll_loss = nn.NLLLoss(reduction='mean')

        query_name_dict = get_query_name_dict()

        if step % 3001:
            if args.aim_run is not None:
                track_params_dists(model, args.aim_run)
        # track_gradients_dists(model, args.aim_run)

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            batch_queries_ = torch.LongTensor(batch_queries_dict[query_structure])
            batch_queries_dict[query_structure] = batch_queries_.cuda() if args.cuda else batch_queries_
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_sample_global = positive_sample
        negative_sample_global = negative_sample

        if isinstance(model, CQD):
            positive_sample_loss = negative_sample_loss = loss = 0.0

            loss_description = 'Unknown Loss'
            if hasattr(args, 'loss'):
                loss_description = args.loss

            # for _ in [0]:
            positive_sample = positive_sample_global
            negative_sample = negative_sample_global

            def reorder(logits: Tensor, idxs: Tensor) -> Tensor:
                return logits[torch.argsort(idxs)]

            if args.loss in {'nlp'}:
                for query_type in batch_queries_dict.keys():
                    query_type_name = query_name_dict[query_type]
                    if query_type_name == '1p':
                        input_batch: Tensor = batch_queries_dict[query_type]
                        triples = torch.cat([input_batch, positive_sample.view(-1, 1)], dim=1)
                        loss_query_type = model.loss(triples)

                        positive_sample_loss += loss_query_type
                        negative_sample_loss += loss_query_type
                        loss += loss_query_type

                        loss_description = '1p Loss'

            elif args.loss in {'1vsall'}:
                batch_size = positive_sample.shape[0]
                negative_sample_all = torch.arange(model.nentity).view(1, -1).repeat(batch_size, 1)

                positive_logit, all_logit, subsampling_weight, all_idxs = \
                    model.forward(positive_sample=positive_sample,
                                  negative_sample=negative_sample_all,
                                  subsampling_weight=subsampling_weight,
                                  batch_queries_dict=batch_queries_dict,  # {query_type: input_batch} # { '1p': [(a, p), (b, p)], '2i': [(a, b, p)] }
                                  batch_idxs_dict=batch_idxs_dict)  # {query_type: input_batch_idxs} # { '1p': [0, 2], '2i': [1] }

                assert {i for i in all_idxs} == {i for i in range(len(all_idxs))}

                all_idxs = torch.LongTensor(all_idxs).to(positive_logit.device)

                assert all_idxs.shape[0] == positive_logit.shape[0]
                assert all_idxs.shape[0] == all_logit.shape[0]

                positive_logit = reorder(positive_logit, all_idxs)
                all_logit = reorder(all_logit, all_idxs)

                do_check = False
                if do_check is True:
                    for query_structure_, queries_ in batch_queries_dict.items():
                        # print('QUERIES', query_structure_, queries_.shape)
                        idxs_ = batch_idxs_dict[query_structure_]
                        pos_, all_, _, all_idx_ = \
                            model.forward(positive_sample=positive_sample,
                                          negative_sample=negative_sample_all,
                                          subsampling_weight=subsampling_weight,
                                          batch_queries_dict={query_structure_: queries_},
                                          batch_idxs_dict={query_structure_: idxs_})
                        print('XXX', query_structure_, pos_.shape, positive_logit.shape)
                        for i in range(len(all_idx_)):
                            a = pos_[i, 0].item()
                            b = positive_logit[all_idx_[i], 0].item()
                            assert math.fabs(a - b) < 1e-12

                if model.use_logits is True:
                    positive_logit, all_logit = KGReasoning.to_logits(positive_logit, all_logit)

                # Erik: do we need to re-arrange positive_sample? I don't think so but let's double-check
                # all_logit: [B, N], positive_sample: [B]
                # Log_softmax + NLL
                loss_query_type = ce_loss(all_logit, positive_sample)

                positive_sample_loss += loss_query_type
                negative_sample_loss += loss_query_type
                loss += loss_query_type

            elif args.loss in {'bce', 'abce', 'bcel', 'abcel'}:
                positive_logit, negative_logit, subsampling_weight, all_idxs = \
                    model.forward(positive_sample=positive_sample,
                                  negative_sample=negative_sample,
                                  subsampling_weight=subsampling_weight,
                                  batch_queries_dict=batch_queries_dict,  # {query_type: input_batch},
                                  batch_idxs_dict=batch_idxs_dict)
                all_idxs = torch.LongTensor(all_idxs).to(positive_logit.device)
                # positive_logit = positive_logit[all_idxs, :]
                # negative_logit = negative_logit[all_idxs, :]
                positive_logit = reorder(positive_logit, all_idxs)
                negative_logit = reorder(negative_logit, all_idxs)

                if args.loss in {'abce', 'abcel'}:
                    # [B, N]
                    batch_size = negative_logit.shape[0]
                    negative_logit = negative_logit.view(batch_size, -1)
                    # [B]
                    negative_logit, _ = torch.max(negative_logit, dim=-1)
                    # [B, 1]
                    negative_logit = negative_logit.view(batch_size, 1)

                if model.use_logits is True:
                    positive_logit, negative_logit = KGReasoning.to_logits(positive_logit, negative_logit)

                loss_fun = bce_loss if args.loss in {'bce', 'abce'} else bcel_loss

                pos_labels = torch.ones_like(positive_logit)
                neg_labels = torch.zeros_like(negative_logit)

                positive_sample_loss = loss_fun(positive_logit, pos_labels)
                negative_sample_loss = loss_fun(negative_logit, neg_labels)

                loss += (positive_sample_loss + negative_sample_loss) / 2

            elif args.loss in {'nll'}:
                batch_size = positive_sample.shape[0]
                negative_sample_all = torch.arange(model.nentity).view(1, -1).repeat(batch_size, 1)

                positive_logit, all_logit, subsampling_weight, all_idxs = \
                    model.forward(positive_sample=positive_sample,
                                  negative_sample=negative_sample_all,
                                  subsampling_weight=subsampling_weight,
                                  batch_queries_dict=batch_queries_dict,  # {query_type: input_batch} # { '1p': [(a, p), (b, p)], '2i': [(a, b, p)] }
                                  batch_idxs_dict=batch_idxs_dict)  # {query_type: input_batch_idxs} # { '1p': [0, 2], '2i': [1] }

                assert {i for i in all_idxs} == {i for i in range(len(all_idxs))}

                all_idxs = torch.LongTensor(all_idxs).to(positive_logit.device)

                assert all_idxs.shape[0] == positive_logit.shape[0]
                assert all_idxs.shape[0] == all_logit.shape[0]

                # positive_logit = positive_logit[all_idxs, :]
                # all_logit = all_logit[all_idxs, :]

                positive_logit = reorder(positive_logit, all_idxs)
                all_logit = reorder(all_logit, all_idxs)

                if model.use_logits is True:
                    positive_logit, all_logit = KGReasoning.to_logits(positive_logit, all_logit)

                # Erik: do we need to re-arrange positive_sample? I don't think so but let's double-check
                # all_logit: [B, N], positive_sample: [B]
                loss_query_type = nll_loss(all_logit, positive_sample)

                positive_sample_loss += loss_query_type
                negative_sample_loss += loss_query_type
                loss += loss_query_type

            elif args.loss in {'hinge'}:
                positive_logit, negative_logit, subsampling_weight, all_idxs = \
                    model.forward(positive_sample=positive_sample,
                                  negative_sample=negative_sample,
                                  subsampling_weight=subsampling_weight,
                                  batch_queries_dict=batch_queries_dict,  # {query_type: input_batch},
                                  batch_idxs_dict=batch_idxs_dict)
                all_idxs = torch.LongTensor(all_idxs).to(positive_logit.device)
                # positive_logit = positive_logit[all_idxs, :]
                # negative_logit = negative_logit[all_idxs, :]
                positive_logit = reorder(positive_logit, all_idxs)
                negative_logit = reorder(negative_logit, all_idxs)

                if model.use_logits is True:
                    positive_logit, negative_logit = KGReasoning.to_logits(positive_logit, negative_logit)

                pos_labels = torch.ones_like(positive_logit)
                neg_labels = -1 * torch.ones_like(negative_logit)

                predictions = torch.cat((positive_logit, negative_logit), dim=0)
                labels = torch.cat((pos_labels, neg_labels), dim=0)

                loss += hinge_loss(predictions, labels)

                positive_sample_loss = loss
                negative_sample_loss = loss

            elif args.loss in {'ce'}:
                positive_logit, negative_logit, subsampling_weight, all_idxs = \
                    model.forward(positive_sample=positive_sample,
                                  negative_sample=negative_sample,
                                  subsampling_weight=subsampling_weight,
                                  batch_queries_dict=batch_queries_dict,  # {query_type: input_batch},
                                  batch_idxs_dict=batch_idxs_dict)
                all_idxs = torch.LongTensor(all_idxs).to(positive_logit.device)
                # positive_logit = positive_logit[all_idxs, :]
                # negative_logit = negative_logit[all_idxs, :]
                positive_logit = reorder(positive_logit, all_idxs)
                negative_logit = reorder(negative_logit, all_idxs)

                if model.use_logits is True:
                    positive_logit, negative_logit = KGReasoning.to_logits(positive_logit, negative_logit)

                # [B, 1 + N]
                pos_neg_logit = torch.cat((positive_logit, negative_logit), dim=-1)
                labels = torch.zeros(positive_logit.shape[0], dtype=torch.long, device=pos_neg_logit.device)

                loss += ce_loss(pos_neg_logit, labels)

                positive_sample_loss = loss
                negative_sample_loss = loss

            elif args.loss in {'mr', 'amr', 'mr-sum', 'amr-sum'}:
                positive_logit, negative_logit, subsampling_weight, all_idxs = \
                    model.forward(positive_sample=positive_sample,
                                  negative_sample=negative_sample,
                                  subsampling_weight=subsampling_weight,
                                  batch_queries_dict=batch_queries_dict,  # {query_type: input_batch},
                                  batch_idxs_dict=batch_idxs_dict)
                all_idxs = torch.LongTensor(all_idxs).to(positive_logit.device)
                # positive_logit = positive_logit[all_idxs, :]
                # negative_logit = negative_logit[all_idxs, :]
                positive_logit = reorder(positive_logit, all_idxs)
                negative_logit = reorder(negative_logit, all_idxs)

                if args.loss in {'amr', 'amr-sum'}:
                    batch_size = negative_logit.shape[0]
                    negative_logit = negative_logit.view(batch_size, -1)
                    negative_logit, _ = torch.max(negative_logit, dim=-1)
                    negative_logit = negative_logit.view(batch_size, 1)

                if model.use_logits is True:
                    positive_logit, negative_logit = KGReasoning.to_logits(positive_logit, negative_logit)

                loss += mr_loss(positive_logit, negative_logit, torch.ones_like(positive_logit))

                positive_sample_loss = loss
                negative_sample_loss = loss

            try:
                if args.aim_run is not None:
                    args.aim_run.track(loss.item(), name="Loss", context={'type': loss_description})

                if model.use_per_relation_params:
                    if hasattr(model.t_norm, 'p'):
                        if args.aim_run is not None:
                            args.aim_run.track(model.t_norm.p.item(), name = "Tnorm param")

                    for relation_id in range(model.nrelation):
                        if hasattr(model.negations[f"negation_{relation_id}"], 'p'):
                            if args.aim_run is not None:
                                args.aim_run.track(model.negations[f"negation_{relation_id}"].p.item(),
                                                   name=f"Tneg param",
                                                   context={'Relation': relation_id})
                else:
                    if hasattr(model.t_norm, 'p'):
                        if args.aim_run is not None:
                            args.aim_run.track(model.t_norm.p.item(), name="Tnorm param")

                    if model.negations is not None and len(model.negations) > 0:
                        if hasattr(model.negations[f"negation_{0}"], 'p'):
                            if args.aim_run is not None:
                                args.aim_run.track(model.negations[f"negation_{0}"].p.item(), name="Tneg param")

            except RuntimeError as e:
                print("Unable to track aim with error: ", e)

        else:
            positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

            negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
            positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
            positive_sample_loss = - (subsampling_weight * positive_score).sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss)/2

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if ((step + 1) % args.gradient_accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):

        query_name_dict = get_query_name_dict()

        if hasattr(model, "test_k"):
            training_k = model.k
            model.k = model.test_k if model.test_k is not None else model.k

        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        requires_grad = isinstance(model, CQD) and model.method == 'continuous'

        with torch.inference_mode(mode=not requires_grad):
            disable_tqdm = (not args.print_on_screen) or not os.environ.get("ENABLE_TQDM", False)

            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=disable_tqdm):

                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)

                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)

                for query_structure in batch_queries_dict:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                    if args.cuda:
                        batch_queries_dict[query_structure] = batch_queries_dict[query_structure].cuda()

                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    scatter_src = torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1)
                    if args.cuda:
                        scatter_src = scatter_src.cuda()
                    # achieve the ranking of all entities
                    ranking = ranking.scatter_(1, argsort, scatter_src)
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
                if args.aim_run is not None:
                    args.aim_run.track(metrics[query_structure][metric], name=metric, context={'task_type': query_name_dict[query_structure]})

            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        if hasattr(model, "test_k"):
            model.k = training_k

        return metrics
