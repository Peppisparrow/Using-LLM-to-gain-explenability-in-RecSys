#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Maurizio Ferrari Dacrema
"""

from RecSysFramework.Recommenders.DataIO import DataIO
import scipy.sparse as sps
import numpy as np
from tqdm import tqdm
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import torch, copy

from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from RecSysFramework.Utils.PyTorch.DataIterator import BPRIterator
from RecSysFramework.Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict


def normalized_adjacency_matrix(URM, add_self_connection = False):

    n_users, n_items = URM.shape

    Zero_u_u_sps = sps.csr_matrix((n_users, n_users))
    Zero_i_i_sps = sps.csr_matrix((n_items, n_items))
    A = sps.bmat([[Zero_u_u_sps,     URM ],
                  [URM.T, Zero_i_i_sps   ]], format="csr")

    if add_self_connection:
        A = A + sps.eye(A.shape[0])

    D_inv = 1/(np.sqrt(np.array(A.sum(axis = 1)).squeeze()) + 1e-6)
    A_tilde = sps.diags(D_inv).dot(A).dot(sps.diags(D_inv)).astype(np.float32)

    return A_tilde


def from_sparse_to_tensor(A_tilde):
    A_tilde = sps.coo_matrix(A_tilde)
    A_tilde = torch.sparse_coo_tensor(np.vstack([A_tilde.row, A_tilde.col]), A_tilde.data, A_tilde.shape)
    A_tilde = A_tilde.coalesce()

    return A_tilde


torch.set_default_dtype(torch.float32)

class _LightGCNModel(torch.nn.Module):

    def __init__(self, URM_train, n_layers, embedding_size, dropout_rate, device):
        super(_LightGCNModel, self).__init__()

        self.n_layers = n_layers
        self.A_split = False
        self.dropout_rate = dropout_rate
        self.n_users, self.n_items = URM_train.shape
        self.Graph = normalized_adjacency_matrix(URM_train, add_self_connection=False)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=embedding_size,
                                                 device=device)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=embedding_size,
                                                 device=device)

        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.Graph = sps.coo_matrix(self.Graph)
        self.Graph = torch.sparse_coo_tensor(np.vstack([self.Graph.row, self.Graph.col]), self.Graph.data,
                                             self.Graph.shape, device=device)
        self.Graph = self.Graph.coalesce()

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        # if self.A_split:
        #     graph = []
        #     for g in self.Graph:
        #         graph.append(self.__dropout_x(g, keep_prob))
        # else:
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout_rate > 0.0:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(1 - self.dropout_rate)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            # if self.A_split:
            #     temp_emb = []
            #     for f in range(len(g_droped)):
            #         temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
            #     side_emb = torch.cat(temp_emb, dim=0)
            #     all_emb = side_emb
            # else:
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    # def getUsersRating(self, users):
    #     all_users, all_items = self.computer()
    #     users_emb = all_users[users.long()]
    #     items_emb = all_items
    #     rating = self.f(torch.matmul(users_emb, items_emb.t()))
    #     return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCNRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """ LightGCNRecommender

    Consider the adjacency matrix   A = |  0     URM |
                                        | URM.T   0  |

    LightGCN learns user and item embeddings E based on a convolution over the graph, the model in matrix form is
    Given D the diagonal matrix containing the node degree, the embeddings are computed as:
    A_tilde = D^(-1/2)*A*D^(-1/2)
    E^(0) = randomly initialized
    E^(k+1) = A_tilde*E^(k)

    E_final = alpha_0 * E^(0) + alpha_1 * E^(1) ... alpha_k * E^(k)
            = alpha_0 * E^(0) + alpha_1 * A_tilde * E^(0) ... alpha_k * A_tilde^k * E^(0)

    In LightGCN E^(0) is trained and alpha can be optimized and learned, but the paper states
    a good predefined value is 1/(K+1)

    @inproceedings{DBLP:conf/sigir/0001DWLZ020,
      author    = {Xiangnan He and
                   Kuan Deng and
                   Xiang Wang and
                   Yan Li and
                   Yong{-}Dong Zhang and
                   Meng Wang},
      editor    = {Jimmy X. Huang and
                   Yi Chang and
                   Xueqi Cheng and
                   Jaap Kamps and
                   Vanessa Murdock and
                   Ji{-}Rong Wen and
                   Yiqun Liu},
      title     = {LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
      booktitle = {Proceedings of the 43rd International {ACM} {SIGIR} conference on
                   research and development in Information Retrieval, {SIGIR} 2020, Virtual
                   Event, China, July 25-30, 2020},
      pages     = {639--648},
      publisher = {{ACM}},
      year      = {2020},
      url       = {https://doi.org/10.1145/3397271.3401063},
      doi       = {10.1145/3397271.3401063},
      timestamp = {Wed, 03 Aug 2022 15:48:33 +0200},
      biburl    = {https://dblp.org/rec/conf/sigir/0001DWLZ020.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """

    RECOMMENDER_NAME = "LightGCNRecommender"

    def __init__(self, URM_train, use_cython_sampler=True, use_gpu=True, verbose=True):
        super(LightGCNRecommender, self).__init__(URM_train, verbose=verbose)

        self._data_iterator_class = BPRIterator_cython if use_cython_sampler else BPRIterator

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

    def fit(self,
            epochs=None,
            GNN_layers_K=None,
            batch_size=None,
            embedding_size=None,
            learning_rate=None,
            l2_reg=None,
            sgd_mode=None,
            dropout_rate=None,
            **earlystopping_kwargs):

        self.l2_reg = l2_reg

        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size=batch_size,
                                                        set_n_samples_to_draw=self.URM_train.nnz)

        torch.cuda.empty_cache()

        self._model = _LightGCNModel(self.URM_train,
                                     GNN_layers_K,
                                     embedding_size,
                                     dropout_rate,
                                     self.device)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, 0.0)

        ###############################################################################
        ### This is a standard training with early stopping part

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()
        self._model_state = self._model_state_best

    def _prepare_model_for_validation(self):
        with torch.no_grad():
            self._model.eval()
            self.USER_factors, self.ITEM_factors = self._model.computer()

            self.USER_factors = self.USER_factors.detach().cpu().numpy()
            self.ITEM_factors = self.ITEM_factors.detach().cpu().numpy()

            self._model_state = clone_pytorch_model_to_numpy_dict(self._model)
            self._model.train()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self._model_state_best = copy.deepcopy(self._model_state)

    def _run_epoch(self, num_epoch):

        if self.verbose:
            batch_iterator = tqdm(self._data_iterator)
        else:
            batch_iterator = self._data_iterator

        epoch_loss = 0
        for batch in batch_iterator:
            # Clear previously computed gradients
            self._optimizer.zero_grad()

            user_batch, pos_item_batch, neg_item_batch = batch
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            # Compute the loss function of the current batch
            loss, reg_loss = self._model.bpr_loss(user_batch, pos_item_batch, neg_item_batch)
            loss = loss + reg_loss * self.l2_reg

            # Compute gradients given current loss
            loss.backward()
            epoch_loss += loss.item()

            # Apply gradient using the selected _optimizer
            self._optimizer.step()

        self._print("Loss {:.2E}".format(epoch_loss))

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"USER_factors": self.USER_factors,
                             "ITEM_factors": self.ITEM_factors,
                             "_model_state": self._model_state,
                             }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

