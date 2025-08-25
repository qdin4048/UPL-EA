# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:33:55 2021

@author: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

SEED = 12306
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class upl_ea(nn.Module):
    def __init__(
        self, 
        input_dim, 
        primal_X_0,
        gamma,
        e,
        KG, 
        inputs,
        ent_r,
        device,
    ): 
        super(upl_ea, self).__init__()

        self.word_embedding = nn.Embedding.from_pretrained(primal_X_0, freeze=True)
        self.gamma = gamma
        self.input_dim = input_dim
        self.e = e
        self.KG = KG
        self.device = device
        self.relu = nn.ReLU()

        self.gnn = HGCN(input_dim=self.input_dim)
        self.Dense = nn.Parameter(self.glorot([3*self.input_dim, self.input_dim]).float(), requires_grad = True)
        self.Bias = nn.Parameter(torch.zeros([self.input_dim]).float(), requires_grad = True)

        self.adj = self.get_sparse_tensor(self.e, self.KG, self.device)
        self.ent_r = ent_r


    def build(self, inputs):
        """word-embedding-consisted entity features layer"""
        word_embedding = self.word_embedding.weight
        word_embedding = F.normalize(word_embedding, 2, -1)

        """Global Relation Aggr."""
        outvec_r = self.compute_r(word_embedding, inputs)
        neighbor_outvec_r = torch.spmm(self.ent_r, outvec_r)
        node_embedding_nr = torch.cat((word_embedding, neighbor_outvec_r), axis = -1)
        node_embedding_nr = self.relu(torch.matmul(node_embedding_nr, self.Dense) + self.Bias)
        node_embedding_nr = word_embedding + node_embedding_nr

        """Local Relation Aggr."""
        node_embedding = self.gnn(node_embedding_nr, self.adj)

        return node_embedding


    def forward(self, Inputs):

        node_embedding = self.build(Inputs[4])#[-1]

        """The alignment task with a hinge loss"""
        t = Inputs[2].shape[0]
        k = Inputs[2].shape[1]

        l_x = node_embedding[Inputs[0]] #[t, input_dim]
        r_x = node_embedding[Inputs[1]]
        A = torch.sum(torch.abs(l_x - r_x), 1)#[t, ]

        l_x_neg = torch.repeat_interleave(l_x, k, 0) #[t*k, input_dim]
        r_x_neg = torch.repeat_interleave(r_x, k, 0)

        neg_right = torch.reshape(Inputs[2], (t*k,)) #[t*k]
        neg_r_x = node_embedding[neg_right] #[t*k, input_dim]
        B = torch.sum(torch.abs(l_x_neg - neg_r_x), 1) #[t*k, ]
        C = - torch.reshape(B, (t, k)) #[t, k]
        D = A + self.gamma #[t, ]
        L1 = self.relu(torch.add(C, torch.reshape(D, (t, 1))))*Inputs[-1] #[t, k]

        neg2_left = torch.reshape(Inputs[3], (t*k,))
        neg_l_x = node_embedding[neg2_left]
        B = torch.sum(torch.abs(neg_l_x - r_x_neg), 1)
        C = - torch.reshape(B, (t, k))
        L2 = self.relu(torch.add(C, torch.reshape(D, (t, 1))))*Inputs[-1]

        hinge_loss = (torch.sum(L1) + torch.sum(L2)) / (2.0)

        return hinge_loss


    def glorot(self, shape):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = torch.distributions.uniform.Uniform(torch.tensor([-init_range]), 
                        torch.tensor([init_range])).sample(shape).squeeze(-1)
        return initial


    def get_mat(self, e, KG):
        du = [{e_id} for e_id in range(e)]
        for tri in KG:
            if tri[0] != tri[2]:
                du[tri[0]].add(tri[2])
                du[tri[2]].add(tri[0])
        du = [len(d) for d in du]
        M = {}
        for tri in KG:
            if tri[0] == tri[2]:
                continue
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = 1
            else:
                pass
            if (tri[2], tri[0]) not in M:
                M[(tri[2], tri[0])] = 1
            else:
                pass

        for i in range(e):
            M[(i, i)] = 1
        return M, du


    # get a sparse tensor based on relational triples
    def get_sparse_tensor(self, e, KG, device):
        # print('getting a sparse tensor...')
        M, du = self.get_mat(e, KG)
        ind = []
        val = []
        # M_arr = np.zeros((e, e))
        for fir, sec in M:
            ind.append((sec, fir))
            val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))

        M = torch.sparse_coo_tensor(np.array(ind).transpose(), val, [e,e], device = device)

        return M


    def compute_r(self, inlayer, inputs):
        L=torch.spmm(inputs[0], inlayer)#/torch.unsqueeze(inputs[2],-1)
        R=torch.spmm(inputs[1], inlayer)#/torch.unsqueeze(inputs[3],-1)

        r_forward=torch.cat((L,R),axis=-1)
        r_reverse=torch.cat((-L,-R),axis=-1)
        r_embeddings = torch.cat((r_forward, r_reverse), axis=0)
        
        return r_embeddings

#%%
class HGCN(nn.Module):
    # add a layer
    def __init__(self, input_dim, dropout=0.0, device="cpu"):
        super(HGCN, self).__init__()
        self.kernel_gate = nn.Parameter(self.glorot([input_dim, input_dim]).float(), requires_grad = True)
        self.bias_gate = nn.Parameter(torch.zeros([input_dim]).float(), requires_grad = True)
        self.Weight_1 = nn.Parameter(self.glorot([input_dim, input_dim]).float(), requires_grad = True)
        self.Weight_2 = nn.Parameter(self.glorot([input_dim, input_dim]).float(), requires_grad = True)
        self.Dense = nn.Parameter(self.glorot([3*input_dim, input_dim]).float(), requires_grad = True)
        self.Bias = nn.Parameter(torch.zeros([input_dim]).float(), requires_grad = True)
        
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.input_dim = input_dim

    def forward(self, x, adj):
        """Constructing layers of the model"""

        # HGCN_layer1
        gcn_layer1 = self.add_dense_layer(x, adj, self.relu, dropout=self.dropout, dense = self.Weight_1)
        hgcn_layer1 = self.highway(x, gcn_layer1, self.input_dim, self.kernel_gate, self.bias_gate)

        # HGCN_layer2
        gcn_layer2 = self.add_dense_layer(hgcn_layer1, adj, self.relu, dropout=self.dropout, dense = self.Weight_2)
        node_embedding = self.highway(hgcn_layer1, gcn_layer2, self.input_dim, self.kernel_gate, self.bias_gate)
        return node_embedding
    
    def glorot(self, shape):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = torch.distributions.uniform.Uniform(torch.tensor([-init_range]), 
                        torch.tensor([init_range])).sample(shape).squeeze(-1)
        return initial

    def highway(self, layer1, layer2, dimension, kernel_gate, bias_gate):
        transform_gate = torch.matmul(layer1, kernel_gate) + bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate

        return transform_gate * layer2 + carry_gate * layer1
    
    def add_dense_layer(self, inlayer, M, act_func, dropout, dense):
        inlayer = F.dropout(inlayer, p = dropout, training=self.training)
        inlayer_1 = torch.matmul(inlayer, dense)
        tosum = torch.spmm(M, inlayer_1)
        if act_func is None:
            return tosum
        else:
            return act_func(tosum)