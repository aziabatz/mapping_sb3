#!/usr/bin/env python

## Parametrized Policy network.

import sys
from timeit import repeat
import torch
import numpy as np
import torch.nn as nn

sys.path.append('../Env')
sys.path.append('../utils')


class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.argmax(log_p, dim=1).long()


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = 16
        self.epsilon = 7 * self.P

    def forward(self, log_p):
        # ***** TEMPORAL: *****  Forma de hacer exploración. No sé si es válida ni está probada.
        """
        i = torch.randint(low=0, high=1000, size=(1,))
        if not (i % self.epsilon):
            n = torch.multinomial(log_p.exp(), 2).long()
            x = n[:,1]
            # print(".exploration. ", n[:,0], x)
            print(".", end='')
            return x
        """
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


class PolicyNetwork(nn.Module):

    def __init__(self, params, device):

        super(PolicyNetwork, self).__init__()

        self.policy_params = params["Graph"]
        self.P = self.policy_params["P"]
        self.M = self.policy_params["M"]

        self.hyper_params = params["Hyperparameters"]
        self.K = self.hyper_params["K"]

        self.network_params = params["Policy"]
        self.n_outputs = self.network_params["n_outputs"]
        self.n_embed = self.network_params["n_embed"]
        self.n_hidden = self.network_params["n_hidden"]

        self.gnn_params = params["GNN"]
        self.gnn_dim = self.gnn_params["dimensions"]

        init_min = self.network_params["init_min"]
        init_max = self.network_params["init_max"]
        node_selector = self.network_params["node_selector"]

        # Capacity of nodes per sample: (batches, num_nodes)
        self.capacity = torch.tensor(self.policy_params["capacity"], dtype=torch.int32)
        # self.capacity = self.capacity.repeat(self.K, 1)

        # self.embedding = nn.Linear(in_features=self.P, out_features=self.n_embed, bias=False)
        self.embedding = nn.Linear(in_features=self.gnn_dim, out_features=self.n_embed, bias=False)

        self.encoder = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_hidden, batch_first=True)
        self.decoder = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_hidden, batch_first=True)

        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(self.n_embed))
            self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(self.n_embed))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(self.n_embed))
            self.Vec2 = nn.Parameter(torch.FloatTensor(self.n_embed))

        self.W_q = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden, bias=True)
        self.W_ref = nn.Conv1d(self.n_hidden, self.n_hidden, 1, 1)
        self.W_q2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden, bias=True)
        self.W_ref2 = nn.Conv1d(self.n_hidden, self.n_hidden, 1, 1)

        self.dec_input = nn.Parameter(torch.FloatTensor(self.n_embed))
        self.__initialize_weights(init_min, init_max)
        self.clip_logits = 10
        self.softmax_T = 1.0
        self.n_glimpse = 0  # 1
        self.node_selector = {'greedy': Greedy(), 'sampling': Categorical()}.get(node_selector, None)

        self.device = device

    def __initialize_weights(self, init_min=-0.08, init_max=0.08):

        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x):
        """
        Tensors sizes should be:
            x: 				(batch, n_process, gnn_dim)
            enc_h: 			(batch, n_process, embed)
            dec_input: 		(batch, 1, embed)
            h: 				(1, batch, embed)
            return:
                pi: 		(batch, n_process)
                ll: 		(batch)
        """

        x = x.to(self.device)
        batch, n_process, _ = x.size()
        embed_enc_inputs = self.embedding(
            x)  # Transformación de la entrada de (batch, n_process, 1) a (batch, n_process, n_embed)

        embed = embed_enc_inputs.size(2)  # también self.n_embed
        mask = torch.zeros((batch, n_process), device=self.device)
        enc_h, (h, c) = self.encoder(embed_enc_inputs, None)
        # print("enc_h tiene que ser (batch, n_process, embed) => (1, 16, 128) y es: ", enc_h.shape)
        ref = enc_h
        pi_list, log_ps = [], []
        dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1).to(
            self.device)  # Colocado a formato (batch, 1, embed)
        # print("dec_input tiene que ser (batch, 1, embed) => (1, 1, 128) y es: ", dec_input.shape)

        for i in range(self.P):
            _, (h, c) = self.decoder(dec_input, (h, c))
            # print("h tiene que ser (1, batch, embed) => (1, 1, 128) y es: ", h.shape)
            query = h.squeeze(0)

            # GLIMPSE

            for i in range(self.n_glimpse):
                query, a = self.glimpse(query, ref, mask)

            # POINTER

            logits = self.pointer(query, ref, mask)

            log_p = torch.log_softmax(logits, dim=-1)
            next_node = self.node_selector(log_p)
            dec_input = torch.gather(input=embed_enc_inputs, dim=1,
                                     index=next_node.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed))

            pi_list.append(next_node)
            log_ps.append(log_p)

            mask += torch.zeros((batch, n_process), device=self.device).scatter_(dim=1, index=next_node.unsqueeze(1),
                                                                                 value=1)

        pi = torch.stack(pi_list, dim=1)
        ll = self.get_log_likelihood(torch.stack(log_ps, 1), pi)

        # return pi, ll
        # Pointer network returns a permutation over the inputs. However, interface of the
        #   policy network require to return assignations of processes to nodes (get_mapping).
        return self.__get_mapping(pi), ll

    def glimpse(self, query, ref, mask):

        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))
        u2 = self.W_ref(ref.permute(0, 2, 1))
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)

        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        u = u - 1e8 * mask

        a = nn.functional.softmax(u / self.softmax_T, dim=1)

        query = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        return query, a

    def pointer(self, query, ref, mask):

        u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))
        u2 = self.W_ref2(ref.permute(0, 2, 1))
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)

        u = torch.bmm(V, self.clip_logits * torch.tanh(u1 + u2)).squeeze(1)
        logits = u - 1e8 * mask
        return logits

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        return torch.sum(log_p.squeeze(-1), 1)

    def __get_mapping(self, a):
        # En principio funciona para configuraciones heterogéneas

        k, n = a.size()  # (batches, num_procs)
        map = -torch.ones(k, self.P, dtype=torch.int32).to(self.device)

        cap = self.capacity.detach().clone()

        node = 0
        for p in range(n):
            next_node = a[:, p].long().unsqueeze(-1)
            map = map.scatter_(dim=1, index=next_node, value=node)
            cap[node] -= 1
            if cap[node] == 0:
                node += 1

        return map
