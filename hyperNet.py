import pytorch_lightning as pl
import torch
import embedder
from typing import List, Tuple
from torch import nn
import numpy as np

class MeanBlock(torch.nn.Module):
    def __init__(self, in_planes_1, in_planes_2, out_planes):
        super(MeanBlock, self).__init__()

    def forward(self, x):

        return torch.mean(x, dim = 1) # reduce over the seq dimension


class HypetNet(pl.LightningModule):

    def __init__(self, embedder: embedder.Embedder, dim: int, dropout: float, effective_rank: int, dataset_param: dict):
        """
        :param embedder: the contextualized embedder to use (e.g. BERT)
        :param dim: the dimensionality of the embedding and model state vectors.
        :param dropout: dropout rate for self attention
        :param effective_rank: the rank of the generated matrices.
        """

        super().__init__()
        self.embedder = embedder
        self.dim = dim
        self.dataset_param = dataset_param
        self.effective_rank = effective_rank
        self.mse = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax()

        self.att = torch.nn.MultiheadAttention(dim = dim, num_heads = 8, dropout = dropout)
        # create networks for predicting W_q, W_v and W_o
        self.hyper_networks = [self.create_network() for i in range(3)]

    def create_network(self):
        """
        return a network module for the parameter predictions.
        if the effective rank = d, then the output is of dimensionality 2(d*dim)
        i.e., can be divided into two matrices (dim, d) and (d, dim) whose product is (dim,dim)
        """

        layers = [self.att]
        layers.append(nn.MeanBlock())
        layers.append(nn.Linear(self.dim, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dim, 2048))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dim, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(4096, 2 * self.dim * self.effective_rank))
        return nn.Sequential(*layers)

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters())

    def generate_network(self, concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param concat: the input representation, of shape (batch_size, seq_len * 2, dim)
            (concatenated embeddings and states)
        :return: attention weights, in the form of a tuple (W_q, W_v, W_k)
        """

        weights = []

        for net in self.hyper_networks:

            weights_as_vec = net(concat)

            #divide into matrices

            left, right = weights_as_vec[:self.dim*self.effective_rank], weights_as_vec[self.dim*self.effective_rank:]
            left_mat = left.reshape((self.dim, self.d))
            right_mat = right.reshape((self.d, self.dim))
            weight_mat = left_mat @ right_mat
            weights.append(weight_mat)

        return weights

    def forward(self, batch: List[List[str]]):

        embeddings, states = self.embedder.get_states(batch)

        assert embeddings.shape == states.shape

        concat = torch.cat((embeddings, states), dim = 1)
        W_q, W_v, W_k = self.generate_network(concat)

        assert W_q.shape == W_v.shape == W_k.shape == self.dim**2

        Q,V,K = embeddings @ W_q, embeddings @ W_v, embeddings @ W_k

        # perform self attention
        preds = self.softmax((Q @ torch.transpose(V)) / np.sqrt(self.dim)) @ K

        return embeddings, states, preds


    def training_step(self, batch, batch_nb):

        embeddings, states, preds = self.forward(batch)
        loss = self.mse(states, preds)

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):

        embeddings, states, preds = self.forward(batch)
        loss = self.mse(states, preds)

        return {"val_loss": loss}

   def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    @pl.data_loader
    def train_dataloader(self):
        return self.dataset_params["train"]

    @pl.data_loader
    def val_dataloader(self):
        return self.dataset_params["dev"]

    @pl.data_loader
    def test_dataloader(self):
        return self.dataset_params["test"]