import pytorch_lightning as pl
import torch
import embedder
from typing import List, Tuple
from torch import nn
import numpy as np

class MeanBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        return torch.mean(x, dim = 1) # reduce over the seq dimension

class AttentionModule(torch.nn.Module):

        def __init__(self, dim, dropout):
                super().__init__()
                self.att = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = 8, dropout = dropout)    
                    
        def forward(self, x):

                out, _ =  self.att(x,x,x)

                return out

class HypetNet(pl.LightningModule):

    def __init__(self, embedder: embedder.Embedder, dim: int, dropout: float, effective_rank: int, dataset_parmas: dict):
        """
        :param embedder: the contextualized embedder to use (e.g. BERT)
        :param dim: the dimensionality of the embedding and model state vectors.
        :param dropout: dropout rate for self attention
        :param effective_rank: the rank of the generated matrices.
        """

        super().__init__()
        self.embedder = embedder
        self.dim = dim
        self.dataset_param = dataset_parmas
        self.effective_rank = effective_rank
        self.mse = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax()

        #self.att = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = 8, dropout = dropout)
        self.att_module = AttentionModule(dim, dropout)
        # create networks for predicting W_q, W_v and W_o
        self.hyper_networks = [self.create_network() for i in range(3)]
        
        self.net1 = self.hyper_networks[0]
        self.net2 = self.hyper_networks[1]
        self.net3 = self.hyper_networks[2]
        

    def create_network(self):
    
        """
        return a network module for the parameter predictions.
        if the effective rank = d, then the output is of dimensionality 2(d*dim)
        i.e., can be divided into two matrices (dim, d) and (d, dim) whose product is (dim,dim)
        """

        layers = [self.att_module]
        layers.append(MeanBlock())
        layers.append(nn.Linear(self.dim, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1024, 2048))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2048, 2048))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2048, 2 * self.dim * self.effective_rank))
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

            left, right = weights_as_vec[:, :self.dim*self.effective_rank], weights_as_vec[:, self.dim*self.effective_rank:]
            
            left_mat = left.reshape((concat.shape[0], self.dim, self.effective_rank))
            right_mat = right.reshape((concat.shape[0], self.effective_rank, self.dim))
            
            weight_mat = left_mat @ right_mat 
            assert weight_mat.shape == (concat.shape[0], self.dim, self.dim)

            weights.append(weight_mat)

        return weights

    def forward(self, batch: List[List[str]]):

        embeddings, states = self.embedder.get_states(batch)
        assert embeddings.shape == states.shape
        
        concat = torch.cat((embeddings, states), dim = 1)
        W_q, W_v, W_k = self.generate_network(concat)               
        assert W_q.shape == W_v.shape == W_k.shape == (len(batch), self.dim, self.dim)
        
        Q,V,K = embeddings @ W_q, embeddings @ W_v, embeddings @ W_k
      
        # perform self attention

        preds = self.softmax((Q @ torch.transpose(V,1,2)) / np.sqrt(self.dim)) @ K
        
        assert embeddings.shape == states.shape == preds.shape

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
