import numpy as np
from typing import List, Tuple
import torch

class Embedder(object):
    """
    An abstract class for contextualized embedder.
    """
    def __init__(self):
        pass

    def get_states(self, sentences: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param sentences: the input sentences
        :return: A tuple (uncontextualized_embeddings, states)
                both are pytorch tensors of shape (batch_size, seq_len, dim)
        """
        raise NotImplementedError


class DummyEmbedder(object):

    def __init__(self):
        pass

    def get_states(self, sentences: List[List[str]]) -> Tuple[torch. Tensor, torch.Tensor]:

        batch_size = len(sentences)
        seq_len = len(sentences[0])
        dim = 768
        
        embds, states = torch.rand(batch_size, seq_len, dim) - 0.5, torch.rand(batch_size, seq_len, dim) - 0.5

        return embds, states
