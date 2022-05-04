import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def subsequent_mask(size):
    """Mask out subsequent positions.

        We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
        This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position
        i can depend only on the known outputs at positions less than i.

        TODO: look at this later

    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
    mapping a query and a set of key-value pairs to an output
    query, keys, values and output are all vectors

    weight sum of the values
    weight assigned to each value is computed by a compatibiliyt function of the query with the corresponding key.


    TODO: additive attention : using feed-forward network for scores with a single layer
          Dot product attention : identical, but different scale factors.


    the query is from the decoder hidden state => the word for which we are calculating attention
    the key and value are from the encoder hidden states (key and value are the same). => the word to which we are paying attention => how relevant is that word to the query word.



    The score is the compatibility between the query and key, which can be a dot product between the query and key (or other form of compatibility).
    The scores then go through the softmax function to yield a set of weights whose sum equals 1.
    Each weight multiplies its corresponding values to yield the context vector which utilizes all the input hidden states.

    for self-attention, Q,K,V are often the same resource.



    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm
