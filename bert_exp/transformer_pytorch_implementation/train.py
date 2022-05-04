import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

from bert_exp.transformer_pytorch_implementation.models.encoder_decoder import EncoderDecoder
from bert_exp.transformer_pytorch_implementation.models.encoder         import Encoder, EncoderLayer
from bert_exp.transformer_pytorch_implementation.models.decoder         import Decoder, DecoderLayer
from bert_exp.transformer_pytorch_implementation.models.util_layers     import Generator, MultiHeadedAttention, PositionwiseFeedForward, \
                                                                               Embeddings, PositionalEncoding, LabelSmoothing

from bert_exp.transformer_pytorch_implementation.models.utils           import SimpleLossCompute


from bert_exp.transformer_pytorch_implementation.train_loop.optimiser   import NoamOpt
from bert_exp.transformer_pytorch_implementation.train_loop.utils       import run_epoch
from bert_exp.transformer_pytorch_implementation.train_loop.batch_data  import Batch

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)



def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy

    attn     = MultiHeadedAttention(h, d_model)

    ff       = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(

        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# Train the simple copy task.
def simple_train():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
