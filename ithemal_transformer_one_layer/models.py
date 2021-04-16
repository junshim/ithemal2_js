# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    #p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 32 # Maximum Length for Positional Embeddings

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

    def set_vocab_size(cls, size):
        Config.vocab_size = size


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        #self.pos_embed = nn.Embedding(3000, cfg.dim) # position embedding
        self.seg_embed = nn.Embedding(512, cfg.dim)

        # drop
        #self.norm = nn.LayerNorm(cfg)
        #self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x,seg):#, pos, seg):
        #seq_len = x.size(0)
        #pos = torch.arange(seq_len, dtype=torch.long, device=x.device)

        e = self.tok_embed(x) + self.seg_embed(seg)  #+ self.pos_embed(pos) #+ self.seg_embed(seg)
        return e 
        
        #drop
        #return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

        self.output = nn.Linear(cfg.dim, cfg.dim)
        #drop
        #self.drop = nn.Dropout(cfg.p_drop_attn)

    def forward(self, x, seg):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
  
        # masking
        b,h,s,s = scores.size()
        indices = torch.triu_indices(s,s,offset=1)
        scores[:,:,indices[0],indices[1]] = float('-inf')

        scores = F.softmax(scores, dim=-1)

        #drop
        #scores = self.drop(F.softmax(scores, dim=-1))

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return self.output(h)


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(self.relu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        #self.norm1 = nn.LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        #self.norm2 = nn.LayerNorm(cfg)

        #drop
        #self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x,seg):
        h = self.attn(x,seg)
        
        #drop
        #h = self.norm1(x + self.drop(self.proj(h)))
        #h = self.norm2(h + self.drop(self.pwff(h)))

        #h = self.norm1(x + self.proj(h))
        #h = self.norm2(h + self.pwff(h))
        h = x + self.proj(h)
        h = h + self.pwff(h)
        return h

class Ithemal(nn.Module):
    """i Ithemal model with Trasformer """
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.instruction_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.prediction = nn.Linear(cfg.dim,1)
        
    def forward(self, item):

        token_output_map = {}

        trans_input = []
        segment = []
        seg = 0
        for instr, token_inputs in zip(item.block.instrs, item.x):
            trans_input = trans_input + token_inputs
            for i in range(len(token_inputs)):
                segment.append(seg)
            seg += 1

        t_output = (self.embed(torch.cuda.LongTensor(trans_input),torch.cuda.LongTensor(segment))).unsqueeze(0)

        for i_block in self.instruction_blocks:
            t_output = i_block(t_output,seg)
        t_output = t_output.squeeze(0)

        # mearge output
        #t_output = t_output.mean(dim=0)
        # not merge
        #t_output = t_output[-1]
        # sum
        t_output = t_output.sum(dim=0)
        
        # do prediction layer
        out = self.prediction(t_output).squeeze()
        return out
            

