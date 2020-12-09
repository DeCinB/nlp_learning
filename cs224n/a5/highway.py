#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self,emb_size):
        super(Highway,self).__init__()
        self.proj=nn.Linear(emb_size,emb_size)
        self.gate=nn.Linear(emb_size,emb_size)

    def forward(self,x):
        x_proj=F.relu(self.proj(x))
        x_gate=self.gate(x).sigmoid()
        x_out=x_gate*x_proj+(1-x_gate)*x

        return x_out


    ### END YOUR CODE

