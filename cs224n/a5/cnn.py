#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab import Vocab, VocabEntry

class CNN(nn.Module):
   
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self,f,emb_size,m_word,k=5):
        super(CNN,self).__init__()

        self.conv1d=nn.Conv1d(in_channels=emb_size,out_channels=f,kernel_size=k,padding=1)

        self.maxpool=nn.MaxPool1d(kernel_size=m_word-k+1)
        
    def forward(self,X_reshape):#X_reshape=(max_sentence_length, batch_size, max_word_length))
        X_conv=self.conv1d(X_reshape)
        X_conv_out=self.maxpool(F.relu(X_conv))

        return torch.squeeze(X_conv_out,-1)

    ### END YOUR CODE
