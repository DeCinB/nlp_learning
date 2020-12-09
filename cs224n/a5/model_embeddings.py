#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word输出的词向量维度
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.#应该是字符表

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size=word_embed_size
        char_embed_size=50#字符的embedding维度
        max_word_length=21#一个单词最长的字符数

        self.char_embedding=nn.Embedding(len(vocab.char2id),
        embedding_dim=char_embed_size,
        padding_idx=vocab.char2id['<pad>'])#字符嵌入

        self.cnn=CNN(k=5,f=self.word_embed_size,emb_size=char_embed_size,m_word=max_word_length)
        self.highway=Highway(self.word_embed_size)
        self.dropout=nn.Dropout(0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        X=self.char_embedding(input)
        sents_len,batch_size,max_word_length,char_embed_size=X.shape
        view_shape=(sents_len*batch_size,max_word_length,char_embed_size)

        X_reshape=X.view(view_shape).transpose(1,2)
        X_conv_out=self.cnn(X_reshape)
        X_highway=self.highway(X_conv_out)
        X_word_emb=self.dropout(X_highway)
        X_word_emb=X_word_emb.view(sents_len,batch_size,self.word_embed_size)
        return X_word_emb
        ### END YOUR CODE

