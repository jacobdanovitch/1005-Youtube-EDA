import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

from tqdm import tqdm_notebook as tqdm
from loguru import logger

import datetime
import random

class YoutubeNeuralNetwork(nn.Module):
    def __init__(self, vocab, embedding_dim=200, context_size=2):
        self.vocab = vocab
        vocab_size = len(self.vocab)
        super(YoutubeNeuralNetwork, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        
        self.clean_ngrams()
        
        if torch.cuda.is_available():
          self = self.cuda()

    def clean_ngrams(self):
      top_words = [word for (word, count) in self.vocab.word2count.items() if count > 50 and word not in 				stopwords.words() and len(word) > 2]
      self.vocab.ngrams = [ng for ng in self.vocab.ngrams if ng[-1] in top_words]
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    # Takes an integer index as lookup
    def word2vec(self, lookup):
        return self.embeddings.weight[[lookup]]
    
    def __getitem__(self, w):
        return self.word2vec(self.vocab.word2index[w])
      
    def train(self, EPOCHS = 3, BATCH_SIZE = 5000):
      loss_function = nn.NLLLoss()
      optimizer = optim.SGD(self.parameters(), lr=0.001)

      logger.info("Begin training")
      losses = []
      for epoch in range(1, EPOCHS+1):
          total_loss = 0
          train_data = random.sample(self.vocab.ngrams, BATCH_SIZE) if BATCH_SIZE else self.vocab.ngrams
          
          i = 0
          for context, target in tqdm(train_data):
              # Step 1. turn the words into integer indices and wrap them in tensors
              context_idxs = torch.tensor([self.vocab.word2index[w] for w in context], dtype=torch.long)

              # Step 2. zero out the gradients from the old instance
              self.zero_grad()

              # Step 3. Run forward pass, get log probs over next words
              log_probs = self(context_idxs)

              # Step 4. Compute your loss function
              loss = loss_function(log_probs, torch.tensor([self.vocab.word2index[target]], dtype=torch.long))

              # Step 5. Do the backward pass and update the gradient
              loss.backward()
              optimizer.step()

              # Get the Python number from a 1-element Tensor by calling tensor.item()
              total_loss += loss.item()
              i+= 1
              
              #if i > (len(train_data)//4) and i > 10000:
              #  self.save_progress("YT_NN_EP{}_TEP{}_BS{}_{}.torch".format(epoch, EPOCHS, len(train_data), datetime.datetime.now()))
          model_name = "YT_NN_EP{}_TEP{}_BS{}.torch".format(epoch, EPOCHS, len(train_data))
          
          losses.append(total_loss/len(train_data))
          logger.info("Epoch {}/{}: {}".format(epoch, EPOCHS, losses[-1]))
          
          self.save_progress(model_name)
      return losses
    
    def save_progress(self, model_name):
      logger.info("Saving model checkpoint to file: {}".format(model_name))
      torch.save(self, model_name)