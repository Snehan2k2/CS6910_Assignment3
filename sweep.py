from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive

drive.mount('/content/drive/')

!unzip "/content/drive/My Drive/aksharantar_sampled.zip"

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    with open('aksharantar_sampled/tam/tam_train.csv', "r", encoding="utf-8") as f:
      train_lines = f.read().split("\n")
    pairs = []
    for i in range(len(train_lines)):
      pairs.append(train_lines[i].split(","))
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
 

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang

input_lang, output_lang = prepareData('eng', 'tam')

MAX_LENGTH = 30

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bidirectional, cell_type, num_layers, embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        if cell_type == "GRU": 
          self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional)
        elif cell_type == "RNN": 
          self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional)
        elif cell_type == "LSTM":
          self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.encoder(output, hidden)
        return output, hidden

    def initHidden(self):

        if self.cell_type == "GRU":
          return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        elif self.cell_type == "RNN":
          return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        elif self.cell_type == "LSTM":
          return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device), torch.zeros(self.num_layers, 1, self.hidden_size, device=device))
        
class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout, bidirectional, cell_type, num_layers, embedding_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        if cell_type == "GRU": 
          self.decoder = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional)
        elif cell_type == "RNN": 
          self.decoder = nn.RNN(embedding_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional)
        elif cell_type == "LSTM":
          self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.decoder(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        if self.cell_type == "GRU":
          return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        elif self.cell_type == "RNN":
          return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        elif self.cell_type == "LSTM":
          return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device), torch.zeros(self.num_layers, 1, self.hidden_size, device=device))
        
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def indexesFromSentence(lang, sentence):
  
    return [lang.word2index[word] if (word in lang.word2index) else 1 for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
  
def trainIters(encoder, decoder, n_iters, train_pairs, val_pairs, learning_rate, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 
    plot_loss_total = 0  

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(train_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            val_accu = evaluateAll(encoder, decoder, val_pairs)
            if iter == 75000:
              printcsv(encoder, decoder, val_pairs)
            print("Accuracy on validation set: ", val_accu)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] 

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() 

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step() 

    return loss.item() / target_length 
  
def evaluate(encoder, decoder, sentence, target_sentence, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        target_tensor = tensorFromSentence(output_lang, target_sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        len_target_tensor = len(target_tensor)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                        encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device) 

        decoder_hidden = encoder_hidden

        decoded_words = []
           

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            if di < len_target_tensor: 
              loss += criterion(decoder_output, target_tensor[di])
          
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words, loss 
  
def evaluateAll(encoder, decoder,pairs):
    accu = 0
    criterion = nn.NLLLoss()
    for i in range(len(pairs)):
        pair = random.choice(pairs)
        output_words, loss = evaluate(encoder, decoder, pair[0], pair[1], criterion)
        output_words = output_words[:-1]
        output_sentence = ''.join(output_words)
        if output_sentence == pair[1]: 
          accu += 1
    return accu/len(pairs)
  
def printcsv(encoder, decoder, pairs):
    criterion = nn.NLLLoss()
    import csv 
    import numpy as np
    output = []
    input_tam = []
    input_eng = []
    for i in range(len(pairs)):
        pair = random.choice(pairs)
        output_words, loss = evaluate(encoder, decoder, pair[0], pair[1], criterion)
        output_words = output_words[:-1]
        output_sentence = ''.join(output_words)
        output.append(output_sentence)
        input_tam.append(pair[1])
        input_eng.append(pair[0])
    fields = ['Input English', 'Input Tamil', 'Output Tamil'] 
    rows = []
    rows.append(input_eng)
    rows.append(input_tam)
    rows.append(output)
    rows = np.array(rows)
    rows = rows.T
    filename = "predictions_vanilla.csv"
    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)
        
!pip install wandb

import wandb

wandb.login()

with open('aksharantar_sampled/tam/tam_valid.csv', "r", encoding="utf-8") as f:
      val_lines = f.read().split("\n")

val_pairs = []

for i in range(len(val_lines)):
  val_pairs.append(val_lines[i].split(","))

with open('aksharantar_sampled/tam/tam_train.csv', "r", encoding="utf-8") as f:
  train_lines = f.read().split("\n")

train_pairs = []

for i in range(len(train_lines)):
  train_pairs.append(train_lines[i].split(","))
  
sweep_config = {
    'method': 'bayes',  # grid, random
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'embedding': {
            'values': [16, 32, 64, 256]
        },
        'layers': {
            'values': [1,2,3]
        },
        'hidden_size': {
            'values': [16, 32, 64, 256]
        },
        'cell_type': {
            'values': ["RNN", "GRU", "LSTM"]
        },
        'learning_rate': {
            'values': [0.1, 0.01, 0.001]
        },
        'dropout': {
            'values': [0.2, 0.3]
        }
    }
}



def train2():

    config_defaults = {
        'embedding': 256,
        'layers': 1,
        'hidden_size': 256,
        'cell_type' : "LSTM",
        'learning_rate' : 0.01,
        'dropout':0.2
    }

    wandb.init(config = sweep_config)
    config = wandb.init().config

    wandb.run.name = 'embedding_' + str(config.embedding) + '_layers_' + str(config.layers) +'_hidden_size_' + str(config.hidden_size) + '_cell_type_' + config.cell_type + '_learning_rate_' + str(config.learning_rate) + '_dropout_' + str(config.dropout)
    encoder = EncoderRNN(input_lang.n_words, config.hidden_size,  config.dropout ,  False ,config.cell_type, config.layers ,config.embedding).to(device)
    decoder = DecoderRNN(output_lang.n_words, config.hidden_size,  config.dropout,  False ,config.cell_type, config.layers  ,config.embedding).to(device)
    trainIters(encoder, decoder, 80000, train_pairs, val_pairs, config.learning_rate)

sweep_id = wandb.sweep(sweep_config, project="Assignment3")
wandb.agent(sweep_id, train2, project='Assignment3', count = 30)
