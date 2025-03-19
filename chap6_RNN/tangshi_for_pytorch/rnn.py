import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("Initialized linear weight")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embedding_random_initial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(w_embedding_random_initial).to(device))

    def forward(self, input_sentence):
        """
        :param input_sentence: a tensor containing word indices.
        :return: a tensor containing word embedding vectors.
        """
        return self.word_embedding(input_sentence.to(device))


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding.to(device)
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        self.num_layers = 2

        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        ).to(device)

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len).to(device)
        self.apply(weights_init)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(
            1, -1, self.word_embedding_dim).to(device)

        h_init = torch.zeros(
            self.num_layers, batch_input.shape[0], self.lstm_dim, device=device)
        c_init = torch.zeros(
            self.num_layers, batch_input.shape[0], self.lstm_dim, device=device)

        output, _ = self.rnn_lstm(batch_input, (h_init, c_init))
        out = output.contiguous().view(-1, self.lstm_dim)

        out = F.relu(self.fc(out))
        out = self.softmax(out)

        if is_test:
            output = out[-1, :].view(1, -1)
        else:
            output = out
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("Initialized linear weight")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embedding_random_initial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(w_embedding_random_initial).to(device))

    def forward(self, input_sentence):
        """
        :param input_sentence: a tensor containing word indices.
        :return: a tensor containing word embedding vectors.
        """
        return self.word_embedding(input_sentence.to(device))


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding.to(device)
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        self.num_layers = 2

        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        ).to(device)

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len).to(device)
        self.apply(weights_init)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(
            1, -1, self.word_embedding_dim).to(device)

        h_init = torch.zeros(
            self.num_layers, batch_input.shape[0], self.lstm_dim, device=device)
        c_init = torch.zeros(
            self.num_layers, batch_input.shape[0], self.lstm_dim, device=device)

        output, _ = self.rnn_lstm(batch_input, (h_init, c_init))
        out = output.contiguous().view(-1, self.lstm_dim)

        out = F.relu(self.fc(out))
        out = self.softmax(out)

        if is_test:
            output = out[-1, :].view(1, -1)
        else:
            output = out
        return output
