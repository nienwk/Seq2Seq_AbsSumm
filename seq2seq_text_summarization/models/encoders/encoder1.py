import torch
import torch.nn as nn
from typing import Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class encoder1(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        embedding_dim : int = 512,
        hidden_dim : int = 512,
        bidirectional : bool = False,
        decoder_hidden_dim : int = None
        ):
        """Encoder RNN using LSTM.\n
        Includes fully connected layers to convert hidden/cell states to appropriate size for use in decoder initial hidden/cell state and attention computation.
        """
        super(encoder1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, decoder_hidden_dim if decoder_hidden_dim!=None else hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, decoder_hidden_dim if decoder_hidden_dim!=None else hidden_dim, bias=False)

    def forward(self, input:torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO docstring
        # TODO transformation of output, (hidden, cell) for decoder intake using fc1 and fc2
        # TODO to use pack_padded_sequence and pad_packed_sequence to handle better batch computation
        """Expects `input` to be of size
        """
        # input = [seq len, batch size, 1]
        embedded = self.embedding(input)
        # embedded = [seq len, batch size, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)

        return output, (hidden, cell)
