import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from typing import Tuple

from ...configs.model_configs import MODEL1_EMBEDDING_DIM, MODEL1_ENCODER_HIDDEN_DIM, MODEL1_ENCODER_NUM_LAYERS, MODEL1_ENCODER_RNN_DROPOUT_P, MODEL1_ENCODER_FC_DROPOUT_P, MODEL1_ENCODER_BIDIRECTIONAL, MODEL1_DECODER_HIDDEN_DIM, MODEL1_ACTIVATION

class Encoder1(nn.Module):
    def __init__(
        self,
        embedding_dim: int = MODEL1_EMBEDDING_DIM,
        hidden_dim: int = MODEL1_ENCODER_HIDDEN_DIM,
        num_layers: int = MODEL1_ENCODER_NUM_LAYERS,
        rnn_dropout_p: float = MODEL1_ENCODER_RNN_DROPOUT_P,
        fc_dropout_p: float = MODEL1_ENCODER_FC_DROPOUT_P,
        bidirectional: bool = MODEL1_ENCODER_BIDIRECTIONAL,
        decoder_hidden_dim : int = MODEL1_DECODER_HIDDEN_DIM,
        activation: str = MODEL1_ACTIVATION,
        ) -> None: 
        """Encoder RNN using LSTM.\n
        Includes fully connected layer to convert cell states to appropriate size for use in decoder initial cell state.\n
        Uses LSTM proj_size parameter to scale outputs and hidden states appropriately.
        """
        super(Encoder1, self).__init__()
        self.bidirectional = bidirectional
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers

        assert decoder_hidden_dim % 2 == 0 if bidirectional else True, \
            f"For bidirectional LSTM, we convert the hidden and cell states' dimensions by dividing decoder hidden dimension by 2. Got decoder hidden dim = {decoder_hidden_dim}"

        proj_size = (decoder_hidden_dim//2) if bidirectional else decoder_hidden_dim

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout_p,
            bidirectional=bidirectional,
            proj_size=proj_size,
            )

        self.fc = nn.Linear(in_features=hidden_dim, out_features=proj_size, bias=True)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation function not supported. Expects 'relu' or 'gelu'. Got {activation}")

        self.dropout = nn.Dropout(p=fc_dropout_p)


    def forward(self, input: PackedSequence) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Expects `input` to be a `torch.nn.utils.rnn.PackedSequence` obtained from passing the `torch.nn.utils.rnn.PackedSequence` of batched text indices from the dataloader into an embedding layer, likely with a helper function.\n
        \\
        Returns a 2-tuple of 2-tuples.\n
        The first 2-tuple consists of a tensor of padded encoder outputs (pads are `float(-inf)`) and a tensor containing the original sequence lengths of the batch of inputs. This is the output of a `torch.nn.utils.rnn.pad_packed_sequence` call.\n
        The second 2-tuple consists of the final hidden state and final cell state, augmented as necessary for direct use as the decoder initial hidden and cell states.
        """
        output, (hidden, cell) = self.rnn(input)
        # IF batch size >= 2, THEN:
        # output original tensor shape = [batch size, max seq len, num_direction * proj_size]; it is converted into a PackedSequence.
        # hidden tensor shape = [num_direction * num_layers, batch size, proj_size]; needs to be manually stacked, as seen below.
        # cell tensor shape = [num_direction * num_layers, batch size, hidden_dim]; needs to be passed via a fully connected layer to convert to an appropriate cell state for the decoder.

        # ELSE if unbatched, i.e. batch size == 1:
        # output original tensor shape = [max seq len, num_direction * proj_size]; it is converted into a PackedSequence.
        # hidden tensor shape = [num_direction * num_layers, 1, proj_size]; needs to be manually stacked, as seen below.
        # cell tensor shape = [num_direction * num_layers, 1, hidden_dim]; needs to be passed via a fully connected layer to convert to an appropriate cell state for the decoder.

        batch_size = hidden.size(1)

        if self.bidirectional:
            hidden = hidden.permute(1,0,2).reshape(batch_size, self.num_layers, self.decoder_hidden_dim).permute(1,0,2)
        # hidden tensor shape [num_layers, batch size, decoder hidden dim]

        cell = self.fc(cell)
        # cell tensor shape = [num_direction * num_layers, batch size, proj_size]
        cell = self.activation(cell)
        cell = self.dropout(cell)
        # NOTE: the order in which to apply activation and dropout is subject to design decisions and case-by-case dataset performance considerations. The choice we made here is arbitrary.

        if self.bidirectional:
            cell = cell.permute(1,0,2).reshape(batch_size, self.num_layers, self.decoder_hidden_dim).permute(1,0,2)
        # cell tensor shape [num_layers, batch size, decoder hidden dim]

        padded_encoder_outputs, input_seq_lengths = pad_packed_sequence(output, batch_first=True, padding_value=float("-inf"))

        return (padded_encoder_outputs, input_seq_lengths), (hidden, cell)
