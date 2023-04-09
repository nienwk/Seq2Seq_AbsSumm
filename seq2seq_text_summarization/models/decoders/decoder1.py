import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configs.model_configs import MODEL1_EMBEDDING_DIM, MODEL1_DECODER_HIDDEN_DIM, MODEL1_DECODER_NUM_LAYERS, MODEL1_DECODER_RNN_DROPOUT_P, MODEL1_DECODER_NUM_ATTENTION_HEAD, MODEL1_DECODER_ATTENTION_DROPOUT_P, MODEL1_ACTIVATION, MODEL1_DECODER_INPUT_FEEDING_FC_DROPOUT_P, MODEL1_DECODER_ATTENTIONAL_FC_OUT_DIM

class Decoder1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = MODEL1_EMBEDDING_DIM,
        hidden_dim: int = MODEL1_DECODER_HIDDEN_DIM,
        num_layers: int = MODEL1_DECODER_NUM_LAYERS,
        rnn_dropout_p: float = MODEL1_DECODER_RNN_DROPOUT_P,
        num_attention_head: int = MODEL1_DECODER_NUM_ATTENTION_HEAD,
        attention_dropout_p: float = MODEL1_DECODER_ATTENTION_DROPOUT_P,
        activation: str = MODEL1_ACTIVATION,
        input_feeding_fc_dropout_p: float = MODEL1_DECODER_INPUT_FEEDING_FC_DROPOUT_P,
        attentional_fc_out_dim: int = MODEL1_DECODER_ATTENTIONAL_FC_OUT_DIM,
        ) -> None:
        """Decoder RNN using LSTM.\n
        Expected to create batch_size of singleton outputs in decoding.
        """
        super(Decoder1, self).__init__()
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout_p,
            bidirectional=False
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_head,
            dropout=attention_dropout_p,
            kdim=hidden_dim, # by assumption of using Encoder1, which matches Decoder1 hidden_dim
            vdim=hidden_dim, # by assumption of using Encoder1, which matches Decoder1 hidden_dim
            batch_first=True
            )

        self.input_feeding_fc = nn.Linear(attentional_fc_out_dim+embedding_dim, embedding_dim)
        self.input_feeding_fc_dropout = nn.Dropout(d=input_feeding_fc_dropout_p)

        if activation == "gelu":
            self.activation = F.gelu()
        elif activation == "relu":
            self.activation = F.relu()
        else:
            raise NotImplementedError(f"Activation function not supported. Expects 'relu' or 'gelu'. Got {activation}")

        self.attentional_fc = nn.Linear(hidden_dim+hidden_dim, attentional_fc_out_dim) # the context vector embedding dim is hidden_dim (see self.attention), hidden state dim is hidden_dim (see self.rnn), output dim is set to attentional_fc_out_dim.

        self.logits_fc = nn.Linear(attentional_fc_out_dim, vocab_size)

    def forward(
        self,
        embedded: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        padded_encoder_outputs: torch.Tensor,
        attention_key_padding_masks: torch.Tensor,
        prev_attentional_vectors: torch.Tensor = None,
        ):
        r"""Expects `embedded` to be a tensor with shape [batch size, 1, embedding_dim].

        `hidden` is expected to be a tensor with shape [num_layers, batch size, hidden_dim]

        `cell` is expected to be a tensor with shape [num_layers, batch size, hidden_dim]
        
        `padded_encoder_outputs` is expected to be returned by a `pad_packed_sequence` call. The shape is expected to be [batch size, longest source seq len, kdim=hidden_dim]. The attention mask will deal with the handling of paddings.

        `attention_key_padding_mask` is expected to be returned by the helper function `compute_attention_key_padding_mask` when called on the sequence length tensor obtained as the other returned output of the `pad_packed_sequence` call used to create `padded_encoder_outputs`.

        `prev_attentional_vectors` is expected to be a tensor with shape [batch size, 1, attentional_fc_out_dim].


        Returns a 5-tuple of Tensors:

        hidden                  with shape [num_layers, batch size, hidden_dim]

        cell                    with shape [num_layers, batch size, hidden_dim]

        attn_output_weights     with shape [batch size, 1, max_source_seq_len]

        attentional_vectors     with shape [batch size, 1, attentional_fc_out_dim]

        logits                  with shape [batch size, 1, vocab size]
        """
        # ------------------------------------------------------------------------------- #
        # Input-feeding technique from the paper, "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.
        # ------------------------------------------------------------------------------- #
        # If `prev_attentional_vectors` exists, then concat with `embedded`, apply fully connected layer with activation and dropout to be used as new input to LSTM
        if prev_attentional_vectors != None:
            embedded = torch.cat((prev_attentional_vectors, embedded), dim=2)
            # embedded shape = [batch size, 1, attentional_fc_out_dim+embedding_dim]
            embedded = self.input_feeding_fc(embedded)
            # embedded shape = [batch size, 1, embedding_dim]
            embedded = self.activation(embedded)
            embedded = self.input_feeding_fc_dropout(embedded)
            # embedded shape = [batch size, 1, embedding_dim]
        
        output, (hidden, cell) = self.rnn(embedded, hidden, cell) # prev hidden and cell (forward's inputs) are used directly in multi-layer LSTM according to the paper.
        # output shape = [batch size, 1, hidden_dim]
        # hidden shape = [num_layers, batch size, hidden_dim]
        # cell shape = [num_layers, batch size, hidden_dim]

        # ------------------------------------------------------------------------------- #
        # Global attention mechanism adapted from the paper, "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.
        # ------------------------------------------------------------------------------- #
        # top layer hidden state collected, i.e. `output` of LSTM
        # compute multiplicative attention using MultiHeadAttention layer using top layer hidden state as query, padded encoder outputs as key and value.
        # NOTE: Can need_weights=False if not implementing attention viewing. Else need to set average_attn_weights=False to view individual heads' attentions.
        attn_output, attn_output_weights = self.attention(
            query=output,
            key=padded_encoder_outputs,
            value=padded_encoder_outputs,
            key_padding_mask=attention_key_padding_masks,
            need_weights=True,
            average_attn_weights=True
            )
        # attn_output shape = [batch size, 1, hidden_dim]
        # attn_output_weights shape = [batch size, 1, max_source_seq_len]

        # attention context vector is concatenated with top layer hidden state, passed to fully connected layer, activation to give attentional_vector (one of the expected forward outputs). This is used in the next time step for input-feeding according to the paper.
        attentional_vectors = torch.cat((attn_output, output), dim=2)
        # attentional_vectors shape = [batch size, 1, hidden_dim+hidden_dim]
        attentional_vectors = self.attentional_fc(attentional_vectors)
        # attentional_vectors shape = [batch size, 1, attentional_fc_out_dim]
        attentional_vectors = self.activation(attentional_vectors)

        # attentional_vector is passed to fully connected layer (vocab version) to give logits (one of the expected forward outputs). DO NOT apply softmax here, as the resulting logits will need to be collated for CrossEntropyLoss criterion.
        logits = self.logits_fc(attentional_vectors)
        # logits shape = [batch size, 1, vocab size]

        # ------------------------------------------------------------------------------- #
        # Return by order of creation::
        # hidden shape = [num_layers, batch size, hidden_dim]
        # cell shape = [num_layers, batch size, hidden_dim]
        # attn_output_weights shape = [batch size, 1, max_source_seq_len]
        # attentional_vectors shape = [batch size, 1, attentional_fc_out_dim]
        # logits shape = [batch size, 1, vocab size]

        # TODO (OPTIONAL): implement attention viewing. Need to slice the weights to the correct size for each batch item since we parallel compute attention with trailing padding (using the key_padding_mask as a hack).

        return hidden, cell, attn_output_weights, attentional_vectors, logits
