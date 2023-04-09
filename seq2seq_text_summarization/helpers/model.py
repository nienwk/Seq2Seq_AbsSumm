from __future__ import annotations
from torch.nn.utils.rnn import PackedSequence
from torch import Tensor, Size, ones, vstack
from torch.nn import Embedding, Module
from typing import Union, Callable, Tuple

#Adapted from https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
def embedding_apply(embedding_layer: Embedding, packed_sequence: PackedSequence) -> PackedSequence:
    """Applies embedding to each element in `packed_sequence`. To be used directly with a dataloader output packed text, inside a model."""
    return PackedSequence(embedding_layer(packed_sequence.data.long()), packed_sequence.batch_sizes, packed_sequence.sorted_indices, packed_sequence.unsorted_indices)

def packed_seq_apply(module: Union[Module, Callable[[float],float]], packed_sequence: PackedSequence) -> PackedSequence:
    """Applies `module` forward method to `data` tensor of `packed_sequence`.\n
    Use ONLY if you know what you are doing. When in any doubt, consult the documentation for `torch.nn.utils.rnn.PackedSequence` OR do NOT use this helper function.\n
    This helper function DOES NOT guarantee that `module` will not change the resulting expected shape of the `data` tensor.\n
    In some cases, this may mess up the `sorted_indices` and `unsorted_indices` entries of the returned `torch.nn.utils.rnn.PackedSequence`, which are simply inherited from the original `packed_sequence`.
    """
    return PackedSequence(module(packed_sequence.data), packed_sequence.batch_sizes, packed_sequence.sorted_indices, packed_sequence.unsorted_indices)

def compute_attention_key_padding_mask(sequence_lengths:Tensor) -> Tensor:
    r"""Computes attention key_padding_mask for MultiHeadAttention from a tensor of sequence lengths.
    
    Will convert sequence_lengths to dtype int32."""
    integer_seq_len = sequence_lengths.int() # conversion
    attention_mask = ones(len(integer_seq_len), max(integer_seq_len).item()) * float("-inf")
    # NOTE len(integer_seq_len) is essentially the batch size, max(integer_seq_len).item() is the longest sequence length in the batch.
    # attention mask shape = [batch size, longest seq len]
    for batch_idx, seq in enumerate(attention_mask):
        seq[:integer_seq_len[batch_idx]] = Tensor([0]*integer_seq_len[batch_idx])
    return attention_mask


def compute_running_ave(old_ave:float, old_len:int, new_x:float) -> Tuple[float,int]:
    new_ave = old_ave + ((new_x - old_ave) / old_len)
    return new_ave, old_len+1

class GeneratedWord(object):
    def __init__(
        self,
        prev_word: Union[GeneratedWord, None],
        vocab_idx: int,
        logits_used: Tensor,
        log_prob: Union[Tensor[float], float], # used to compute hypothesis-length normalized log probability of hypothesis that this `GeneratedWord` belongs to, using helper function `compute_running_ave`.
        hidden: Tensor,
        cell: Tensor,
        attentional_vector: Tensor,
        untrimmed_attn_weights: Tensor,
        batch_idx: int = None,
        source_sequence_len: int = None,
        padded_encoder_output: Tensor = None,
        attention_key_padding_mask: Tensor = None,
        ) -> None:
        r"""Custom class for generated words. Also represents a hypothesis. Keeps track of the following:

        prev_word : The preceding `GeneratedWord` of this word. Manually set to None if using for the start token.

        vocab_idx : The vocabulary index of this `GeneratedWord`.

        logits_used : The pre-SoftMax/LogSoftMax logits used to compute this `GeneratedWord`. Expected shape = [1,1,vocab_size]

        log_prob : The log probability of generating this `GeneratedWord`. Obtained from one of the outputs of a `torch.topk` function call.

        hidden : The LSTM hidden state after processing the previous word, i.e. the input of the sequence timestep that this `GeneratedWord` belongs to. If this `GeneratedWord` is w_t, then hidden is h_t. Should be directly usable for decoding in the next sequence timestep. Expected shape = [num_layers, 1, decoder hidden dim]

        cell : The LSTM cell state after processing the previous word, i.e. the input of the sequence timestep that this `GeneratedWord` belongs to. If this `GeneratedWord` is w_t, then cell is cell_t. Should be directly usable for decoding in the next sequence timestep. Expected shape = [num_layers, 1, decoder hidden dim]

        attentional_vector : The attentional vector obtained by concatenating the attention context vector c_t and hidden h_t, passing through fully connected layer and activation. Expected shape = [1, 1, decoder_attentional_fc_out_dim]. See Attentional Hidden State \tilde{h}_t of paper, Effective Approaches to Attention-based Neural Machine Translation.

        untrimmed_attn_weights : The untrimmed attention weights obtained from the decoder directly. Needs to be the relevant `attn_output_weights` from the decoder and should be of shape = [1, 1, max_source_seq_len (across the original batch of N input to `Model`)]. Uses `source_sequence_len` to be trimmed to the correct shape.

        batch_idx : The index referencing the original batch of N input to `Model` that this `GeneratedWord` is part of. For example, if we generated this word for a `Hypothesis` belonging to the 3rd batch item of the original batch of N, batch_idx should be set to 2. Only needs to be initialized once for starting `GeneratedWord` with no prev_word. Otherwise inherited from prev_word.

        source_sequence_len : The length of the source text sequence for this `GeneratedWord`'s hypothesis. Used to trim `untrimmed_attn_weights` input. Only needs to be initialized once for starting `GeneratedWord` with no prev_word. Otherwise inherited from prev_word.

        padded_encoder_output : The `padded_encoder_output` obtained directly from encoder, which is one of the outputs of a `torch.nn.utils.rnn.pad_packed_sequence` call. Expected shape = [1, batch_max_input_seq_length, decoder_hidden_dim]. Only needs to be initialized once for starting `GeneratedWord` with no prev_word. Otherwise inherited from prev_word.

        attention_key_padding_mask : The `attention_key_padding_mask` computed with helper function `compute_attention_key_padding_mask`. Expected shape = [1, batch_max_input_seq_length]. Only needs to be initialized once for starting `GeneratedWord` with no prev_word. Otherwise inherited from prev_word.
        """

        self.prev_word = prev_word
        self.vocab_idx = vocab_idx

        self.logits_used = logits_used.squeeze(1)
        self.hidden = hidden
        self.cell = cell
        self.attentional_vector = attentional_vector

        if self.prev_word != None:
            self.culmulative_normalized_log_prob, self.curr_seq_len = compute_running_ave(prev_word.culmulative_normalized_log_prob, prev_word.curr_seq_len, log_prob)
            self.batch_idx = prev_word.batch_idx
            self.source_sequence_len = prev_word.source_sequence_len
            self.padded_encoder_output = prev_word.padded_encoder_output
            self.attention_key_padding_mask = prev_word.attention_key_padding_mask
        else:
            assert batch_idx != None, f"Input batch_idx argument cannot be None for GeneratedWord object with no prev_word!"
            assert source_sequence_len != None, f"Input source_sequence_len argument cannot be None for GeneratedWord object with no prev_word!"
            assert padded_encoder_output != None, f"Input padded_encoder_output argument cannot be None for GeneratedWord object with no prev_word!"
            assert attention_key_padding_mask != None, f"Input attention_key_padding_mask argument cannot be None for GeneratedWord object with no prev_word!"
            self.batch_idx = batch_idx
            self.source_sequence_len = source_sequence_len
            self.padded_encoder_output = padded_encoder_output
            self.attention_key_padding_mask = attention_key_padding_mask
            self.culmulative_normalized_log_prob = log_prob
            self.curr_seq_len = 1
        
        assert untrimmed_attn_weights.shape[:2] == Size([1,1]), f"Sanity check failed. untrimmed_attn_weights shape should be [1, 1, max_input_seq_len]. Got {untrimmed_attn_weights.shape}"
        self.attn_weights = untrimmed_attn_weights[:,:,:self.source_sequence_len].squeeze()
        # self.attn_weights shape = [self.source_sequence_len]

    def __len__(self):
        """Returns the current hypothesis length."""
        return self.curr_seq_len

    def __lt__(self, other:GeneratedWord):
        """For use with sort and sorted in Python standard library."""
        return self.culmulative_normalized_log_prob < other.culmulative_normalized_log_prob

    def __repr__(self) -> str:
        """For debugging use."""
        return f"GeneratedWord(batch_idx={self.batch_idx},vocab_idx={self.vocab_idx},hasPrev={self.prev_word!=None},curr_seq_len={self.curr_seq_len},culmulative_log_prob={self.culmulative_normalized_log_prob:0.5f})"

    def get_hypothesis_word_indices(self) -> list[int]:
        r"""Method to get the word indices of the entire hypothesis sequence as a list.
        
        Remember to collate across original batch in correct order for output of Model !!!
        """
        if self.prev_word == None:
            return [self.vocab_idx]
        else:
            return [*self.prev_word.get_hypothesis_word_indices(), self.vocab_idx]
    
    def get_hypothesis_logits(self):
        r"""Method to get the logits used to compute the entire hypothesis sequence.

        Remember to collate across original batch in correct order (list of tensors with shape [seq_len, vocab_size], where seq_len need not be equal),

        `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` to get padded tensor with shape [batch_size, longest_seq_len, vocab_size],

        swap vocab_size dim with longest_seq_len using `permute(0,2,1)` (padded tensor with shape [batch_size, vocab_size, longest_seq_len]),
        
        for correct logit outputs of Model !!!
        """
        if self.prev_word == None:
            return self.logits_used
        else:
            return vstack((self.prev_word.get_hypothesis_logits(), self.logits_used))
    
    def get_attn_weights(self):
        r"""Method to get the attention weights (used for display) over the entire hypothesis sequence.

        Remember to collate across original batch in correct order for output of Model !!!
        """
        if self.prev_word == None:
            return self.attn_weights
        else:
            return vstack((self.prev_word.get_attn_weights(), self.attn_weights))
