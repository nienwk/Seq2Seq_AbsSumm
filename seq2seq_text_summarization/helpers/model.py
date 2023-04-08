from __future__ import annotations
from torch.nn.utils.rnn import PackedSequence
from torch import Tensor, vstack
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

def compute_running_ave(old_ave:float, old_len:int, new_x:float) -> Tuple[float,int]:
    new_ave = old_ave + ((new_x - old_ave) / old_len)
    return new_ave, old_len+1

class GeneratedWord(object):
    def __init__(
        self,
        batch_idx: int,
        prev_word: Union[GeneratedWord, None],
        vocab_idx: int,
        logits_used: Tensor,
        log_prob: float,
        hidden: Tensor,
        cell: Tensor,
        attentioned_vector: Tensor,
        encoder_outputs: Tuple[Tensor, Tensor] = None
        ) -> None:
        r"""Custom class for generated words. Also represents a hypothesis. Keeps track of the following:

        batch_idx : The index referencing the original batch of N input to `Model` that this `GeneratedWord` is part of. For example, if we generated this word for a `Hypothesis` belonging to the 3rd batch item of the original batch of N, batch_idx should be set to 2.

        prev_word : The preceding `GeneratedWord` of this word. Manually set to None if using for the start token.

        vocab_idx : The vocabulary index of this `GeneratedWord`.

        logits_used : The pre-SoftMax/LogSoftMax logits used to compute this `GeneratedWord`.

        log_prob : The log probability of generating this `GeneratedWord`. Obtained from one of the outputs of a `torch.topk` function call.

        hidden : The LSTM hidden state after processing the previous word, i.e. the input of the sequence timestep that this `GeneratedWord` belongs to. If this `GeneratedWord` is w_t, then hidden is h_t. Should be directly usable for decoding in the next sequence timestep.

        cell : The LSTM cell state after processing the previous word, i.e. the input of the sequence timestep that this `GeneratedWord` belongs to. If this `GeneratedWord` is w_t, then cell is cell_t. Should be directly usable for decoding in the next sequence timestep.

        attentioned_vector : The attentioned vector obtained by concatenating the attention context vector c_t and hidden h_t, passing through fully connected layer and activation. See Attentional Hidden State \tilde{h}_t of paper, Effective Approaches to Attention-based Neural Machine Translation.

        encoder_outputs : The 2-tuple of `padded_encoder_output` and `input_seq_lengths` obtained directly from encoder. Only need to be initialized once for starting `GeneratedWord` with no prev_word. Otherwise inherited from prev_word.
        """

        self.batch_idx = batch_idx
        self.prev_word = prev_word
        self.vocab_idx = vocab_idx

        self.logits_used = logits_used
        self.hidden = hidden
        self.cell = cell
        self.attentioned_vector = attentioned_vector

        if self.prev_word != None:
            self.culmulative_normalized_log_prob, self.curr_seq_len = compute_running_ave(prev_word.culmulative_normalized_log_prob, prev_word.curr_seq_len, log_prob)
            self.encoder_outputs = prev_word.encoder_outputs
        else:
            assert encoder_outputs!=None, f"Input encoder_outputs argument cannot be None for GeneratedWord object with no prev_word!"
            self.encoder_outputs = encoder_outputs
            self.culmulative_normalized_log_prob = log_prob
            self.curr_seq_len = 1

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
    
