from torch.nn.utils.rnn import PackedSequence
from torch.nn import Embedding

#Adapted from https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
def embedding_apply(embedding_layer: Embedding, packed_sequence: PackedSequence) -> PackedSequence:
    """Applies embedding to each element in `packed_sequence`. To be used directly with a dataloader output packed text, inside a model."""
    return PackedSequence(embedding_layer(packed_sequence.data.long()), packed_sequence.batch_sizes, packed_sequence.sorted_indices, packed_sequence.unsorted_indices)