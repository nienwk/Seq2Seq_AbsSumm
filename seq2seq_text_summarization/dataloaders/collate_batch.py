from torch import Tensor
from torch.nn.utils.rnn import pack_sequence

# This is used for transforming batch of (text, summ) tuples into PyTorch Tensor using pack_sequence
# Involves pad_sequence and pack_padded_sequence consecutive call under the hood
# enforce_sorted can be set to true for text because we sort by input seq len. This may mess up alignment between text and summ.
# enforce_sorted will be set to false for summ because we cannot sort both at the same time.

def collate_batch(batch):
    text_list, summ_list = [], []
    for text, summ in batch:
        text_list.append(Tensor(text))
        summ_list.append(Tensor(summ))
    return pack_sequence(text_list, enforce_sorted=False), pack_sequence(summ_list, enforce_sorted=False)
    