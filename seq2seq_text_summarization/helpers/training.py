from torchtext.vocab import Vocab
from torch.nn.utils.rnn import unpack_sequence, PackedSequence
from torchmetrics.text.rouge import ROUGEScore
from typing import Tuple
from itertools import product
from numpy import mean as npmean

def setup_metrics_dict(rouge_keys:Tuple[str]):
    "Setup and returns metrics dictionary based on input `rogue_keys` (modified for dictionary keys), and empty lists as values."
    tmp_metrics = ("_fmeasure", "_precision", "_recall")
    tmp_metrics = tuple(map(lambda x:x[0]+x[1], product(rouge_keys, tmp_metrics)))
    tmp_list = tuple([] for _ in range(len(tmp_metrics)))
    return dict(zip(tmp_metrics, tmp_list))

def compute_metrics(metrics_dict:dict[str,list], vocabulary:Vocab, rouge_computer:ROUGEScore, output_seq_list:list[list], packed_target_summ:PackedSequence):
    """Directly appends computed metrics to `metrics_dict`'s lists."""
    tmp_summ = unpack_sequence(packed_target_summ)
    tmp_summ = list(map(lambda x: " ".join(vocabulary.lookup_tokens(x.tolist())), tmp_summ))
    tmp_output = list(map(lambda x: " ".join(vocabulary.lookup_tokens(x)), output_seq_list))
    tmp_rogue = rouge_computer(tmp_output, tmp_summ)
    for metric, l in metrics_dict.items():
        l.append(tmp_rogue[metric].item())
    return metrics_dict

def collate_metrics(metrics_dict:dict[str,list]):
    """Applies np.mean to the lists of metrics_dict"""
    tmp_dict = {k: npmean(v) for k,v in metrics_dict.items()}
    return tmp_dict

def append_metrics(from_metrics_dict:dict[str,list], to_metrics_dict:dict[str,list]):
    assert from_metrics_dict.keys() == to_metrics_dict.keys(), "Metrics dict keys should be the same!"
    tmp_dict = {k:[] for k in to_metrics_dict.keys()}
    for k,v in tmp_dict.items():
        v.extend(to_metrics_dict[k])
        v.extend(from_metrics_dict[k])
    return tmp_dict

def print_metrics(metrics_dict:dict[str,list]):
    tmp_metrics = collate_metrics(metrics_dict)
    for k,v in tmp_metrics.items():
        print(f"cul. ave. {k} : {v:0.4f}")