from typing import Union, Generator, Tuple
from os import PathLike
from collections import OrderedDict
from pandas import read_csv
from torch import save, load
from torchtext.vocab import build_vocab_from_iterator, Vocab, vocab
from torchtext.data.utils import get_tokenizer
from ..configs.data_prep_configs import POSTPROCESSING_DIR, TRAIN_CSV, FIELDNAME_CLEANED_TEXT
from ..configs.dataloader_configs import TOKENIZER, LANGUAGE, PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN, VOCAB_DIR, SOURCE_VOCAB_EXPORT, TARGET_VOCAB_EXPORT

# Basic vocabulary builder functions

def setup_tokenizer(tokenizer_backend=TOKENIZER, language=LANGUAGE):
    tokenizer = get_tokenizer(tokenizer=tokenizer_backend, language=language)
    return tokenizer

def yield_tokens(
    tokenizer_backend=TOKENIZER,
    language=LANGUAGE,
    csvpath=POSTPROCESSING_DIR+TRAIN_CSV,
    column_name=FIELDNAME_CLEANED_TEXT
    ) -> Generator[str, None, None]:

    tokenizer = setup_tokenizer(tokenizer_backend, language)
    df = read_csv(csvpath)
    for entry in df[column_name]:
        yield tokenizer(entry)

def make_vocab_obj(
    tokenizer_backend=TOKENIZER,
    language=LANGUAGE,
    csvpath=POSTPROCESSING_DIR+TRAIN_CSV,
    column_name=FIELDNAME_CLEANED_TEXT
    ) -> Vocab:

    vocabulary = build_vocab_from_iterator(
        yield_tokens(tokenizer_backend, language, csvpath, column_name),
        min_freq=1,
        specials=[PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN],
        special_first=True
    )
    vocabulary.set_default_index(3) # set OOV to UNK_TOKEN
    return vocabulary

# Export and import functions

def export_vocab(vocabulary: Vocab, path: Union[str, PathLike]) -> None:
    ind_to_token = vocabulary.get_itos() # list of token strings
    token_ordered_dict = OrderedDict((token, 1) for token in ind_to_token) # OrderedDict must be of token,frequency. Since vocabulary has no duplicates, freq=1 for all tokens.
    default_ind = vocabulary.get_default_index()
    vocab_size = len(vocabulary) # For sanity checking during importing
    d = {"token_ordered_dict": token_ordered_dict, "default_ind": default_ind, "vocab_size": vocab_size}
    save(d, path)

def export_source_vocab(vocabulary: Vocab, path: Union[str, PathLike] = VOCAB_DIR+SOURCE_VOCAB_EXPORT, verbose: bool = True) -> None:
    export_vocab(vocabulary, path)
    if verbose:
        print(f"Source vocabulary exported! Path: ./{path if type(path)==str else path.__fspath__()}")

def export_target_vocab(vocabulary: Vocab, path: Union[str, PathLike] = VOCAB_DIR+TARGET_VOCAB_EXPORT, verbose: bool = True) -> None:
    export_vocab(vocabulary, path)
    if verbose:
        print(f"Target vocabulary exported! Path: ./{path if type(path)==str else path.__fspath__()}")

def export_same_vocab(
    vocabulary: Vocab,
    source_path: Union[str, PathLike] = VOCAB_DIR+SOURCE_VOCAB_EXPORT,
    target_path: Union[str, PathLike] = VOCAB_DIR+TARGET_VOCAB_EXPORT,
    verbose: bool = True) -> None:
    """Export same vocabulary for both source and target vocabularies"""

    export_source_vocab(vocabulary, source_path, verbose)
    export_target_vocab(vocabulary, target_path, verbose)

def import_vocab(path: Union[str, PathLike]) -> Vocab:
    d = load(path)
    vocabulary = vocab(d["token_ordered_dict"])
    assert len(vocabulary) == d["vocab_size"], "Something went wrong with importing vocabulary! Size of imported vocabulary is not the same as the saved vocabulary!"
    vocabulary.set_default_index(d["default_ind"])
    return vocabulary

def import_source_vocab(path: Union[str, PathLike] = VOCAB_DIR+SOURCE_VOCAB_EXPORT, verbose: bool = True) -> Vocab:
    vocabulary = import_vocab(path)
    if verbose:
        print("Source vocabulary imported!")
    return vocabulary

def import_target_vocab(path: Union[str, PathLike] = VOCAB_DIR+TARGET_VOCAB_EXPORT, verbose: bool = True) -> Vocab:
    vocabulary = import_vocab(path)
    if verbose:
        print("Target vocabulary imported!")
    return vocabulary

def import_same_vocab(path: Union[str, PathLike] = VOCAB_DIR+SOURCE_VOCAB_EXPORT) -> Tuple[Vocab, Vocab]:
    """Imports same vocabulary for both source and target vocabularies. Defaults to importing source vocabulary and pointing target vocabulary to it."""
    source_vocab = import_vocab(path)
    target_vocab = source_vocab
    return source_vocab, target_vocab

# Debugging functions

def test_vocab_obj():
    a = make_vocab_obj()
    print(a(["pain", "medicine", "adfhjahf"]))
    export_source_vocab(a, VOCAB_DIR+"dummyVocab.pth")
    b = import_source_vocab(VOCAB_DIR+"dummyVocab.pth")
    print(b.get_itos()==a.get_itos())
    print(b.get_default_index()==a.get_default_index())

def main():
    pass

if __name__ == "__main__":
    test_vocab_obj()