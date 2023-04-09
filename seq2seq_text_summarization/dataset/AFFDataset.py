from typing import Union
from os import PathLike
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from pandas import read_csv
from ..configs.data_prep_configs import FIELDNAME_CLEANED_TEXT, FIELDNAME_CLEANED_SUMMARY, FIELDNAME_CLEANED_TEXT_LEN, FIELDNAME_CLEANED_SUMM_LEN

class AmazonFineFoodDataset(Dataset):
    def __init__(
        self,
        tokenizer, # Should not be inputting spacy.load() directly. Use `setup_tokenizer()` in `dataloaders.vocab.py`
        vocabulary: Vocab,
        end_token_idx: list[int],
        file_path: Union[str, PathLike],
        text_col_name: str = FIELDNAME_CLEANED_TEXT,
        summ_col_name: str = FIELDNAME_CLEANED_SUMMARY,
        input_len_col_name: str = FIELDNAME_CLEANED_TEXT_LEN,
        summ_len_col_name: str = FIELDNAME_CLEANED_SUMM_LEN,
        ) -> None:
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.end_token_idx = end_token_idx
        self.text_col_name = text_col_name
        self.summ_col_name = summ_col_name
        self.input_len_col_name = input_len_col_name
        self.summ_len_col_name = summ_len_col_name
        self.data = read_csv(file_path)[[text_col_name, summ_col_name, input_len_col_name, summ_len_col_name]]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        review, summary = row[self.text_col_name], row[self.summ_col_name]
        review, summary = self.vocabulary(self.tokenizer(review)), self.vocabulary(self.tokenizer(summary))+self.end_token_idx # append END TOKEN here
        return review, summary
    
    def get_seq_len(self):
        return self.data[[self.input_len_col_name, self.summ_len_col_name]]