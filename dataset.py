import pandas as pd
import torch
import globalconstants
from torch.utils.data import Dataset


KEY_WV_REVIEW = 'wv_review'
KEY_WV_SUMMARY = 'wv_summary'


class AmazonFineFoodDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(file_path)
        self.data = self.data[[globalconstants.FIELDNAME_CLEANED_TEXT, globalconstants.FIELDNAME_CLEANED_SUMMARY]]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Get Word Vector for Full Review
        review = row[globalconstants.FIELDNAME_CLEANED_TEXT]
        doc_review = self.tokenizer(review)
        wv_review = torch.stack([torch.from_numpy(token.vector) for token in doc_review])

        # Get Word Vector for Summary
        summary = row[globalconstants.FIELDNAME_CLEANED_SUMMARY]
        doc_summary = self.tokenizer(summary)
        wv_summary = torch.stack([torch.from_numpy(token.vector) for token in doc_summary])
        
        return {KEY_WV_REVIEW: wv_review, KEY_WV_SUMMARY: wv_summary}


def collate_fn(batch):

    # --- Get Reviews in batch to be of same length 
    batch_review = []
    batch_summary = []


    # Get Max Review Length
    max_review_len = 0
    max_summary_len = 0
    for item in batch:
        review = item[KEY_WV_REVIEW]
        summary = item[KEY_WV_SUMMARY]
        if max_review_len < review.size(0):
            max_review_len = review.size(0)
        if max_summary_len < summary.size(0):
            max_summary_len = summary.size(0)

    for item in batch:
        review = item[KEY_WV_REVIEW]
        summary = item[KEY_WV_SUMMARY]

        # pad the tensors along the first dimension with zeros
        review_padded = torch.zeros((max_review_len, review.size(1)), device=review.device, dtype=review.dtype)
        review_padded[:review.size(0), :] = review

        summary_padded = torch.zeros((max_summary_len, summary.size(1)), device=summary.device, dtype=summary.dtype)
        summary_padded[:summary.size(0), :] = summary

        batch_review.append(review_padded)
        batch_summary.append(summary_padded)

    batch_review = torch.stack(batch_review)
    batch_summary = torch.stack(batch_summary)
    

    return {KEY_WV_REVIEW: batch_review, KEY_WV_SUMMARY: batch_summary}