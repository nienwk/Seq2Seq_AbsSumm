# Main Entry point for Model Training and Evaluation

import pandas as pd
import spacy
import torch
import globalconstants
from dataset import AmazonFineFoodDataset, collate_fn, KEY_WV_REVIEW, KEY_WV_SUMMARY
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

SEED = 12345
TRAIN_VAL_TEST_SPLIT = [0.7, 0.1, 0.2]
BATCH_SIZE = 64


if __name__ == "__main__":
    nlp=spacy.load("en_core_web_sm")

    print('Loading Dataset')
    dataset = AmazonFineFoodDataset(file_path=globalconstants.POSTPROCESSING_DIR+globalconstants.POSTPROCESSED_CSV, tokenizer=nlp)
    print('Dataset Loaded')

    seed_gen = torch.Generator().manual_seed(SEED)

    # Load the dataset and split into train/val/test
    train_data, val_data, test_data = random_split(dataset, TRAIN_VAL_TEST_SPLIT, generator=seed_gen)
    print('Dataset Split Done   Train: ' + str(len(train_data)) + ' Val: ' + str(len(val_data)) + ' Test: ' + str(len(test_data)))

    # Use the PyTorch DataLoader to load the data in batches
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Loop through training dataloader
    for batch in train_dataloader:
        wv_review = batch[KEY_WV_REVIEW]
        wv_summary = batch[KEY_WV_SUMMARY]
        
        # TEST CODE
        print(wv_review.shape)
        print(wv_summary.shape)

        # Process the batch here


    # Loop through validation dataloader
    for batch in val_dataloader:
        wv_review = batch[KEY_WV_REVIEW]
        wv_summary = batch[KEY_WV_SUMMARY]
        
        # TEST CODE
        print(wv_review.shape)
        print(wv_summary.shape)
        
        # Process the batch here

    # Loop through validation dataloader
    for batch in test_dataloader:
        wv_review = batch[KEY_WV_REVIEW]
        wv_summary = batch[KEY_WV_SUMMARY]
        
        # TEST CODE
        print(wv_review.shape)
        print(wv_summary.shape)
        
        # Process the batch here