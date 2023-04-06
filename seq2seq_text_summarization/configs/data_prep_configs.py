# Directories and File Locations
ORIGINAL_DATASET_PATH = 'seq2seq_text_summarization/data/Reviews.csv'
POSTPROCESSING_DIR = 'seq2seq_text_summarization/data/'
POSTPROCESSED_CSV = 'Reviews_processed.csv'
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"

# Field name in CSV with cleaned text
FIELDNAME_CLEANED_SUMMARY = 'cSummary'
FIELDNAME_CLEANED_TEXT = 'cText'
FIELDNAME_CLEANED_TEXT_LEN = "input_seq_len"
FIELDNAME_CLEANED_SUMM_LEN = "summ_seq_len"

# Train-val-test NumPy generated arbitrary seed state
SPLIT_SEED = 3666292228

# Train-val-test ratio
SPLIT_RATIO = (0.8, 0.1, 0.1)

# Input sequence length thresholds for stratification in splitting
SHORT_SEQ = 10
MEDIUM_SEQ = 30
LONG_SEQ = 50
