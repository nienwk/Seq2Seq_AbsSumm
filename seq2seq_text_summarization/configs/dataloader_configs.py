# Config file for dataloading and vocab building

# Batch sizes, based on data set sizes for nicer batch splits
TRAIN_BATCH_SIZE = 8 # based on 54712 training samples
VAL_BATCH_SIZE = 24 # based on 6840 validation samples
TEST_BATCH_SIZE = VAL_BATCH_SIZE

# Dataloader parallel loading, consumes large amounts of RAM based on batch size
NUM_WORKERS = 2

# Multiplier for BucketSampler
BUCKET_MULTIPLIER = 100

# Special tokens to be defined here
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

# Tokenizer
TOKENIZER = "spacy"
LANGUAGE = "en_core_web_sm"

# Vocab dictionary paths
VOCAB_DIR = "seq2seq_text_summarization/dataloaders/"
SOURCE_VOCAB_EXPORT = "source_vocab.pth"
TARGET_VOCAB_EXPORT = "target_vocab.pth"