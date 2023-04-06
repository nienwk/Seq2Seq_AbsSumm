from seq2seq_text_summarization.data_prep.clean_reviews_csv import clean_reviews
from seq2seq_text_summarization.data_prep.data_splitter import generate_splits
from seq2seq_text_summarization.dataloaders.vocab import make_vocab_obj, export_same_vocab

def main():
    clean_reviews()
    generate_splits()
    export_same_vocab(make_vocab_obj())


if __name__=="__main__":
    main()