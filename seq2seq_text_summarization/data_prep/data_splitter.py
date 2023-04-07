from pandas import read_csv, DataFrame
from ..configs.data_prep_configs import POSTPROCESSED_CSV, POSTPROCESSING_DIR, FIELDNAME_CLEANED_TEXT, FIELDNAME_CLEANED_SUMMARY, FIELDNAME_CLEANED_TEXT_LEN, FIELDNAME_CLEANED_SUMM_LEN, SPLIT_SEED, SPLIT_RATIO, SHORT_SEQ, MEDIUM_SEQ, LONG_SEQ, TRAIN_CSV, VAL_CSV, TEST_CSV
from sklearn.model_selection import train_test_split
import spacy

def seq_len_category(length):
    if length <= SHORT_SEQ:
        return 0
    elif (length > SHORT_SEQ) & (length <= MEDIUM_SEQ):
        return 1
    elif (length > MEDIUM_SEQ ) & (length <= LONG_SEQ):
        return 2
    else:
        return -1

def generate_splits():
    print("-"*50)
    print("Reading data...")
    data = read_csv(POSTPROCESSING_DIR+POSTPROCESSED_CSV)
    print("Reading complete.")
    print("-"*50)
    print(f"Original dataset size is {len(data)}.")
    print("-"*50)

    nlp = spacy.load("en_core_web_sm").tokenizer
    data[FIELDNAME_CLEANED_TEXT_LEN] = data[FIELDNAME_CLEANED_TEXT].apply(lambda x: len(nlp(x)))
    data[FIELDNAME_CLEANED_SUMM_LEN] = data[FIELDNAME_CLEANED_SUMMARY].apply(lambda x: len(nlp(x)))
    data["input_seq_category"] = data[FIELDNAME_CLEANED_TEXT_LEN].apply(seq_len_category)
    data.drop(data[data["input_seq_category"]==-1].index, inplace=True) # we drop too long sequences, to prevent out-of-memory in training/testing
    print(f"After dropping too long input text (with length > {LONG_SEQ}), dataset size is {len(data)}.")
    print("-"*50)

    # # Uncomment to draw histogram
    # import matplotlib.pyplot as plt
    # plt.hist(data[FIELDNAME_CLEANED_TEXT_LEN], [0, SHORT_SEQ, MEDIUM_SEQ, LONG_SEQ])
    # plt.savefig("data_distribution_len_category.png", bbox_inches="tight")
    # plt.close()

    train_val, test = train_test_split(data, test_size=SPLIT_RATIO[2]/sum(SPLIT_RATIO), train_size=(SPLIT_RATIO[0]+SPLIT_RATIO[1])/sum(SPLIT_RATIO), random_state=SPLIT_SEED, shuffle=True, stratify=data["input_seq_category"])
    train, val = train_test_split(train_val, test_size=SPLIT_RATIO[1]/(SPLIT_RATIO[0]+SPLIT_RATIO[1]), train_size=SPLIT_RATIO[0]/(SPLIT_RATIO[0]+SPLIT_RATIO[1]), random_state=SPLIT_SEED+1, shuffle=True, stratify=train_val["input_seq_category"])
    
    print(f"Number of training samples obtained : {len(train)}")
    print(f"Number of short (length <= {SHORT_SEQ}) training input sequences : {len(train[train['input_seq_category']==0])}")
    print(f"Number of medium ({SHORT_SEQ} < length <= {MEDIUM_SEQ}) training input sequences : {len(train[train['input_seq_category']==1])}")
    print(f"Number of long ({MEDIUM_SEQ} < length <= {LONG_SEQ}) training input sequences : {len(train[train['input_seq_category']==2])}")
    print("-"*50)
    print(f"Number of validation samples obtained : {len(val)}")
    print(f"Number of short (length <= {SHORT_SEQ}) validation input sequences : {len(val[val['input_seq_category']==0])}")
    print(f"Number of medium ({SHORT_SEQ} < length <= {MEDIUM_SEQ}) validation input sequences : {len(val[val['input_seq_category']==1])}")
    print(f"Number of long ({MEDIUM_SEQ} < length <= {LONG_SEQ}) validation input sequences : {len(val[val['input_seq_category']==2])}")
    print("-"*50)
    print(f"Number of testing samples obtained : {len(test)}")
    print(f"Number of short (length <= {SHORT_SEQ}) testing input sequences : {len(test[test['input_seq_category']==0])}")
    print(f"Number of medium ({SHORT_SEQ} < length <= {MEDIUM_SEQ}) testing input sequences : {len(test[test['input_seq_category']==1])}")
    print(f"Number of long ({MEDIUM_SEQ} < length <= {LONG_SEQ}) testing input sequences : {len(test[test['input_seq_category']==2])}")
    print("-"*50)

    train_df = DataFrame(train[[FIELDNAME_CLEANED_TEXT, FIELDNAME_CLEANED_SUMMARY, FIELDNAME_CLEANED_TEXT_LEN, FIELDNAME_CLEANED_SUMM_LEN]])
    train_df.to_csv(POSTPROCESSING_DIR+TRAIN_CSV)
    val_df = DataFrame(val[[FIELDNAME_CLEANED_TEXT, FIELDNAME_CLEANED_SUMMARY, FIELDNAME_CLEANED_TEXT_LEN, FIELDNAME_CLEANED_SUMM_LEN]])
    val_df.to_csv(POSTPROCESSING_DIR+VAL_CSV)
    test_df = DataFrame(test[[FIELDNAME_CLEANED_TEXT, FIELDNAME_CLEANED_SUMMARY, FIELDNAME_CLEANED_TEXT_LEN, FIELDNAME_CLEANED_SUMM_LEN]])
    test_df.to_csv(POSTPROCESSING_DIR+TEST_CSV)

    print("Data split complete.")

def main():
    generate_splits()

if __name__ == "__main__":
    main()