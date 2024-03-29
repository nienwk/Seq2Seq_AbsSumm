# This script is for cleaning Reviews.csv from Amazon Fine Food Reviews Dataset

import numpy as np  
import pandas as pd
import os
import re
import nltk
from ..configs.data_prep_configs import ORIGINAL_DATASET_PATH, FIELDNAME_CLEANED_TEXT, FIELDNAME_CLEANED_SUMMARY, POSTPROCESSING_DIR, POSTPROCESSED_CSV
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
import warnings

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                        "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                        "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                        "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                        "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                        "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                        "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                        "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                        "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                        "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                        "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                        "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                        "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                        "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                        "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                        "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                        "you're": "you are", "you've": "you have"}

# Text Cleaning
def text_cleaner(text):
    stop_words = set(stopwords.words('english')) 
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)

    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 

    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

# Summary Cleaning
def summary_cleaner(text):

    newString = re.sub('"','', text)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)

    newString = newString.lower()

    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in newString.split() if not w in stop_words]

    newString=''
    for i in tokens:
        if len(i)>1:                                 
            newString=newString+i+' '  
    return newString

def clean_reviews():
    pd.set_option("display.max_colwidth", 200)
    warnings.filterwarnings("ignore")
    nltk.download('stopwords')

    path_input = ORIGINAL_DATASET_PATH
    print("Begin cleaning: " + path_input)
    df=pd.read_csv(ORIGINAL_DATASET_PATH, nrows=100000)

    df.drop_duplicates(subset=['Text'],inplace=True)  #dropping duplicates
    df.dropna(axis=0,inplace=True)   #dropping na

    # Call text_cleaner function
    cleaned_text = []
    for t in df['Text']:
        cleaned_text.append(text_cleaner(t))

    # Call summary_cleaner function
    cleaned_summary = []
    for t in df['Summary']:
        cleaned_summary.append(summary_cleaner(t))

    df2 = pd.DataFrame()
    df2[FIELDNAME_CLEANED_TEXT]=cleaned_text
    df2[FIELDNAME_CLEANED_SUMMARY]=cleaned_summary
    df2[FIELDNAME_CLEANED_SUMMARY].replace('', np.nan, inplace=True)

    df2.dropna(axis=0,inplace=True)
    #data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x : '_START_ '+ x + ' _END_')

    print("Size after processing: " + str(df2.shape[0]))

    print("Writing new csv with cleaned data...")
    isExist = os.path.exists(POSTPROCESSING_DIR)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(POSTPROCESSING_DIR)

    path_output = POSTPROCESSING_DIR+POSTPROCESSED_CSV
    df2.to_csv(path_output, index=False)

    print("CSV Output Ready! Path: " + path_output)

def main():
    clean_reviews()

if __name__ == "__main__":
    main()