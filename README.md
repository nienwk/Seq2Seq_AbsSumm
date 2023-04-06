# AI6127-Group-Project
This is a project on summarization for AI6127 Deep Neural Networks for Natural Language Processing

## Environment Setup
From a command line, run:
`conda env create -f environment.yml`

Activate it using the command:
`conda activate AI6127-proj`

If you choose to use Anaconda Navigator, use the provided `environment.yml` for a guide to know what to install.

## Dataset Setup and Processing
Create a `data/` folder in the `seq2seq_text_summarization` directory of your local repository.<br>
Download [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and unzip the contents to ```data/``` folder.<br>
You should now have ```data/Reviews.csv``` file in `seq2seq_text_summarization` directory of your local repository.<br>

After the above, run the ```prepData.py``` script.<br>
This would generate the train, validation and test splits of the data, while printing some statistics.<br>
The file names of the splits are `train.csv`, `val.csv` and `test.csv`.<br>
In each of the output csv's, there will be some columns to note:
- ```cText``` which is the cleaned version of ```Text``` column.
- ```cSummary``` which is the cleaned version of ```Summary``` column.
- `input_seq_len` which is the tokenized length of the cleaned version of `Text`.
- `summ_seq_len` which is the tokenized length of the cleaned version of `Summary`.

You only need to execute ```prepData.py``` once, unless there are changes to the way we process the data before training the model.
Model training would use the generated csv's, namely:
- `train.csv`
- `val.csv`
- `test.csv`

## Training the Model
The entry point of the program for training the Summarization model(s) is at ```train.py```.

## TO-DO List
- [x] Setup BucketIterator equivalent.
- [ ] Prepare encoder models.
  - [ ] Prepare dimension matching with decoder model. See `proj_size` of `LSTM` module.
- [ ] Prepare decoder models
  - [ ] Prepare attention layer inside decoder model
- [ ] Prepare sequence-to-sequence models
  - [ ] Prepare beam search in sequence-to-sequence model
- [x] Setup saving utilities
- [ ] Test saving utilities
- [x] Setup training regime
- [ ] Test training regime
- [ ] Setup testing regime
- [ ] Test testing regime