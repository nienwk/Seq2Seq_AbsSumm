# AI6127-Group-Project
This is a project on summarization for AI6127 Deep Neural Networks for Natural Language Processing.

## Environment Setup
From a command line, run:
`conda env create -f environment.yml`

Activate it using the command:
`conda activate AI6127-proj`

If you choose to use Anaconda Navigator, use the provided `environment.yml` for a guide to know what to install.

Do note that `nltk` may require manual setup to get working from a fresh install. Refer to the documentation for it if required.

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
The entry point of the program for training the summarization model is at ```train.py```.<br>
We highly suggest playing with the hyperparameter options supported to get better results. The default configuration provided is arbitrary as we were stress testing the code.

See `python train.py --help` for an extensive list of supported options.<br>
However, do take a look at the TODO list below to see if the supported option is *actually* supported and stable for use.

Although there is a helper bash script (`runModel1.sh`) provided to execute a sample code run on an Ubuntu 20.04 Linux OS, it is mainly used by the team to debug the code and may only be used as a guide.<br>

### On reproducibility
We have implemented support for ensuring reproducibility of the training/validation/testing results, via amenities provided to us by our core library framework, `PyTorch`.<br>
To get reproducible results, use the command line argument `-d --no-benchmark --pytorch-seed <SEED> --trainloader-seed <SEED>`, where `<SEED>` is a randomly selected longInteger seed. Note that it does not have to be the same seed for both options. We recommend using the return of a `torch.seed()` call in a `Python` shell to generate the seeds for better statistical properties (according to PyTorch documentation).<br>
Lastly, for full reproducibility involving RNN's (such as our model), it is required to set the environment variable `CUBLAS_WORKSPACE_CONFIG=:16:8` or `CUBLAS_WORKSPACE_CONFIG=:4096:2`. See PyTorch LSTM [documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) for more details.

## Test the Model
Once you are done training your model, use `test.py` to get the average test loss and test metrics across the test dataset.<br>
See `python test.py --help` for the required command line arguments to successfully execute the script.<br>
Alternatively, see `testModel1.sh` for an example of the required command.

## TO-DO List
- [x] Setup BucketIterator equivalent.
- [x] Prepare encoder models.
  - [x] Prepare dimension matching with decoder model. See `proj_size` of `LSTM` module.
- [x] Prepare decoder models
  - [x] Prepare attention layer inside decoder model
- [x] Prepare sequence-to-sequence models
  - [x] Prepare beam search in sequence-to-sequence model
  - [ ] Implement teacher-forcing
- [x] Setup saving utilities
- [x] Test saving utilities
- [ ] Test loading utility, for resuming checkpoints
- [x] Setup training regime
- [x] Test training regime
- [x] Setup testing regime
- [x] Test testing regime
- [ ] Setup plotting utilities for results display

## Extras
- [ ] Support more optimizer choices
- [ ] Support more scheduler choices
- [ ] Support different source and target vocabularies
- [ ] Support additional models