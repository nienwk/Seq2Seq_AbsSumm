# AI6127-Group-Project
This is a project on summarization for AI6127 Deep Neural Networks for Natural Language Processing

## Environment Setup
Python version 3.9, Anaconda Navigator 2.4.0.<br>
On Anaconda Navigator, install ```pandas``` and ```nltk```<br>.
<br>
Open terminal for your current Anaconda Environment and install the following:
```
# Assumption: Your device can support CUDA and have CUDA toolkit installed
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Torchtext
conda install -c pytorch torchtext

# Spacy
conda install -c conda-forge spacy
conda install -c conda-forge spacy-model-en_core_web_sm

# Misc Utilities
conda install -c anaconda beautifulsoup4
conda install -c anaconda lxml
```

## Dataset Setup and Processing
Create a ```dataset/``` folder in the top directory of your local repository.<br>
Download [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and unzip the contents to ```dataset/``` folder.<br>
You should now have ```dataset/Reviews.csv``` file in your local repository.<br>

After the above, run the ```clean_reviews_csv.py``` script.<br>
This would generate a ```postprocessing/Reviews_processed.csv``` file which contains only valid rows from the original ```dataset/Reviews.csv``` file.<br>
There will also be two new columns:<br>
```cText``` which is the cleaned version of ```Text``` column.<br>
and ```cSummary``` which is the cleaned version of ```Summary``` column.<br>
<br>
You only need to generate ```postprocessing/Reviews_processed.csv``` once, unless there are changes to the way we process the data before training the model.
Model training would use this generated csv.

## Training the Model
The entry point of the program for training the Summarization model(s) is at ```main.py```.
