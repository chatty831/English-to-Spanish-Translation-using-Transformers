# Spanish-to-English-Translation-using-Transformers
# Aim of the notebook.
Attention-based sequence-to-sequence model that can effectively understand the context of Spanish sentences and translate them into clear and coherent English sentences.

# Downloading the dataset

The dataset used is a paired corpus of **English-Spanish**, provided by [Anki](https://www.manythings.org/anki/).

The code starts by downloading a zip file containing the dataset for English to Spanish translation. The dataset is stored in the `spa-eng` folder and can be found in the file named `spa.txt`.

Let's take a look at how the data looks like:

![Sample Image](https://github.com/chatty831/English-to-Spanish-Translation-using-Transformers/blob/5f040ced1a12c661600497fcdc580a538168d1ea/Data_sample.png)

- Each line in the `spa.txt` file contains an English word/sentence and their corresponding Spanish translation.

- Some words might have multiple translation because of context. 

- Our first objective is to extract each line, and then separate the  English and Spanish words/sentences into two separate arrays. These will act as our input and target sentences for training the model

# Model Checkpoint
The trained model checkpoint can be found at the [drive](https://drive.google.com/file/d/1HZ6zTHWkAKe-yHjBBagmQECNXMF9A1OX/view?usp=sharing) link.
Put the downloaded file in the `/training_checkpoints/` and you can use the checkpoint restore to restore the model again.
