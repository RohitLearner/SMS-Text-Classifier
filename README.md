

# Neural Network SMS Text Classifier
This repository contains my solution approach from the  [FreeCodeCamp](https://www.freecodecamp.org/learn/) Machine Learning with Python Project - [SMS Text Classifier](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/neural-network-sms-text-classifier) challenge. ( ⭐️ Star repo on GitHub — it helps! )

<img src="https://miro.medium.com/max/700/1*Fm58r_RQ53sEHfwFa28LpA.png" align="center" height=80% width = 90%>
 
In this challenge, we will create a machine learning model that will classify SMS messages as either "ham" or "spam". A "ham" message is a normal message sent by a friend. A "spam" message is an advertisement or a message sent by a company.. 

We need to create a function called `predict_message` that takes a message string as an argument and returns a list. The first element in the list should be a number between zero and one that indicates the likeliness of "ham" (0) or "spam" (1). The second element in the list should be the word "ham" or "spam", depending on which is most likely.

For this challenge, you will use the [SMS Spam Collection dataset](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). The dataset has already been grouped into train data and test data.

You can access [the full project instructions and starter code on Google Colaboratory](https://colab.research.google.com/github/freeCodeCamp/boilerplate-neural-network-sms-text-classifier/blob/master/fcc_sms_text_classification.ipynb).

Create a copy of the notebook either in your own account or locally. Once you complete the project and it passes the test (included at that link), "HAVE FUN WHILE SOLVING IT". If you are submitting a Google Colaboratory link, make sure to turn on link sharing for "anyone with the link."

## Code
All code section are available directly and the detailed description of the data can be found in [colab](https://colab.research.google.com/drive/1F6T2j8A17jZgANg4iHlePa7hH5TKxZgw).

The project needs the following header files for the implementation:

 1. import numpy as np
 2. import pandas as pd
 3. import matplotlib.pyplot as plt
 4. import tensorflow as tf
 5. from tensorflow import keras
 6. import tensorflow_datasets as tfds
 7. from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
 8. import seaborn as sns
 9. from tensorflow.keras.preprocessing.text import Tokenizer
 10. from tensorflow.keras.preprocessing.sequence import pad_sequences
 11. from tensorflow.keras.callbacks import EarlyStopping
 12. from tensorflow.keras.models import Sequential
 13. from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

To run the project, do the following steps:

 1. Dowload the repo.
 2. Install all the above mentioned libraries.
 3. `python3 fcc_sms_text_classification.py`
 4. Check if following output is present in the end.
> You passed the challenge. Great job!
## Diagrams and Results
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/Accuracy_Graph.png" align="center">
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/Loss_graph.png" align="center">

**Spam Wordcloud**
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/spam_wordcloud.png" align="center">

**Ham Wordcloud**
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/ham_wordcloud.png" align="center">
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/imbalanced%20data.png" align="center">
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/downsampling.png" align="center">

**Model Summary**
<img src="https://raw.githubusercontent.com/RohitLearner/SMS-Text-Classifier/main/Diagram/Model_Summary.png" align="center">

    Train Dataset Accuracy:
    loss: 0.0826 - accuracy: 0.9777 
    Test Dataset Accuracy:
    loss: 0.1043 - accuracy: 0.9662

## Contributors

- Rohit Kumar Singh

## Feedback
Feel free to send us feedback on [file an issue](https://github.com/RohitLearner/SMS-Text-Classifier/issues). Feature requests are always welcome. If you wish to contribute, please take a quick look at this [colab](https://colab.research.google.com/drive/1F6T2j8A17jZgANg4iHlePa7hH5TKxZgw#scrollTo=Dxotov85SjsC).

> Written with [StackEdit](https://stackedit.io/).
