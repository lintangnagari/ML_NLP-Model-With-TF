# %% [markdown]
# # **FIRST PROJECT : BUILD MODEL NLP WITH TENSORFLOW 👨🏽‍💻**

# %% [markdown]
# * Name         : Lintang Nagari
# * Email        : unggullintangg@gmail.com
# * Linkedin     : <a href='https://www.linkedin.com/in/lintangnagari/'>Lintang Nagari</a>
# * Github       : <a href='https://github.com/lnt-ngr'>lnt-ngr</a>

# %% [markdown]
# **Here are the submission criteria you must meet:**
# 
# * Use any dataset, but it must have at least 1000 samples.
# * Must use LSTM in the model architecture.
# * Must use a `sequential model`.
# * Validation set **should be 20%** of the total dataset.
# * Must use Embedding.
# * Must use a tokenizer function.
# * Model accuracy should be **at least 75%** on both the train set and validation set.
# 
# **Dataset : https://www.kaggle.com/datasets/crxxom/daily-google-news/download?datasetVersionNumber=4**
# 
# **About Dataset**
# 
# This dataset contains metadata of millions of news articles from Google News, including `title, publisher, DateTime, link, and category`.
# 
# This is also an automation project in which data is scraped every day at 4am UTC on 8 major categories. This dataset is expected to have a monthly update, thus the data collected daily will be merged into a single monthly csv file and published on Kaggle at the end of each month. One may expect the value of the dataset to continuously grow through time.

# %% [markdown]
# ### __IMPORT LIBRARY__

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
import zipfile
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

print(tf.__version__)

# %% [markdown]
# ### __EXTRACT FILE & MERGE FILE__

# %%
# News Data
berita1 = pd.read_csv('./Berita/2023_9.csv')
berita2 = pd.read_csv('./Berita/2023_10.csv')
berita3 = pd.read_csv('./Berita/2023_11.csv')
berita4 = pd.read_csv('./Berita/2023_12.csv')

Berita= [berita1, berita2, berita3, berita4]

# %%
# Merge Data
df = pd.concat(Berita, axis=0, ignore_index=True)
df.head()

# %%
df.isna().any()


# %%
df = df.dropna()
df.info()

# %%
# To remove unnecessary data

df = df.drop(columns=[
    'Publisher', 'DateTime', 'Link'
])
df['Category'].value_counts()

# %%
#To remove special characters from the "Title" column
df['Title'] = df['Title'].map(lambda x: re.sub(r'\W+', ' ', x))
df.head()

# %% [markdown]
# ## __DATA VISUALIZATION__

# %%
plt.style.use('fivethirtyeight')
mpl.rcParams['grid.color'] = "black"
df['Category'].value_counts().plot(kind = 'bar')
plt.show()

# %% [markdown]
# ## __ONE HOT ENCODING__

# %%
Category = pd.get_dummies(df.Category, dtype=int)
df = pd.concat([df, Category], axis=1)
df = df.drop(columns='Category', axis=1)
df.head()

# %% [markdown]
# ## __Converting a DataFrame to NumPy array__

# %%
Title = df['Title'].values
Category = df[['Sports', 'Headlines', 'Entertainment',
               'Business', 'Worldwide', 'Technology', 
               'Health', 'Science']].values

# %% [markdown]
# ## __SPLIT DATA INTO TRAINING & TESTING__

# %%
Title_train, Title_test, Category_train, Category_test = train_test_split(Title, Category, test_size = 0.2, random_state=123, shuffle=True)  

# %% [markdown]
# ## __TOKENIZER FUNCTION__

# %%
vocab_size = 50000

tokenizer = Tokenizer(num_words=vocab_size, oov_token="x")
tokenizer.fit_on_texts(Title_train)
tokenizer.fit_on_texts(Title_test)

sekuens_train = tokenizer.texts_to_sequences(Title_train)
padded_train = pad_sequences(sekuens_train, padding='post', maxlen=100, truncating='post')

sekuens_test = tokenizer.texts_to_sequences(Title_test)
padded_test = pad_sequences(sekuens_test, padding='post', maxlen=100, truncating='post')



# %% [markdown]
# ## __CALLBACK__

# %%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90 and logs.get('val_accuracy')>0.90):
      print("\nAkurasi dan validasi telah mencapai nilai > 90%!")
      self.model.stop_training = True
callbacks = myCallback()

# %% [markdown]
# ## __COMPILE FUNCTION, OPTEMIZER & LOSS FUNCTION__

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256, input_length=100),
    tf.keras.layers.LSTM(64, return_sequences=True, batch_input_shape=(128, 100, 200)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# %% [markdown]
# ## __FIT FUNCTION__

# %%
num_epochs =10
history = model.fit(padded_train, Category_train, epochs=num_epochs, 
                 validation_data=(padded_test, Category_test), verbose=1,
                 callbacks=[callbacks])

# %% [markdown]
# ## __PLOTTING__

# %%
#loss Plot
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Plot')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()

# %%
#Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Plot')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()


