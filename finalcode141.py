#!/usr/bin/env python
# coding: utf-8

# # 141C Final Project – Predict Text Complexity
# Quilvio Hernandez, Xing Yang Lan, Aditya Kallepalli

# Inspirations:
# - [Guide to learn NLP](https://www.kaggle.com/donmarch14/commonlit-detailed-guide-to-learn-nlp)
# - [Clean Visualizations](https://www.kaggle.com/gaetanlopez/how-to-make-clean-visualizations)
# - [Understanding the Competition](https://www.kaggle.com/heyytanay/commonlit-eda-understanding-the-competition)

# ## Libraries

# In[1]:


# Array Support packages
import numpy as np
import pandas as pd

# Progress bar
# https://tqdm.github.io/docs/notebook/
from tqdm.notebook import tqdm

# Helper packages
import string
import time
import math
import re
from random import randint
from collections import Counter

# sci-kit learn functions for Linear Models
# https://scikit-learn.org/stable/
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# TF functions for NN models
# https://www.tensorflow.org/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
# https://www.tensorflow.org/tutorials/keras/keras_tuner
import kerastuner as kt

# NLP packages
# https://www.nltk.org/
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')



# https://nlpretext.readthedocs.io/en/latest/
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (unpack_english_contractions, 
                                        normalize_whitespace, 
                                        remove_punct, 
                                        lower_text, 
                                        remove_eol_characters)

# Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# Config setting
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('fivethirtyeight')
sns.set_palette(sns.color_palette('pastel'))


# In[2]:


sns.palplot(sns.color_palette('pastel'))


# ## EDA

# In[3]:


# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[4]:


print('The shape of the training dataset is %s' % (train.shape,))
print('There are %d missing values in the excerpt column' % train.excerpt.isna().sum())
print('There are %d missing values in the target column' % train.target.isna().sum())
train.head()


# In[5]:


print(test.shape)
test.head()


# In[6]:


target_min = train.target.min()
target_max = train.target.max()
print('The minimum value in the target column is %f' % target_min)
print('The maximum value in the target column is %f' % target_max)


# In[7]:


easy_hard = train[(train.target == target_min )| (train.target == target_max)].reset_index(drop=True)
print('The hardest passage corresponds to the target value %f. The passage is below:\n %s' % (target_min, easy_hard.excerpt[0]))
print('\nThe easy passage corresponds to the target value %f. The passage is below:\n %s' % (target_max, easy_hard.excerpt[1]))


# In[8]:


def target_sample(n, data = train):
    """
    Examine n samples from the train dataset.
    For convenience, do not take n greater than 5.
    """
    for i in range(n):
        sample = randint(0, len(data))
        print('Sample: ' + str(sample) + "\n" +              'Target: ' + str(data.iloc[sample]["target"]) + '\n' +              'Text: ' + data.iloc[sample]["excerpt"] + '\n\n'
             )
# Look at samples from training dataset
target_sample(3)


# In[9]:


se_min = train.standard_error.min()
se_max = train.standard_error.max()
print('The minimum value in the standard error column is %f' % se_min)
print('The maximum value in the standard error column is %f' % se_max)


# In[10]:


agree_disagree = train[(train.standard_error == se_min )| (train.standard_error == se_max)].reset_index(drop=True)
print('The passage corresponds to the standard error value of %f. This suggests all readers agreed on the difficulty level. The passage is below:\n %s' % (se_min, agree_disagree.excerpt[0]))
print('\nThe passage corresponds to the standard error value of %f. This suggests readers disagreed the most on the difficulty level of this passage compared to the others. The passage is below:\n %s' % (se_max, agree_disagree.excerpt[1]))


# In[11]:


# Distribution plot of the target variable
fig, ax = plt.subplots(figsize=(20,8))
sns.histplot(train.target, kde = True)
ax.set_ylabel('')
ax.grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
ax.text(-3.75,260,'Distribution of the target variable',fontsize=23,fontweight='bold', fontfamily='monospace')
ax.text(-3.75,250,'Most of the passages in the dataset have a reading ease score between -2 and 0',fontsize=17,fontweight='light', fontfamily='monospace')

plt.tight_layout()
plt.savefig('target_dist.png')


# In[12]:


fig, ax = plt.subplots(figsize=(5, 8))
ax = sns.boxplot(y = 'target', data = train)
ax = sns.swarmplot(y = 'target', data = train, size = 4, color = sns.color_palette('pastel')[1], alpha = .8)


plt.tight_layout()

plt.savefig('target_box.png')


# In[13]:


fig, ax = plt.subplots(figsize=(20,8))
sns.histplot(train.standard_error, kde = True)
ax.set_ylabel('')
ax.grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
ax.text(-0.025,265,'Distribution of the standard error variable',fontsize=23,fontweight='bold', fontfamily='monospace')
ax.text(-0.025,255,'Most of the passages in the dataset have a standard error between .45 and .5',fontsize=17,fontweight='light', fontfamily='monospace')

plt.tight_layout()
plt.savefig('se_dist.png')


# In[14]:


fig, ax = plt.subplots(figsize=(5, 8))
ax = sns.boxplot(y = 'standard_error', data = train)
ax = sns.stripplot(y = 'standard_error', data = train, size = 3, color = sns.color_palette('pastel')[1], alpha = .4)


plt.tight_layout()
plt.savefig('se_box.png')


# In[15]:


ax = sns.jointplot(x = 'target', y = 'standard_error', kind = 'hist', data = train, height = 8)
plt.ylim(.4, .65)
plt.tight_layout()
plt.savefig('jointplot.png')


# In[16]:


train['excerpt_char_count'] = train['excerpt'].apply(
    lambda x : len(x)
)
train['excerpt_word_count'] = train['excerpt'].apply(
    lambda x : len(x.split(' '))
)


# In[17]:


fig, ax = plt.subplots(figsize=(20,8))
sns.histplot(train.excerpt_char_count, kde = True, line_kws = {'linewidth':6})
ax.set_ylabel('')
ax.grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
ax.text(660,230,'Distribution of the excerpt lengths',fontsize=23,fontweight='bold', fontfamily='monospace')
ax.text(660,220,'Most of the passages in the dataset have between 800 and 1100 characters',fontsize=17,fontweight='light', fontfamily='monospace')

plt.tight_layout()
plt.savefig('excerpt_len.png')


# In[18]:


fig, ax = plt.subplots(figsize=(20,8))
sns.histplot(train.excerpt_word_count, kde = True, line_kws = {'linewidth':6})
ax.set_ylabel('')
ax.grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
ax.text(125,265,'Distribution of the excerpt word counts',fontsize=23,fontweight='bold', fontfamily='monospace')
ax.text(125,255,'Excerpt word counts range from 130 to 210',fontsize=17,fontweight='light', fontfamily='monospace')

plt.tight_layout()
plt.savefig('word_count.png')


# In[20]:


# from wordcloud import WordCloud

# corpus = list(train['excerpt'].values)
# clean_named = (' '.join(str(w) for w in corpus))
# type(clean_named)

# wordcloud = WordCloud(width = 1200, height = 600,
#                 background_color ='white',
#                 min_font_size = 10).generate(clean_named)

# # plot the WordCloud image                       
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
# plt.savefig("wordcloud.png")
# plt.show()


# ## Data Cleaning

# In[19]:


# nlpretext functions to clean text
preprocessor = Preprocessor()
preprocessor.pipe(unpack_english_contractions)
preprocessor.pipe(remove_eol_characters)
preprocessor.pipe(lower_text)
preprocessor.pipe(remove_punct)
preprocessor.pipe(normalize_whitespace)


# In[20]:


# custom function to remove stopwords using nltk dictionary
def no_stopwords(passage):
    passage = ' '.join(word for word in passage.split(' ') if word not in stop_words)
    return passage


# In[21]:


# custom function to stem words using nltk.SnowballStemmer
stemmer = nltk.SnowballStemmer("english")

def nltk_stem(passage):
    passage = ' '.join(stemmer.stem(word) for word in passage.split(' '))
    return passage


# In[22]:


# apply all the steps above in one function
def preprocess_data(passage):
    passage = passage.apply(preprocessor.run)
    passage = passage.apply(no_stopwords)
    passage = passage.apply(nltk_stem)
    
    return passage


# In[23]:


train['clean_text'] = preprocess_data(train['excerpt'])
print('Text before cleaning: \n' + train['excerpt'][0] + '\n')
print('Text after cleaning: \n' + train['clean_text'][0])


# ## Linear Models

# In[24]:


# partition the target variable into 10 discrete intervals
train['target_bin'] = pd.cut(train.target, 10, labels=[i for i in range(10)])

# placeholder column to be filled by for loop
train['k_fold'] = 0

# Generate the train/test indices to split the data
skf = StratifiedKFold(n_splits=10, random_state=32, shuffle=True)
gen_fold = skf.split(train.id, y=train.target_bin)

# Assign k-fold for train and test
for k_fold, (train_index, val_index) in enumerate(gen_fold):
    train.loc[val_index, 'k_fold'] = k_fold

train['k_fold'] = train['k_fold'].astype('int8')


# In[25]:


train.head()


# In[26]:


t = train[train["k_fold"]!=0].reset_index(drop=True)
v = train[train["k_fold"]==0].reset_index(drop=True)

t_excerpt, t_score = t['excerpt'].values, t['target'].values
v_excerpt, v_score = v['excerpt'].values, v['target'].values

t_clean, t_score = t['clean_text'].values, t['target'].values
v_clean, v_score = v['clean_text'].values, v['target'].values


# In[27]:


print(t_excerpt.shape, v_excerpt.shape)
print(t_score.shape, v_score.shape)


# ### All Text

# #### Ridge

# In[28]:


# Model to use 
ridge_base = Ridge(fit_intercept=True, normalize=False)

# Create pipeline to go through steps 
ridge_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1, 1)),
    ridge_base
)

# Train model
ridge_pipeline.fit(t_excerpt, t_score)

# Predict and calculate MSE
pred = ridge_pipeline.predict(v_excerpt)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using Ridge and TfIdfVectorizer: %0.5f" % rmse)


# #### Linear

# In[29]:


# Model to use 
linear_base = LinearRegression(fit_intercept=True, normalize=False)

# Create pipeline to go through steps 
linear_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1, 1)),
    linear_base
)

# Train model
linear_pipeline.fit(t_excerpt, t_score)

# Predict and calculate MSE
pred = linear_pipeline.predict(v_excerpt)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using LinearRegression and TfIdfVectorizer: %0.5f" % rmse)


# #### Lasso

# In[30]:


# Model to use 
lasso_base = Lasso(fit_intercept=True, normalize=False)

# Create pipeline to go through steps 
lasso_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1, 1)),
    lasso_base
)

# Train model
lasso_pipeline.fit(t_excerpt, t_score)

# Predict and calculate MSE
pred = lasso_pipeline.predict(v_excerpt)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using Lasso and TfIdfVectorizer: %0.5f" % rmse)


# #### ElasticNet

# In[31]:


# Model to use 
elastic_base = ElasticNet(fit_intercept=True, normalize=False)

# Create pipeline to go through steps 
elastic_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1, 1)),
    elastic_base
)

# Train model
elastic_pipeline.fit(t_excerpt, t_score)

# Predict and calculate MSE
pred = elastic_pipeline.predict(v_excerpt)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using ElasticNet and TfIdfVectorizer: %0.5f" % rmse)


# Elastic and Lasso perform poorly. Ignore them going forward.

# Train models with cleaned data

# ### Clean data

# In[32]:


# Model to use 
ridge_base = Ridge(fit_intercept=True, normalize=False)

# Create pipeline to go through steps 
ridge_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1, 1)),
    ridge_base
)

# Train model
ridge_pipeline.fit(t_clean, t_score)

# Predict and calculate MSE
pred = ridge_pipeline.predict(v_clean)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using Ridge and TfIdfVectorizer: %0.5f" % rmse)


# In[33]:


# Model to use 
linear_base = LinearRegression(fit_intercept=True, normalize=False)

# Create pipeline to go through steps 
linear_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1, 1)),
    linear_base
)

# Train model
linear_pipeline.fit(t_clean, t_score)

# Predict and calculate MSE
pred = linear_pipeline.predict(v_clean)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using LinearRegression and TfIdfVectorizer: %0.5f" % rmse)


# Minimal improvement for ridge. Worse for Linear

# ### Grid Search for Ridge

# Previous ran for Tfidf as well and found nothing better than parameters we've been using (default).
# 
# 'tfidfvectorizer__lowercase': [True, False],
# 'tfidfvectorizer__stop_words': ['english', None],
# 'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
# 'tfidfvectorizer__norm': ['l1', 'l2']
# {'ridge__alpha': 0.01, 'ridge__max_iter': 1000, 'tfidfvectorizer__lowercase': False, 'tfidfvectorizer__ngram_range': (1, 1), 'tfidfvectorizer__norm': 'l1', 'tfidfvectorizer__stop_words': None}

# In[35]:


ridge_optim = Ridge(fit_intercept=True, normalize=False)
ridge_params = [{
    'ridge__alpha': [.000001, .00001, .0001, .001, .005, .01, .05, .1, .5, 1, 10],
    'ridge__max_iter': [1000,100000]}]
ridge_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, lowercase=False, ngram_range = (1,1), norm = 'l1', stop_words = None),
    ridge_optim
)

# Train model
search = GridSearchCV(ridge_pipeline, ridge_params, n_jobs=-1)
search.fit(t_clean, t_score)

# Evaluate the performance on validation set
pred = search.predict(v_clean)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using Ridge and TfIdfVectorizer: %0.5f" % rmse)
print(search.best_params_)


# ### Count Vectorizer

# In[36]:


ridge_best = Ridge(alpha = .01, fit_intercept=True, normalize=False, max_iter = 1000)
ridge_pipeline = make_pipeline(
    CountVectorizer(binary=True, lowercase=False, ngram_range = (1,1), stop_words = None),
    ridge_best
)

# Train model
ridge_pipeline.fit(t_clean, t_score)

# Predict and calculate MSE
pred = ridge_pipeline.predict(v_clean)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using Ridge and CountVectorizer: %0.5f" % rmse)


# ### Best

# In[70]:


ridge_best = Ridge(alpha = .01, fit_intercept=True, normalize=False, max_iter = 1000)
ridge_pipeline = make_pipeline(
    TfidfVectorizer(binary=True, lowercase=False, ngram_range = (1,1), norm = 'l1', stop_words = None),
    ridge_best
)

# Train model
ridge_pipeline.fit(t_clean, t_score)

# Predict and calculate MSE
pred = ridge_pipeline.predict(v_clean)
rmse = mean_squared_error(v_score, pred, squared=False)

print("RMSE Loss using Ridge and TfidfVectorizer: %0.5f" % rmse)


# ## Word Embedding Models

# ### Callbacks and Helper functions

# In[38]:


# Reduce learning rate when a metric has stopped improving. 
learning_rate_reduction = ReduceLROnPlateau(monitor='val_root_mean_squared_error', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.00001)

# Stop training when a monitored metric has stopped improving.
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)


# In[39]:


# Output helper function from https://www.kaggle.com/donmarch14/commonlit-detailed-guide-to-learn-nlp
def predict_complexity(model, excerpt):
    # Create the sequences
    padding_type='post'
    sample_sequences = tokenizer.texts_to_sequences(excerpt)
    excerpt_padded = pad_sequences(sample_sequences, padding=padding_type, 
                                 maxlen=maxlen) 
    classes = model.predict(excerpt_padded)
    for x in range(len(excerpt_padded)):
        print(excerpt[x])
        print(classes[x])
        print('\n')


# In[40]:


# Helper plotting function from https://www.kaggle.com/donmarch14/commonlit-detailed-guide-to-learn-nlp
def plot_graphs(history, string, model):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title(model)
    plt.legend([string, 'val_'+string])
    savefig = model + '_' + string + '.png'
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()


# ### Set Parameters

# In[41]:


corpus = train.clean_text


# In[42]:


# Get dictionary size
dictionary = Counter()
train['clean_text'].str.split().apply(dictionary.update)
print(len(dictionary.keys()))


# In[43]:


# Set parameters
dict_size = 17233
dim_embed = 32
maxlen = 40
trunc = 'post'
pad = 'post'
tok = "<OOV>"


# In[44]:


tokenizer = Tokenizer(num_words = dict_size, oov_token=tok)
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index


# In[45]:


train_seq = tokenizer.texts_to_sequences(corpus)
train_pad = pad_sequences(train_seq,maxlen=maxlen, 
                                truncating=trunc, padding=pad)

train_labels = np.array(train.target)


# In[46]:


print(train_pad.shape)
print(train_labels.shape)


# ### Simple Model

# In[47]:


# The Sequential class builds a stack of layers
# Embedding layer is the input layer of our word embeddings
# GlobalAveragePooling1D layer is a pooling operation for 1D data that takes the global average
# Dense(n) layer is a densely-connected NN layer with output dimensionality of n
simple_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_size, dim_embed, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),  
    tf.keras.layers.Dense(1)
])

# compile method configures the model for training
# We are using RMSE as the evaluating metric
simple_model.compile(loss='mean_squared_error', metrics=[RootMeanSquaredError()])
simple_model.summary()


# In[48]:


# Train model using 50 epochs
history = simple_model.fit(train_pad, train_labels, 
                    epochs=50, validation_split=0.1,
                   callbacks=[early_stopping,learning_rate_reduction])


# We can see from the difference in the RMSE on the train and test set the model is overfitting (0.4376 vs. 0.8697)

# ### Optimizing Model

# In[49]:


# Embedding size should be the fourth root of the dictionary size
def fourth_root(num):
    return math.sqrt(math.sqrt(num))

fourth_root(17233)


# In[50]:


dim_embed = 12


# In[51]:


excerpt_num=[]
word_count=[]
counter=1
for item in corpus:
    excerpt_num.append(counter)
    counter=counter+1
    word_count.append(len(item.split(' ')))
sorted_count = sorted(word_count)
plt.xlabel('Excerpt')
plt.ylabel('Word Count')
plt.title('Word Counts in Excerpts')
plt.plot(excerpt_num,sorted_count)

plt.show()


# In[52]:


maxlen = 130


# In[53]:


train_pad = pad_sequences(train_seq,maxlen=maxlen, 
                                truncating=trunc, padding=pad)

train_labels = np.array(train.target)


# In[54]:


print(train_pad.shape)
print(train_labels.shape)


# In[55]:


"""
Same as above but now we're using 2 dense layers.
The first two layers are the same just with our updated parameters for optimization
The first Dense layer now has 32 units or nodes and uses ReLU as the activation function. 
We're also using L2 regularization as is common with NLP problems
"""
opt_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_size, dim_embed, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])
opt_model.compile(loss='mean_squared_error',optimizer='adam', metrics=[RootMeanSquaredError()])
opt_model.summary()


# In[56]:


history = opt_model.fit(train_pad, train_labels, 
                    epochs=100, validation_split=0.1,
                    callbacks=[early_stopping,learning_rate_reduction])


# In[57]:


predict_complexity(opt_model, test['excerpt'])
plot_graphs(history, "root_mean_squared_error", 'opt_model')
plot_graphs(history, "loss", 'opt_model')
# model is still overfitting


# ### Convolutional Neural Network (CNN/ConvNet)

# In[58]:


"""
Same as above but with an additional 1D convolution layer
"""
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_size, dim_embed, input_length=maxlen),
    tf.keras.layers.Conv1D(dim_embed, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(), 
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])

# Default learning rate for the Adam optimizer is 0.001
# Let's slow down the learning rate by 10.
learning_rate = 0.0001
cnn_model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[RootMeanSquaredError()])
cnn_model.summary()


# In[59]:


history = cnn_model.fit(train_pad, train_labels, 
                    epochs=100, validation_split=0.1,
                    callbacks=[early_stopping,learning_rate_reduction])


# In[60]:


predict_complexity(cnn_model, test['excerpt'])
plot_graphs(history, "root_mean_squared_error", 'cnn_model')
plot_graphs(history, "loss", 'cnn_model')
# model is still overfitting


# In[61]:


def build_model(hyper):
    model = keras.Sequential()
    model.add(tf.keras.layers.Embedding(dict_size, dim_embed, input_length=maxlen))

    # Find optimal number of units in dense layer between 16 and 256
    opt_units = hyper.Int('units', min_value=16, max_value=256, step=8)
    model.add(keras.layers.Dense(units=opt_units, activation='relu'))
    model.add(keras.layers.Dense(1))


    grid_lr = hyper.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=grid_lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=[RootMeanSquaredError()])

    return model


# In[62]:


tuner = kt.Hyperband(build_model,
                     max_epochs=10,
                     objective = kt.Objective("val_root_mean_squared_error", direction="min"),
                     factor=5,
                     directory='my_dir',
                     project_name='intro_to_kt')


# In[63]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# In[64]:


tuner.search(train_pad, train_labels, epochs=5, validation_split=0.1, callbacks=[early_stop])

# Get the optimal hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]


# In[65]:


print(f"""Hyperband search complete. \nOptimal units of first dense layer: {best_hyperparameters.get('units')} \nOptimal learning rate: {best_hyperparameters.get('learning_rate')}.""")


# In[66]:


"""
Model with best hyperparameters
"""
hyper_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_size, dim_embed, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(160, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])
hyper_model.compile(loss='mean_squared_error',optimizer='adam', metrics=[RootMeanSquaredError()])
hyper_model.summary()

learning_rate = 0.001
hyper_model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[RootMeanSquaredError()])
hyper_model.summary()


# In[67]:


history = hyper_model.fit(train_pad, train_labels, 
                    epochs=100, validation_split=0.1,
                    callbacks=[early_stopping,learning_rate_reduction])


# In[68]:


predict_complexity(hyper_model, test['excerpt'])
plot_graphs(history, "root_mean_squared_error", "hyper_model")
plot_graphs(history, "loss", "hyper_model")

